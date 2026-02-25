"""
Agent Manager Service

Manages on-demand spawning, execution, and lifecycle of agents.
Replaces the always-running agent model with on-demand spawning.
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Any
import httpx
import psutil

from backend.services.agent_registry_service import agent_registry

logger = logging.getLogger("AgentManager")

DEFAULT_AGENT_EXECUTE_TIMEOUT = 120.0
AGENT_EXECUTE_TIMEOUTS = {
    "document": 240.0,
    "document_agent": 240.0,
    "spreadsheet": 180.0,
    "spreadsheet_agent": 180.0,
    "browser": 300.0,
    "browser_agent": 300.0,
    "universal": 120.0,
    "universal_agent": 120.0,
    "mail": 120.0,
    "mail_agent": 120.0,
    "zoho_books": 180.0,
    "coding": 180.0,
    "coding_agent": 180.0,
}


def _get_agent_timeout(agent_id: str) -> float:
    env_key = f"AGENT_TIMEOUT_{agent_id.upper()}"
    if env_key in os.environ:
        try:
            return float(os.environ[env_key])
        except ValueError:
            logger.warning(f"Invalid {env_key} value: {os.environ[env_key]}")
    return AGENT_EXECUTE_TIMEOUTS.get(agent_id, DEFAULT_AGENT_EXECUTE_TIMEOUT)


@dataclass
class AgentInstance:
    """Represents a running agent instance."""

    agent_id: str
    process: subprocess.Popen
    port: int
    pid: int
    start_time: float
    last_used: float = field(default_factory=time.time)
    healthy: bool = False
    workspace_path: Optional[Path] = None


class PortPool:
    """
    Manages port allocation for agents.
    Uses default ports when available, falls back to dynamic pool.
    """

    DEFAULT_PORTS = {
        "browser": 8090,
        "browser_agent": 8090,
        "spreadsheet": 9000,
        "spreadsheet_agent": 9000,
        "mail": 8040,
        "mail_agent": 8040,
        "gmail": 8003,
        "gmail_agent": 8003,
        "document": 8050,
        "document_agent": 8050,
        "zoho_books": 8060,
        "universal": 8070,
        "universal_agent": 8070,
        "coding": 8080,
        "coding_agent": 8080,
    }

    DYNAMIC_RANGE = (9001, 9100)  # Ports 9001-9100 for dynamic allocation

    def __init__(self):
        self.allocated_ports: Dict[str, int] = {}  # agent_id -> port
        self.used_ports: Set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self, agent_id: str) -> int:
        """Allocate a port for an agent."""
        async with self._lock:
            # Check if already allocated
            if agent_id in self.allocated_ports:
                return self.allocated_ports[agent_id]

            # Try default port first
            default_port = self.DEFAULT_PORTS.get(agent_id)
            if default_port and default_port not in self.used_ports:
                if not self._is_port_in_use(default_port):
                    self.allocated_ports[agent_id] = default_port
                    self.used_ports.add(default_port)
                    logger.info(f"Allocated default port {default_port} to {agent_id}")
                    return default_port

            # Find available port in dynamic range
            for port in range(self.DYNAMIC_RANGE[0], self.DYNAMIC_RANGE[1]):
                if port not in self.used_ports and not self._is_port_in_use(port):
                    self.allocated_ports[agent_id] = port
                    self.used_ports.add(port)
                    logger.info(f"Allocated dynamic port {port} to {agent_id}")
                    return port

            raise RuntimeError(f"No available ports for agent {agent_id}")

    async def release(self, agent_id: str):
        """Release a port back to the pool."""
        async with self._lock:
            if agent_id in self.allocated_ports:
                port = self.allocated_ports.pop(agent_id)
                self.used_ports.discard(port)
                logger.info(f"Released port {port} from {agent_id}")

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use by another process."""
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                return result == 0  # Port is in use
        except Exception:
            return True  # Assume in use on error


class ProcessManager:
    """Manages agent subprocesses."""

    AGENT_MODULE_MAP = {
        "browser": "agents.browser_agent",
        "browser_agent": "agents.browser_agent",
        "browser_automation_agent": "agents.browser_agent",
        "spreadsheet": "agents.spreadsheet_agent",
        "spreadsheet_agent": "agents.spreadsheet_agent",
        "mail": "agents.mail_agent",
        "mail_agent": "agents.mail_agent",
        "gmail": "agents.gmail_agent",
        "gmail_agent": "agents.gmail_agent",
        "document": "agents.document_agent_lib",
        "document_agent": "agents.document_agent_lib",
        "zoho_books": "agents.zoho_books",
        "universal": "agents.universal_agent",
        "universal_agent": "agents.universal_agent",
        "coding": "agents.coding_agent",
        "coding_agent": "agents.coding_agent",
    }

    # Special cases: agents that don't use __init__.py pattern
    AGENT_STARTUP_FILES = {
        "zoho_books": "zoho_books_agent.py",
    }

    def __init__(self, backend_dir: Path):
        self.backend_dir = backend_dir

    async def start_agent(self, agent_id: str, port: int) -> subprocess.Popen:
        """
        Start an agent subprocess.

        Returns the process handle.
        """
        module_path = self.AGENT_MODULE_MAP.get(agent_id)
        if not module_path:
            raise ValueError(f"Unknown agent: {agent_id}")

        # Determine how to start the agent
        agent_dir = self.backend_dir / module_path.replace(".", "/")

        # Check for different startup patterns
        if (agent_dir / "__init__.py").exists():
            # FastAPI pattern: uvicorn __init__:app
            cmd = [
                "python",
                "-m",
                "uvicorn",
                f"{module_path}.__init__:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ]
        elif (agent_dir / "agent.py").exists():
            # Direct script pattern
            cmd = [
                "python",
                "-m",
                module_path,
                "--port",
                str(port),
            ]
        elif agent_id in self.AGENT_STARTUP_FILES:
            # Special case: agent has different startup file
            startup_file = self.AGENT_STARTUP_FILES[agent_id]
            if (agent_dir / startup_file).exists():
                # Run the specific file with uvicorn
                module_name = startup_file.replace(".py", "")
                cmd = [
                    "python",
                    "-m",
                    "uvicorn",
                    f"{module_path}.{module_name}:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--log-level",
                    "warning",
                ]
            else:
                raise RuntimeError(
                    f"Startup file {startup_file} not found for {agent_id}"
                )
        else:
            raise RuntimeError(f"Cannot find startup script for {agent_id}")

        # Set environment variables
        env = dict(os.environ)
        env["AGENT_PORT"] = str(port)
        env["AGENT_ID"] = agent_id

        # Fix PYTHONPATH so agents can import from backend.*
        # Add parent directory of backend to PYTHONPATH
        parent_dir = str(self.backend_dir.parent)
        current_pythonpath = env.get("PYTHONPATH", "")
        if current_pythonpath:
            env["PYTHONPATH"] = f"{parent_dir}{os.pathsep}{current_pythonpath}"
        else:
            env["PYTHONPATH"] = parent_dir

        logger.info(f"Starting agent {agent_id} on port {port}")
        logger.debug(f"PYTHONPATH: {env['PYTHONPATH']}")
        logger.debug(f"Command: {' '.join(cmd)}")

        logs_dir = self.backend_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{agent_id}.log"
        try:
            log_file = open(log_path, "a", encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to open log file {log_path}: {e}")
            log_file = subprocess.DEVNULL

        # Start process
        process = subprocess.Popen(
            cmd,
            cwd=str(self.backend_dir),
            env=env,
            stdout=log_file,
            stderr=log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        setattr(process, "_log_file", log_file)

        logger.info(f"Agent {agent_id} started with PID {process.pid}")
        return process

    async def stop_agent(self, process: subprocess.Popen, timeout: int = 10) -> bool:
        """Gracefully stop an agent process."""
        if process.poll() is not None:
            log_file = getattr(process, "_log_file", None)
            if log_file and log_file is not subprocess.DEVNULL:
                try:
                    log_file.close()
                except Exception:
                    pass
            # Already stopped
            return True

        try:
            # Try graceful termination first
            if os.name == "nt":
                process.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()

            # Wait for process to exit
            try:
                process.wait(timeout=timeout)
                logger.info(f"Agent process {process.pid} terminated gracefully")
                log_file = getattr(process, "_log_file", None)
                if log_file and log_file is not subprocess.DEVNULL:
                    try:
                        log_file.close()
                    except Exception:
                        pass
                return True
            except subprocess.TimeoutExpired:
                # Force kill
                process.kill()
                process.wait()
                logger.warning(f"Agent process {process.pid} killed forcefully")
                log_file = getattr(process, "_log_file", None)
                if log_file and log_file is not subprocess.DEVNULL:
                    try:
                        log_file.close()
                    except Exception:
                        pass
                return True
        except Exception as e:
            logger.error(f"Error stopping agent process: {e}")
            return False

    def is_running(self, process: subprocess.Popen) -> bool:
        """Check if a process is still running."""
        return process.poll() is None


class HealthChecker:
    """Checks agent health before use."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.check_interval = 0.5  # Check every 500ms

    async def wait_for_ready(self, port: int, agent_id: str) -> bool:
        """
        Poll /health endpoint until agent is ready.

        Returns True if healthy, False if timeout.
        """
        start_time = time.time()
        url = f"http://localhost:{port}/health"

        logger.info(f"Waiting for agent on port {port} to be ready...")

        async with httpx.AsyncClient() as client:
            while time.time() - start_time < self.timeout:
                try:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        logger.info(f"Agent on port {port} is healthy")
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Not ready yet, wait and retry
                    pass
                except Exception as e:
                    logger.debug(f"Health check error: {e}")

                await asyncio.sleep(self.check_interval)

        logger.error(f"Agent on port {port} failed health check after {self.timeout}s")
        return False

    async def check_health(self, port: int) -> bool:
        """Quick health check."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/health", timeout=5.0
                )
                return response.status_code == 200
        except Exception:
            return False


class AutoTerminator:
    """Automatically terminates idle agents."""

    def __init__(self, agent_manager: "AgentManager", idle_timeout: int = 300):
        self.agent_manager = agent_manager
        self.idle_timeout = idle_timeout  # seconds
        self._monitoring = False
        self._task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start the background monitoring task."""
        if self._monitoring:
            return

        self._monitoring = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"AutoTerminator started (idle_timeout={self.idle_timeout}s)")

    async def stop_monitoring(self):
        """Stop the background monitoring task."""
        self._monitoring = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AutoTerminator stopped")

    async def _monitor_loop(self):
        """Background loop that checks for idle agents."""
        check_interval = 60  # Check every minute

        while self._monitoring:
            try:
                await asyncio.sleep(check_interval)
                await self._check_idle_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in AutoTerminator loop: {e}")

    async def _check_idle_agents(self):
        """Check for and terminate idle agents."""
        current_time = time.time()
        agents_to_terminate = []

        for agent_id, instance in self.agent_manager.active_agents.items():
            idle_time = current_time - instance.last_used
            if idle_time > self.idle_timeout:
                agents_to_terminate.append(agent_id)
                logger.info(
                    f"Agent {agent_id} idle for {idle_time:.0f}s, "
                    f"marking for termination"
                )

        for agent_id in agents_to_terminate:
            try:
                await self.agent_manager.terminate_agent(agent_id)
            except Exception as e:
                logger.error(f"Error terminating idle agent {agent_id}: {e}")


class AgentManager:
    """
    Main service for managing agent lifecycle.

    Usage:
        agent_manager = AgentManager()
        await agent_manager.initialize()

        # Execute task (spawns agent automatically if needed)
        result = await agent_manager.execute('browser', task)

        # Cleanup
        await agent_manager.shutdown()
    """

    def __init__(self, backend_dir: Optional[Path] = None):
        self.backend_dir = backend_dir or Path(__file__).parent.parent
        self.active_agents: Dict[str, AgentInstance] = {}

        # Components
        self.port_pool = PortPool()
        self.process_manager = ProcessManager(self.backend_dir)
        self.health_checker = HealthChecker()
        self.auto_terminator = AutoTerminator(self, idle_timeout=300)

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the agent manager."""
        if self._initialized:
            return

        # Start auto-terminator
        await self.auto_terminator.start_monitoring()

        self._initialized = True
        logger.info("AgentManager initialized")

    async def shutdown(self):
        """Shutdown the agent manager and terminate all agents."""
        logger.info("Shutting down AgentManager...")

        # Stop auto-terminator
        await self.auto_terminator.stop_monitoring()

        # Terminate all agents
        await self.terminate_all()

        self._initialized = False
        logger.info("AgentManager shutdown complete")

    async def spawn_agent(self, agent_id: str) -> AgentInstance:
        """
        Spawn an agent if not already running.

        Returns the agent instance (existing or newly created).
        """
        async with self._lock:
            # Check if already active
            if agent_id in self.active_agents:
                instance = self.active_agents[agent_id]
                if self.process_manager.is_running(instance.process):
                    # Update last used time
                    instance.last_used = time.time()
                    logger.debug(f"Reusing existing agent {agent_id}")
                    return instance
                else:
                    # Process died, remove it
                    logger.warning(f"Agent {agent_id} process died, respawning")
                    await self._cleanup_agent(agent_id)

            # Allocate port
            port = await self.port_pool.allocate(agent_id)

            try:
                # Start agent process
                process = await self.process_manager.start_agent(agent_id, port)

                # Create instance
                instance = AgentInstance(
                    agent_id=agent_id,
                    process=process,
                    port=port,
                    pid=process.pid,
                    start_time=time.time(),
                    healthy=False,
                )

                # Wait for health check
                healthy = await self.health_checker.wait_for_ready(port, agent_id)
                if not healthy:
                    raise RuntimeError(f"Agent {agent_id} failed health check")

                instance.healthy = True
                self.active_agents[agent_id] = instance

                logger.info(
                    f"Agent {agent_id} spawned successfully "
                    f"(PID: {process.pid}, Port: {port})"
                )

                return instance

            except Exception as e:
                # Cleanup on failure
                await self.port_pool.release(agent_id)
                if "process" in locals():
                    await self.process_manager.stop_agent(process)
                raise

    async def execute(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task on an agent (spawns if needed).

        Args:
            agent_id: The agent to use
            task: Task dictionary with 'prompt', 'action', 'payload', etc.

        Returns:
            Agent response dictionary
        """
        if not self._initialized:
            await self.initialize()

        # Ensure agent is running
        instance = await self.spawn_agent(agent_id)

        # Prepare request
        url = f"http://localhost:{instance.port}/execute"

        # Standard UAP format
        uap_request = {
            "type": "execute",
            "prompt": task.get("prompt", ""),
            "action": task.get("action"),
            "payload": task.get("payload", {}),
            "task_id": task.get("task_id"),
            "thread_id": task.get("thread_id"),
        }

        timeout = _get_agent_timeout(agent_id)
        logger.info(f"Executing task on {agent_id} (port {instance.port}, timeout={timeout}s)")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=uap_request,
                    timeout=timeout,
                    headers={
                        "Content-Type": "application/json",
                        "X-User-ID": task.get("user_id", "anonymous"),
                    },
                )

                response.raise_for_status()
                result = response.json()

                # Update last used time
                instance.last_used = time.time()

                logger.info(f"Task completed on {agent_id}")
                return result

        except httpx.TimeoutException:
            logger.error(f"Timeout executing task on {agent_id}")
            return {
                "status": "error",
                "error": f"Agent {agent_id} timed out after {timeout:.0f}s",
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from {agent_id}: {e.response.status_code}")
            return {
                "status": "error",
                "error": f"Agent {agent_id} returned error: {e.response.status_code}",
            }
        except Exception as e:
            logger.error(f"Error executing task on {agent_id}: {e}")
            return {
                "status": "error",
                "error": f"Failed to execute on {agent_id}: {str(e)}",
            }

    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent process.

        Returns True if terminated successfully.
        """
        async with self._lock:
            if agent_id not in self.active_agents:
                return False

            instance = self.active_agents[agent_id]
            logger.info(f"Terminating agent {agent_id} (PID: {instance.pid})")

            # Stop process
            success = await self.process_manager.stop_agent(instance.process)

            # Cleanup
            await self._cleanup_agent(agent_id)

            return success

    async def terminate_all(self):
        """Terminate all active agents."""
        logger.info(f"Terminating all {len(self.active_agents)} active agents")

        # Copy keys to avoid modification during iteration
        agent_ids = list(self.active_agents.keys())

        for agent_id in agent_ids:
            try:
                await self.terminate_agent(agent_id)
            except Exception as e:
                logger.error(f"Error terminating agent {agent_id}: {e}")

    async def _cleanup_agent(self, agent_id: str):
        """Clean up agent resources."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
        await self.port_pool.release(agent_id)

    def get_active_agents(self) -> Dict[str, AgentInstance]:
        """Get dictionary of active agents."""
        return dict(self.active_agents)

    def is_agent_active(self, agent_id: str) -> bool:
        """Check if an agent is currently active."""
        if agent_id not in self.active_agents:
            return False
        instance = self.active_agents[agent_id]
        return self.process_manager.is_running(instance.process)


# Singleton instance
_agent_manager: Optional[AgentManager] = None


def get_agent_manager() -> AgentManager:
    """Get the singleton agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager

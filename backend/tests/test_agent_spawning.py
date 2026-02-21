"""
Orchestrator Agent Spawning Tests

Tests the orchestrator's ability to:
1. Spawn individual agents (sequential)
2. Spawn multiple agents in parallel
3. Run multiple instances of the same agent
4. Execute tasks through the full orchestrator pipeline

Run:
    cd d:\\Internship\\Orbimesh\\backend
    python tests/test_agent_spawning.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import asyncio
import logging
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpawnTest")


# ============================================================================
# HELPERS
# ============================================================================

AGENTS_TO_TEST = ["universal", "spreadsheet", "mail", "document", "coding"]
# NOTE: browser excluded from default tests because it requires Playwright browser install

# Simple health check payloads (don't require real work, just verify agent responds)
AGENT_HEALTH_PROMPTS = {
    "universal": {"prompt": "Return the text: HEALTH_OK", "action": "execute_task"},
    "spreadsheet": {"prompt": "Respond with OK", "action": "execute"},
    "mail": {"prompt": "Respond with OK", "action": "execute"},
    "document": {"prompt": "Respond with OK", "action": "execute"},
    "coding": {"prompt": "Return the text: HEALTH_OK", "action": "execute"},
}


async def spawn_agent_directly(agent_id: str, port: int) -> Optional[subprocess.Popen]:
    """Spawn an agent subprocess directly (bypassing AgentManager for isolation)."""
    from services.agent_manager import ProcessManager

    pm = ProcessManager(ROOT)
    try:
        process = await pm.start_agent(agent_id, port)
        logger.info(f"  Started {agent_id} on port {port}, PID={process.pid}")
        return process
    except Exception as e:
        logger.error(f"  Failed to start {agent_id}: {e}")
        return None


async def wait_for_health(port: int, agent_id: str, timeout: int = 30) -> bool:
    """Poll /health endpoint until agent is ready."""
    import httpx

    start = time.time()
    url = f"http://localhost:{port}/health"
    while time.time() - start < timeout:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=3.0)
                if resp.status_code == 200:
                    logger.info(f"  {agent_id} healthy on port {port} ({time.time()-start:.1f}s)")
                    return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    logger.error(f"  {agent_id} health check timed out after {timeout}s")
    return False


async def execute_on_agent(port: int, agent_id: str, task: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Send a task to a running agent via HTTP POST."""
    import httpx

    url = f"http://localhost:{port}/execute"
    uap_request = {
        "type": "execute",
        "prompt": task.get("prompt", ""),
        "action": task.get("action"),
        "payload": task.get("payload", {}),
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=uap_request, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def kill_process(process: subprocess.Popen, agent_id: str):
    """Kill a spawned process."""
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            process.kill()
        logger.info(f"  Terminated {agent_id} (PID={process.pid})")


async def kill_port(port: int):
    """Kill any process on the given port."""
    import psutil
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            try:
                psutil.Process(conn.pid).terminate()
            except Exception:
                pass


# ============================================================================
# TEST 1: Sequential Agent Spawning
# ============================================================================

async def test_sequential_spawning():
    """Test spawning agents one by one, verifying health for each."""
    print("\n" + "=" * 72)
    print("  TEST 1: SEQUENTIAL AGENT SPAWNING")
    print("=" * 72)

    results = {}
    base_port = 10100  # Use high ports to avoid conflicts with running services

    for i, agent_id in enumerate(AGENTS_TO_TEST):
        port = base_port + i
        await kill_port(port)

        print(f"\n  [{i+1}/{len(AGENTS_TO_TEST)}] Spawning {agent_id} on port {port}...")
        process = await spawn_agent_directly(agent_id, port)
        if not process:
            results[agent_id] = {"spawned": False, "healthy": False, "time": 0}
            continue

        start = time.time()
        healthy = await wait_for_health(port, agent_id, timeout=30)
        spawn_time = time.time() - start

        results[agent_id] = {
            "spawned": True,
            "healthy": healthy,
            "time": round(spawn_time, 1),
            "pid": process.pid,
        }

        status = "PASS" if healthy else "FAIL"
        print(f"  {status}: {agent_id} - spawned={True}, healthy={healthy}, time={spawn_time:.1f}s")

        # Cleanup
        await kill_process(process, agent_id)
        await asyncio.sleep(0.5)  # Let port release

    print(f"\n  Summary: {sum(1 for r in results.values() if r['healthy'])}/{len(results)} agents spawned successfully")
    return results


# ============================================================================
# TEST 2: Parallel Agent Spawning
# ============================================================================

async def test_parallel_spawning():
    """Test spawning multiple agents at the same time."""
    print("\n" + "=" * 72)
    print("  TEST 2: PARALLEL AGENT SPAWNING")
    print("=" * 72)

    base_port = 10200
    processes = {}

    # Kill any existing processes on target ports
    for i in range(len(AGENTS_TO_TEST)):
        await kill_port(base_port + i)

    print(f"\n  Spawning {len(AGENTS_TO_TEST)} agents simultaneously...")
    start = time.time()

    # Spawn all agents in parallel
    spawn_tasks = []
    for i, agent_id in enumerate(AGENTS_TO_TEST):
        port = base_port + i
        spawn_tasks.append(spawn_agent_directly(agent_id, port))

    spawned = await asyncio.gather(*spawn_tasks) 
    spawn_time = time.time() - start

    for i, (agent_id, process) in enumerate(zip(AGENTS_TO_TEST, spawned)):
        if process:
            processes[agent_id] = {"process": process, "port": base_port + i}

    print(f"  Spawned {len(processes)}/{len(AGENTS_TO_TEST)} agents in {spawn_time:.1f}s")

    # Wait for all to become healthy (in parallel)
    print(f"  Waiting for health checks...")
    health_start = time.time()

    health_tasks = [
        wait_for_health(info["port"], agent_id, timeout=30)
        for agent_id, info in processes.items()
    ]
    health_results = await asyncio.gather(*health_tasks)
    health_time = time.time() - health_start

    results = {}
    for (agent_id, info), healthy in zip(processes.items(), health_results):
        results[agent_id] = {
            "spawned": True,
            "healthy": healthy,
            "pid": info["process"].pid,
            "port": info["port"],
        }
        status = "PASS" if healthy else "FAIL"
        print(f"  {status}: {agent_id} (port={info['port']}, pid={info['process'].pid})")

    total_time = time.time() - start
    healthy_count = sum(1 for r in results.values() if r["healthy"])
    print(f"\n  Summary: {healthy_count}/{len(results)} healthy, total time: {total_time:.1f}s")

    # Cleanup
    for agent_id, info in processes.items():
        await kill_process(info["process"], agent_id)

    results["_meta"] = {"total_time": round(total_time, 1), "health_time": round(health_time, 1)}
    return results


# ============================================================================
# TEST 3: Multiple Instances of Same Agent
# ============================================================================

async def test_multi_instance():
    """Test spawning 3 instances of the universal agent on different ports."""
    print("\n" + "=" * 72)
    print("  TEST 3: MULTIPLE INSTANCES OF SAME AGENT")
    print("=" * 72)

    agent_id = "universal"
    ports = [10300, 10301, 10302]
    instances = []

    for port in ports:
        await kill_port(port)

    print(f"\n  Spawning 3 instances of '{agent_id}' on ports {ports}...")
    start = time.time()

    # Spawn all 3 in parallel
    spawn_tasks = [spawn_agent_directly(agent_id, port) for port in ports]
    processes = await asyncio.gather(*spawn_tasks)
    spawn_time = time.time() - start

    for i, (port, process) in enumerate(zip(ports, processes)):
        if process:
            instances.append({"port": port, "process": process, "pid": process.pid})
            print(f"  Instance {i+1}: PID={process.pid}, port={port}")
        else:
            print(f"  Instance {i+1}: FAILED to spawn on port {port}")

    # Health check all instances
    print(f"\n  Health checking {len(instances)} instances...")
    health_tasks = [wait_for_health(inst["port"], f"{agent_id}_{i}") for i, inst in enumerate(instances)]
    health_results = await asyncio.gather(*health_tasks)

    results = []
    for inst, healthy in zip(instances, health_results):
        inst["healthy"] = healthy
        results.append(inst)
        status = "PASS" if healthy else "FAIL"
        print(f"  {status}: Instance port={inst['port']}, pid={inst['pid']}")

    total_time = time.time() - start
    healthy_count = sum(1 for r in results if r["healthy"])
    print(f"\n  Summary: {healthy_count}/{len(results)} instances healthy, time: {total_time:.1f}s")

    # Verify instances are truly independent (different PIDs)
    pids = [r["pid"] for r in results]
    unique_pids = len(set(pids))
    print(f"  Unique PIDs: {unique_pids}/{len(pids)} (should all be unique)")
    if unique_pids == len(pids):
        print(f"  PASS: All instances are independent processes")
    else:
        print(f"  FAIL: Some instances share PIDs!")

    # Cleanup
    for inst in instances:
        await kill_process(inst["process"], f"{agent_id}_{inst['port']}")

    return {
        "instances_spawned": len(instances),
        "instances_healthy": healthy_count,
        "unique_pids": unique_pids,
        "total_time": round(total_time, 1),
    }


# ============================================================================
# TEST 4: AgentManager Integration (Full Lifecycle)
# ============================================================================

async def test_agent_manager_lifecycle():
    """Test the full AgentManager lifecycle: init -> spawn -> execute -> terminate."""
    print("\n" + "=" * 72)
    print("  TEST 4: AGENT MANAGER FULL LIFECYCLE")
    print("=" * 72)

    from services.agent_manager import AgentManager

    manager = AgentManager(backend_dir=ROOT)
    results = {}

    try:
        print("\n  Initializing AgentManager...")
        await manager.initialize()
        print("  PASS: AgentManager initialized")

        # Spawn and execute on universal agent
        agent_id = "universal"
        print(f"\n  Spawning {agent_id} via AgentManager...")
        start = time.time()

        instance = await manager.spawn_agent(agent_id)
        spawn_time = time.time() - start
        print(f"  PASS: Spawned {agent_id} (PID={instance.pid}, port={instance.port}, time={spawn_time:.1f}s)")

        active = manager.get_active_agents()
        print(f"  Active agents: {list(active.keys())}")

        # Verify agent is active
        is_active = manager.is_agent_active(agent_id)
        print(f"  is_agent_active('{agent_id}'): {is_active}")

        # Terminate
        print(f"\n  Terminating {agent_id}...")
        terminated = await manager.terminate_agent(agent_id)
        print(f"  {'PASS' if terminated else 'FAIL'}: terminate_agent returned {terminated}")

        active_after = manager.get_active_agents()
        print(f"  Active agents after terminate: {list(active_after.keys())}")

        results = {
            "init": True,
            "spawn": True,
            "spawn_time": round(spawn_time, 1),
            "is_active": is_active,
            "terminated": terminated,
            "cleaned_up": len(active_after) == 0,
        }

    except Exception as e:
        print(f"  FAIL: {e}")
        results["error"] = str(e)
    finally:
        await manager.shutdown()
        print("  AgentManager shutdown complete")

    return results


# ============================================================================
# TEST 5: Parallel Execution via Hands._execute_parallel
# ============================================================================

async def test_parallel_execution_via_hands():
    """Test the Hands._execute_parallel method with multiple agent actions."""
    print("\n" + "=" * 72)
    print("  TEST 5: PARALLEL EXECUTION PATTERN (Simulated)")
    print("=" * 72)

    # This test validates the parallel dispatch pattern from Hands
    # We simulate what _execute_parallel does: asyncio.gather on multiple agents

    from services.agent_manager import AgentManager

    manager = AgentManager(backend_dir=ROOT)
    await manager.initialize()

    agents = ["universal", "spreadsheet"]
    results = {}

    try:
        print(f"\n  Spawning {len(agents)} agents in parallel for execution test...")

        # Spawn both in parallel (what _execute_parallel does internally)
        spawn_tasks = [manager.spawn_agent(aid) for aid in agents]
        instances = await asyncio.gather(*spawn_tasks, return_exceptions=True)

        spawned = []
        for agent_id, inst in zip(agents, instances):
            if isinstance(inst, Exception):
                print(f"  FAIL: {agent_id} spawn failed: {inst}")
            else:
                print(f"  PASS: {agent_id} spawned (port={inst.port}, pid={inst.pid})")
                spawned.append(agent_id)

        # Now execute on all spawned agents in parallel
        if spawned:
            print(f"\n  Executing tasks on {len(spawned)} agents in parallel...")
            start = time.time()

            exec_tasks = []
            for agent_id in spawned:
                inst = manager.active_agents[agent_id]
                exec_tasks.append(
                    execute_on_agent(inst.port, agent_id, AGENT_HEALTH_PROMPTS.get(agent_id, {"prompt": "OK"}), timeout=60.0)
                )

            exec_results = await asyncio.gather(*exec_tasks, return_exceptions=True)
            exec_time = time.time() - start

            for agent_id, result in zip(spawned, exec_results):
                if isinstance(result, Exception):
                    print(f"  FAIL: {agent_id} execution exception: {result}")
                    results[agent_id] = {"executed": False, "error": str(result)}
                else:
                    success = result.get("status") != "error"
                    print(f"  {'PASS' if success else 'FAIL'}: {agent_id} execution {'succeeded' if success else 'failed'}")
                    results[agent_id] = {"executed": success, "result_status": result.get("status", "unknown")}

            print(f"\n  Parallel execution time: {exec_time:.1f}s")
            results["_meta"] = {"exec_time": round(exec_time, 1)}

    except Exception as e:
        print(f"  FAIL: {e}")
        results["error"] = str(e)
    finally:
        await manager.shutdown()

    return results


# ============================================================================
# RUNNER
# ============================================================================

async def run_all_tests():
    """Run all orchestrator spawning tests."""
    all_results = {}
    test_pass_count = 0
    test_total = 5

    # Test 1: Sequential
    try:
        r = await test_sequential_spawning()
        healthy = sum(1 for k, v in r.items() if isinstance(v, dict) and v.get("healthy"))
        all_results["sequential"] = {"healthy": healthy, "total": len(AGENTS_TO_TEST)}
        if healthy > 0:
            test_pass_count += 1
    except Exception as e:
        print(f"  FATAL: {e}")
        all_results["sequential"] = {"error": str(e)}

    await asyncio.sleep(2)  # Let ports release

    # Test 2: Parallel
    try:
        r = await test_parallel_spawning()
        healthy = sum(1 for k, v in r.items() if isinstance(v, dict) and v.get("healthy"))
        all_results["parallel"] = {"healthy": healthy, "total": len(AGENTS_TO_TEST)}
        if healthy > 0:
            test_pass_count += 1
    except Exception as e:
        print(f"  FATAL: {e}")
        all_results["parallel"] = {"error": str(e)}

    await asyncio.sleep(2)

    # Test 3: Multi-instance
    try:
        r = await test_multi_instance()
        all_results["multi_instance"] = r
        if r.get("instances_healthy", 0) > 0:
            test_pass_count += 1
    except Exception as e:
        print(f"  FATAL: {e}")
        all_results["multi_instance"] = {"error": str(e)}

    await asyncio.sleep(2)

    # Test 4: AgentManager lifecycle
    try:
        r = await test_agent_manager_lifecycle()
        all_results["lifecycle"] = r
        if r.get("spawn") and r.get("terminated"):
            test_pass_count += 1
    except Exception as e:
        print(f"  FATAL: {e}")
        all_results["lifecycle"] = {"error": str(e)}

    await asyncio.sleep(2)

    # Test 5: Parallel execution
    try:
        r = await test_parallel_execution_via_hands()
        all_results["parallel_execution"] = r
        executed = sum(1 for k, v in r.items() if isinstance(v, dict) and v.get("executed"))
        if executed > 0:
            test_pass_count += 1
    except Exception as e:
        print(f"  FATAL: {e}")
        all_results["parallel_execution"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS: {test_pass_count}/{test_total} test groups passed")
    print("=" * 72)
    for name, result in all_results.items():
        status = "PASS" if not result.get("error") else "FAIL"
        print(f"  {status}: {name} - {json.dumps(result, default=str)[:100]}")
    print("=" * 72 + "\n")

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())

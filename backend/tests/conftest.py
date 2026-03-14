"""
Shared pytest fixtures for all agent tests.

Provides:
- agent_server: starts an agent subprocess, waits for /health, tears down after session
- http_client: async httpx.AsyncClient
- execute_agent: helper coroutine for POST /execute
- sample_csv / sample_xlsx / sample_docx / sample_pdf: paths to test data
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import httpx
import pytest
import pytest_asyncio

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).parent.parent                        # backend/
PROJECT_ROOT = BACKEND_DIR.parent                                 # Orbimesh/
TEST_DATA_DIR = Path(__file__).parent / "test_data"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# ── Agent default ports (mirrors agent_manager.py PortPool) ───────────────────
AGENT_PORTS = {
    "spreadsheet_agent":  9000,
    "mail_agent":         8040,
    "document_agent":     8050,
    "universal_agent":    8070,
    "zoho_books":         8060,
    "integrations_agent": 8075,
}

AGENT_MODULE_MAP = {
    "spreadsheet_agent":  "agents.spreadsheet_agent",
    "mail_agent":         "agents.mail_agent",
    "document_agent":     "agents.document_agent",
    "universal_agent":    "agents.universal_agent",
    "zoho_books":         "agents.zoho_books",
    "integrations_agent": "agents.integrations_agent",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_env() -> dict:
    """Build environment for agent subprocess with correct PYTHONPATH."""
    env = dict(os.environ)
    python_path = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{python_path}{os.pathsep}{existing}" if existing else python_path
    return env


def _is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


def _wait_for_health(port: int, timeout: int = 40) -> bool:
    """Poll GET /health until agent is ready or timeout expires."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# ── Core fixture: agent subprocess ────────────────────────────────────────────

@pytest.fixture(scope="session")
def agent_server(request):
    """
    Parametrized fixture factory: starts an agent subprocess for the session.

    Usage in test files:
        @pytest.fixture(scope="session")
        def spreadsheet_server():
            yield from _start_agent("spreadsheet_agent")

    Or call start_agent() directly if your test module defines its own fixture.
    """
    # Not directly used — agents define their own session-level fixture below.
    yield None


def start_agent(agent_id: str) -> Generator:
    """
    Generator helper used by per-agent fixtures.
    Starts the agent if not already running, yields port, then stops it.
    """
    port = AGENT_PORTS[agent_id]
    module = AGENT_MODULE_MAP[agent_id]
    already_running = _is_port_in_use(port)
    process: Optional[subprocess.Popen] = None

    if not already_running:
        cmd = [
            sys.executable, "-m", "uvicorn",
            f"{module}.__init__:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--log-level", "warning",
        ]
        env = _build_env()
        env["AGENT_PORT"] = str(port)
        env["AGENT_ID"] = agent_id

        process = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

        ready = _wait_for_health(port, timeout=60)
        if not ready:
            process.kill()
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(
                f"Agent '{agent_id}' failed to start on port {port}.\n"
                f"STDERR:\n{stderr}"
            )

        print(f"\n✅ Agent '{agent_id}' started on port {port} (PID={process.pid})")
    else:
        print(f"\n♻️  Agent '{agent_id}' already running on port {port}, reusing.")

    try:
        yield port
    finally:
        if process is not None:
            print(f"\n🛑 Stopping agent '{agent_id}' (PID={process.pid})")
            if os.name == "nt":
                process.terminate()
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


# ── Async HTTP client ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def http_client():
    """Async httpx client scoped per test."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield client


# ── Execute helper ─────────────────────────────────────────────────────────────

async def execute_agent(
    client: httpx.AsyncClient,
    port: int,
    prompt: str = "",
    action: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    thread_id: str = "test_thread",
) -> Dict[str, Any]:
    """POST /execute to an agent and return the parsed JSON response."""
    body: Dict[str, Any] = {
        "prompt": prompt,
        "thread_id": thread_id,
    }
    if action:
        body["action"] = action
    if payload:
        body["payload"] = payload

    resp = await client.post(f"http://localhost:{port}/execute", json=body)
    resp.raise_for_status()
    return resp.json()


# ── Test data paths ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_csv() -> Path:
    return TEST_DATA_DIR / "sales_data.csv"


@pytest.fixture(scope="session")
def sample_xlsx() -> Path:
    return TEST_DATA_DIR / "employees.xlsx"


@pytest.fixture(scope="session")
def sample_docx() -> Path:
    return TEST_DATA_DIR / "sample_report.docx"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    return TEST_DATA_DIR / "sample_invoice.pdf"

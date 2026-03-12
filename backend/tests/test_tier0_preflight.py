"""
DENISCO BETA — TIER 0: PRE-FLIGHT SYSTEM HEALTH CHECKS
=======================================================
Hard gate: ALL 8 checks must pass before any other testing begins.

Checks:
  T0-01  Orchestrator API alive                GET /api/health → 200
  T0-02  Spreadsheet agent healthy             GET :9000/health → 200
  T0-03  Document agent healthy                GET :8050/health → 200
  T0-04  Browser agent healthy                 GET :8090/health → 200
  T0-05  File upload endpoint works            POST /api/upload with .xlsx
  T0-06  WebSocket connects & responds         /ws/chat — send prompt, get event stream
  T0-07  Multi-turn memory in same thread      Two messages same thread_id; second references first
  T0-08  Agent auto-spawn                      Orchestrator routes to spreadsheet agent when file uploaded

Run:
    cd c:/Users/akush/Orchestrator-preview
    PYTHONUTF8=1 venv/Scripts/python backend/tests/test_tier0_preflight.py

Prerequisites:
    - Orchestrator running: cd backend && ../venv/Scripts/python -m uvicorn main:app --reload
    - At minimum spreadsheet agent can be pre-started or auto-spawned by orchestrator
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
import time
import uuid
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx
import websockets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORCHESTRATOR_BASE = "http://localhost:8000"
SPREADSHEET_AGENT_BASE = "http://localhost:9000"
DOCUMENT_AGENT_BASE = "http://localhost:8050"
BROWSER_AGENT_BASE = "http://localhost:8090"
WS_URL = "ws://localhost:8000/ws/chat"

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
MRP_XLSX = TEST_DATA_DIR / "MRP_Requisition_For_Production_Plan.xlsx"

HTTP_TIMEOUT = 10.0   # seconds for simple health checks
WS_TIMEOUT   = 120.0  # seconds for WebSocket conversation tests
SPAWN_TIMEOUT = 120.0  # seconds to wait for agent auto-spawn

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: list[Dict[str, Any]] = []

def record(check_id: str, name: str, passed: bool, detail: str = "", elapsed: float = 0.0):
    status = "PASS" if passed else "FAIL"
    icon   = "PASS" if passed else "FAIL"
    results.append({"id": check_id, "name": name, "status": status, "detail": detail, "elapsed": elapsed})
    print(f"  {icon} | {check_id} | {name}")
    if detail:
        print(f"         {detail}")
    if elapsed:
        print(f"         Time: {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# T0-01: Orchestrator API health
# ---------------------------------------------------------------------------

async def check_t0_01():
    name = "Orchestrator API alive"
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(f"{ORCHESTRATOR_BASE}/api/health")
        elapsed = time.time() - start
        if r.status_code == 200:
            body = r.json()
            record("T0-01", name, True, f"status={body.get('status','?')}  ({elapsed:.2f}s)", elapsed)
        else:
            record("T0-01", name, False, f"HTTP {r.status_code} — {r.text[:120]}", elapsed)
    except Exception as e:
        record("T0-01", name, False, f"Connection refused or error: {e}")


# ---------------------------------------------------------------------------
# T0-02/03/04: Agent health checks
# ---------------------------------------------------------------------------

async def check_agent_health(check_id: str, name: str, base_url: str):
    """
    Agents use LAZY initialization — server responds to /health immediately but
    'initialized' is False until the first /execute request arrives.
    PASS conditions:
      - HTTP 200 with any status (server is reachable — it will init on first use)
    FAIL condition:
      - Connection refused (agent process not running at all)
    """
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get(f"{base_url}/health")
        elapsed = time.time() - start
        if r.status_code == 200:
            body = r.json()
            initialized = body.get("initialized", False)
            status_str = body.get("status", "?")
            uptime = body.get("uptime_seconds", 0)
            agent_id = body.get("agent_id", "?")
            if initialized:
                detail = f"agent_id={agent_id}  status=ready  uptime={uptime:.0f}s"
            else:
                detail = f"agent_id={agent_id}  status={status_str} (will init on first request — normal)"
            record(check_id, name, True, detail, elapsed)
        else:
            record(check_id, name, False, f"HTTP {r.status_code}")
    except httpx.ConnectError:
        record(check_id, name, False, f"Not reachable at {base_url} — agent process is not running")
    except Exception as e:
        record(check_id, name, False, str(e))


# ---------------------------------------------------------------------------
# T0-05: File upload endpoint
# ---------------------------------------------------------------------------

async def check_t0_05():
    name = "File upload endpoint works"
    start = time.time()
    if not MRP_XLSX.exists():
        record("T0-05", name, False, f"Test file not found: {MRP_XLSX}")
        return
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(MRP_XLSX, "rb") as f:
                r = await client.post(
                    f"{ORCHESTRATOR_BASE}/api/upload",
                    files={"files": (MRP_XLSX.name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                )
        elapsed = time.time() - start
        if r.status_code == 200:
            body = r.json()
            if isinstance(body, list) and len(body) > 0:
                file_name = body[0].get("file_name", "?")
                file_path = body[0].get("file_path", "?")
                record("T0-05", name, True, f"Uploaded → file_name={file_name}  path={file_path}", elapsed)
            else:
                record("T0-05", name, False, f"Unexpected response shape: {str(body)[:200]}", elapsed)
        else:
            record("T0-05", name, False, f"HTTP {r.status_code} — {r.text[:200]}", elapsed)
    except Exception as e:
        record("T0-05", name, False, str(e))


# ---------------------------------------------------------------------------
# Shared WebSocket helper
# ---------------------------------------------------------------------------

async def ws_run_conversation(
    messages: list[str],
    thread_id: Optional[str] = None,
    timeout: float = WS_TIMEOUT,
    files: list[Dict] = None,
) -> Tuple[bool, list[str], float]:
    """
    Send one or more messages over a single WebSocket connection
    on the SAME thread_id. Returns (success, collected_responses, elapsed).
    """
    thread_id = thread_id or str(uuid.uuid4())
    owner = {"user_id": "test_user_tier0"}
    responses: list[str] = []
    start = time.time()

    try:
        async with websockets.connect(WS_URL, ping_interval=30, open_timeout=10) as ws:
            for i, prompt in enumerate(messages):
                payload = {
                    "thread_id": thread_id,
                    "prompt": prompt,
                    "owner": owner,
                    "files": files or [],
                }
                await ws.send(json.dumps(payload))

                # Collect events until we see __end__ or timeout
                turn_response = ""
                deadline = time.time() + timeout
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    node = event.get("node", "")

                    # Capture final response
                    if node == "__end__":
                        final = event.get("final_response") or ""
                        if final:
                            turn_response = final
                        break

                    # Also capture from response nodes
                    if node in ("final_answer", "answer", "response"):
                        turn_response = event.get("content", event.get("message", ""))

                    # Capture from streaming text
                    if "final_response" in event and event["final_response"]:
                        turn_response = event["final_response"]

                responses.append(turn_response)

        elapsed = time.time() - start
        return True, responses, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, [str(e)], elapsed


# ---------------------------------------------------------------------------
# T0-06: WebSocket connects & responds
# ---------------------------------------------------------------------------

async def check_t0_06():
    name = "WebSocket connects & responds"
    success, responses, elapsed = await ws_run_conversation(
        ["Hello! Can you tell me briefly what you can help me with?"],
        timeout=60.0,
    )
    if not success:
        record("T0-06", name, False, f"WebSocket error: {responses[0]}", elapsed)
        return
    resp = responses[0] if responses else ""
    if resp and len(resp) > 10:
        record("T0-06", name, True, f"Got response ({len(resp)} chars): \"{resp[:100]}...\"", elapsed)
    elif resp:
        record("T0-06", name, True, f"Got short response: \"{resp}\"", elapsed)
    else:
        record("T0-06", name, False, "Connected but received empty final_response. Check __end__ event handling.", elapsed)


# ---------------------------------------------------------------------------
# T0-07: Multi-turn memory
# ---------------------------------------------------------------------------

async def check_t0_07():
    name = "Multi-turn memory in same thread"
    thread_id = f"t0_07_{uuid.uuid4().hex[:8]}"

    # Turn 1: introduce a number the model should remember
    success, responses, elapsed1 = await ws_run_conversation(
        ["My reference number for this session is DENISCO-4472. Please confirm you have noted it."],
        thread_id=thread_id,
        timeout=60.0,
    )
    if not success:
        record("T0-07", name, False, f"Turn 1 WS error: {responses[0]}")
        return

    resp1 = responses[0] if responses else ""
    if not resp1:
        record("T0-07", name, False, "Turn 1 returned empty response — cannot test memory")
        return

    # Turn 2: ask for the number back
    success2, responses2, elapsed2 = await ws_run_conversation(
        ["What was the reference number I gave you at the start of this session?"],
        thread_id=thread_id,  # SAME thread
        timeout=60.0,
    )
    if not success2:
        record("T0-07", name, False, f"Turn 2 WS error: {responses2[0]}")
        return

    resp2 = responses2[0] if responses2 else ""
    total = elapsed1 + elapsed2

    if "4472" in resp2 or "DENISCO" in resp2:
        record("T0-07", name, True,
               f"Memory confirmed — Turn2 response contains reference: \"{resp2[:120]}\"", total)
    elif resp2:
        record("T0-07", name, False,
               f"Turn2 response did not contain '4472' or 'DENISCO'.\n         Response: \"{resp2[:200]}\"", total)
    else:
        record("T0-07", name, False, "Turn 2 returned empty response", total)


# ---------------------------------------------------------------------------
# T0-08: Agent auto-spawn (routing check via orchestrator with file)
# ---------------------------------------------------------------------------

async def check_t0_08():
    name = "Agent auto-spawn / routing with file"
    if not MRP_XLSX.exists():
        record("T0-08", name, False, f"Test file not found: {MRP_XLSX}")
        return

    # First, upload the file via HTTP
    file_entry = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(MRP_XLSX, "rb") as f:
                r = await client.post(
                    f"{ORCHESTRATOR_BASE}/api/upload",
                    files={"files": (MRP_XLSX.name, f,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                )
        if r.status_code == 200 and r.json():
            uploaded = r.json()[0]
            file_entry = {
                "file_name": uploaded["file_name"],
                "file_path": uploaded["file_path"],
                "file_type": "spreadsheet",
                "source": "user_upload",
            }
        else:
            record("T0-08", name, False, f"Upload failed: HTTP {r.status_code}")
            return
    except Exception as e:
        record("T0-08", name, False, f"Upload error: {e}")
        return

    # Now send through WebSocket with the file — the orchestrator must route to spreadsheet agent
    start = time.time()
    thread_id = f"t0_08_{uuid.uuid4().hex[:8]}"
    owner = {"user_id": "test_user_tier0"}

    try:
        async with websockets.connect(WS_URL, ping_interval=30, open_timeout=10) as ws:
            payload = {
                "thread_id": thread_id,
                "prompt": "I uploaded a spreadsheet file. What columns does it have and how many rows of data are there?",
                "owner": owner,
                "files": [file_entry],
            }
            await ws.send(json.dumps(payload))

            agent_called = False
            agent_name = None
            final_response = ""
            deadline = time.time() + SPAWN_TIMEOUT

            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                node = event.get("node", "")

                # Watch for agent dispatch events
                if node in ("task_started", "agent_dispatched", "task_progress"):
                    agent_name = event.get("agent_name") or event.get("agent") or ""
                    if agent_name:
                        agent_called = True

                # Capture result from task_completed
                if node == "task_completed":
                    result_summary = event.get("result_summary") or event.get("result") or ""
                    if result_summary and len(result_summary) > 20:
                        final_response = result_summary

                if node == "__end__":
                    final_response = event.get("final_response") or final_response
                    break

                if "final_response" in event and event["final_response"]:
                    final_response = event["final_response"]

        elapsed = time.time() - start

        if final_response and len(final_response) > 20:
            detail = f"Response received ({len(final_response)} chars)"
            if agent_called:
                detail += f"  agent_dispatched={agent_name}"
            detail += f"\n         Preview: \"{final_response[:150]}\""
            # Pass if we got a meaningful response referencing spreadsheet content
            keywords = ["column", "row", "sheet", "data", "requisition", "material", "supplier", "header"]
            has_content = any(k.lower() in final_response.lower() for k in keywords)
            record("T0-08", name, has_content, detail, elapsed)
        else:
            record("T0-08", name, False,
                   f"No meaningful response received within {SPAWN_TIMEOUT}s. agent_called={agent_called}",
                   elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record("T0-08", name, False, f"WebSocket error: {e}", elapsed)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_all():
    print()
    print("=" * 70)
    print("  DENISCO BETA — TIER 0: PRE-FLIGHT HEALTH CHECKS")
    print("  All 8 checks must PASS before any other tier can run.")
    print("=" * 70)
    print()

    # Run HTTP checks in parallel (they're fast and independent)
    print("  [HTTP checks — running in parallel]")
    await asyncio.gather(
        check_t0_01(),
        check_agent_health("T0-02", "Spreadsheet agent healthy  (:9000)", SPREADSHEET_AGENT_BASE),
        check_agent_health("T0-03", "Document agent healthy     (:8050)", DOCUMENT_AGENT_BASE),
        check_agent_health("T0-04", "Browser agent healthy      (:8090)", BROWSER_AGENT_BASE),
        check_t0_05(),
    )

    # WebSocket checks must run sequentially (they're stateful)
    print()
    print("  [WebSocket checks — running sequentially]")
    await check_t0_06()
    await check_t0_07()
    await check_t0_08()

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  TIER 0 RESULTS")
    print("=" * 70)

    passed_checks = [r for r in results if r["status"] == "PASS"]
    failed_checks = [r for r in results if r["status"] == "FAIL"]
    total = len(results)
    passed = len(passed_checks)
    failed = len(failed_checks)

    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  {icon} | {r['id']:<6} | {r['name']}")

    print()
    print(f"  Score: {passed}/{total}")
    print()

    if failed == 0:
        print("  VERDICT: GO — All pre-flight checks passed.")
        print("  You may proceed to Tier 1 and Tier 2 testing.")
    else:
        print(f"  VERDICT: NO-GO — {failed} check(s) failed.")
        print("  Fix all failures before proceeding to Tier 1.")
        print()
        print("  Failed checks:")
        for r in failed_checks:
            print(f"    {r['id']} — {r['name']}")
            print(f"           {r['detail']}")

    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    go = asyncio.run(run_all())
    sys.exit(0 if go else 1)

#!/usr/bin/env python3
"""
Spreadsheet Agent — HTTP Test Suite  (BaseAgent API)
=====================================================

The spreadsheet agent runs a BaseAgent server on port 9000.
Endpoints:
  GET  /health          — health check
  POST /execute          — JSON body: {prompt, action, payload, thread_id, task_id}
  GET  /capabilities     — list registered capabilities

Prerequisites:
  1. Start the Spreadsheet Agent server:
       cd C:\\Users\\akush\\Orchestrator-preview
       .\\venv\\Scripts\\Activate.ps1
       python -m backend.agents.spreadsheet_agent          # runs on port 9000

  2. Place your Excel file in:
       backend/tests/test_data/MRP_Requisition_For_Production_Plan.xlsx
     Or pass the path explicitly:
       python tests/test_spreadsheet_agent_standalone.py --doc "C:/full/path/to/file.xlsx"

Usage:
  cd backend
  python tests/test_spreadsheet_agent_standalone.py                          # all tests
  python tests/test_spreadsheet_agent_standalone.py --doc path/to/file.xlsx  # specify file
  python tests/test_spreadsheet_agent_standalone.py --agent-url http://localhost:9000
  python tests/test_spreadsheet_agent_standalone.py --delay 15               # longer gap between prompts
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
TEST_DATA_DIR = BACKEND_DIR / "tests" / "test_data"

# ── Colours ────────────────────────────────────────────────────────────────
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; CY = "\033[96m"
    B = "\033[1m"; E = "\033[0m"

# ── Result tracking ───────────────────────────────────────────────────────
_results: List[Dict[str, Any]] = []


def header(t: str):
    print(f"\n{C.B}{'=' * 72}\n  {t}\n{'=' * 72}{C.E}")

def log(group: str, name: str, ok: bool, msg: str = "", dur: float = 0):
    tag = f"{C.G}PASS{C.E}" if ok else f"{C.R}FAIL{C.E}"
    d = f" ({dur:.2f}s)" if dur else ""
    print(f"  {tag} | {name}{d}")
    if msg:
        for i, line in enumerate(msg[:500].split("\n")):
            print(f"{'         ' if i == 0 else '           '}{line}")
    _results.append({"group": group, "test": name, "pass": ok, "msg": msg, "time": dur})


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS — your MRP Requisition questions
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS = [
    "How many total rows of data are there (excluding headers)?",
    "What are all the column names?",
    "How many unique Item Codes are present?",
    "What date range does this data cover?",
    'How many requisitions are in "Pending" status?',
    "List all unique users who created or modified these requisitions",
    # ── 3 additional prompts to reach 9 ──────────────────────────────────
    "What is the most common status value in this dataset?",
    "Which user has the highest number of requisitions?",
    "Show me the top 5 rows sorted by the most recently modified date",
]

DEFAULT_DOC = "MRP_Requisition_For_Production_Plan.xlsx"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: extract answer text from AgentResponse
# ═══════════════════════════════════════════════════════════════════════════

def extract_answer(data: Dict) -> str:
    """Pull a human-readable answer from an AgentResponse dict."""
    if data.get("summary"):
        return str(data["summary"])
    result = data.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("message", "summary", "answer", "data"):
            if result.get(key):
                val = result[key]
                if isinstance(val, dict) and val.get("message"):
                    return str(val["message"])
                return str(val)
    if data.get("data") and isinstance(data["data"], dict):
        if data["data"].get("message"):
            return str(data["data"]["message"])
    if data.get("error_message"):
        return f"ERROR: {data['error_message']}"
    return json.dumps(data)[:400]


# ═══════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

async def test_agent_health(agent_url: str) -> bool:
    """Check the spreadsheet agent is reachable."""
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{agent_url}/health")
            status = r.json().get("status", "")
            # "not_initialized" is fine — agent is running but no session yet
            ok = r.status_code == 200 and status in ("healthy", "not_initialized", "ready")
            log("agent", "Health check", ok, r.text[:200], time.time() - t)
            return ok
    except Exception as e:
        log("agent", "Health check", False, f"Cannot reach agent at {agent_url} — {e}", time.time() - t)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 2. LIST CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════

async def test_capabilities(agent_url: str) -> Optional[List[str]]:
    """List registered capabilities."""
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{agent_url}/capabilities")
            data = r.json()
            caps = data.get("capabilities", [])
            cap_names = [cap.get("name", cap) if isinstance(cap, dict) else str(cap) for cap in caps]
            ok = r.status_code == 200 and len(cap_names) > 0
            log("agent", f"Capabilities ({len(cap_names)} found)", ok,
                ", ".join(cap_names[:15]), time.time() - t)
            return cap_names if ok else None
    except Exception as e:
        log("agent", "Capabilities", False, str(e), time.time() - t)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD FILE  (via action="load_file" with file_path)
# ═══════════════════════════════════════════════════════════════════════════

async def test_load_file(agent_url: str, doc_path: str) -> Optional[str]:
    """
    Load the Excel file using the load_file capability.
    Sends the absolute file path — the agent reads it from disk.
    Returns file_id on success.
    """
    import httpx
    t = time.time()
    fname = Path(doc_path).name
    try:
        body = {
            "prompt": f"Load the file {fname}",
            "action": "load_file",
            "payload": {
                "file_path": doc_path.replace("\\", "/"),
                "thread_id": "test-session",
            },
            "thread_id": "test-session",
        }
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{agent_url}/execute", json=body)
            data = resp.json()
            ok = resp.status_code == 200 and data.get("status") == "success"

            # Extract file_id from result
            file_id = None
            result = data.get("result") or {}
            if isinstance(result, dict):
                file_id = result.get("file_id")
                inner = result.get("data") or {}
                if not file_id and isinstance(inner, dict):
                    file_id = inner.get("file_id")

            msg = extract_answer(data)
            if file_id:
                msg = f"file_id={file_id}  |  {msg}"
            log("agent", f"Load '{fname}' (action=load_file)", ok, msg, time.time() - t)
            return file_id if ok else None
    except Exception as e:
        log("agent", f"Load '{fname}'", False, str(e), time.time() - t)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 3b. UPLOAD FILE  (via action="upload_file" with base64 content)
# ═══════════════════════════════════════════════════════════════════════════

async def test_upload_file(agent_url: str, doc_path: str) -> Optional[str]:
    """
    Upload the Excel file using the upload_file capability.
    Sends base64-encoded content — fallback if load_file fails.
    Returns file_id on success.
    """
    import httpx
    t = time.time()
    fname = Path(doc_path).name
    try:
        with open(doc_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")

        body = {
            "prompt": f"Upload the file {fname}",
            "action": "upload_file",
            "payload": {
                "content": content_b64,
                "filename": fname,
                "thread_id": "test-session",
            },
            "thread_id": "test-session",
        }
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(f"{agent_url}/execute", json=body)
            data = resp.json()
            ok = resp.status_code == 200 and data.get("status") == "success"

            file_id = None
            result = data.get("result") or {}
            if isinstance(result, dict):
                file_id = result.get("file_id")
                inner = result.get("data") or {}
                if not file_id and isinstance(inner, dict):
                    file_id = inner.get("file_id")

            msg = extract_answer(data)
            if file_id:
                msg = f"file_id={file_id}  |  {msg}"
            log("agent", f"Upload '{fname}' (action=upload_file)", ok, msg, time.time() - t)
            return file_id if ok else None
    except Exception as e:
        log("agent", f"Upload '{fname}'", False, str(e), time.time() - t)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 4. QUERY via /execute  (prompt-driven, LLM plans the capability)
# ═══════════════════════════════════════════════════════════════════════════

async def test_query(agent_url: str, prompt: str, idx: int,
                     thread_id: str = "test-session") -> Optional[Dict]:
    """Send a natural-language prompt via /execute (JSON body)."""
    import httpx
    t = time.time()
    label = f"Q{idx+1}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
    try:
        body = {
            "prompt": prompt,
            "thread_id": thread_id,
            "task_id": f"test-q{idx+1}",
        }
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post(f"{agent_url}/execute", json=body)
            data = resp.json()
            ok = resp.status_code == 200 and data.get("status") == "success"
            answer = extract_answer(data)
            log("query", label, ok, answer[:500], time.time() - t)
            return data if ok else None
    except Exception as e:
        log("query", label, False, str(e), time.time() - t)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def print_summary():
    header("TEST SUMMARY")
    groups: Dict[str, List] = {}
    for r in _results:
        groups.setdefault(r["group"], []).append(r)
    tp = tf = 0
    for g, tests in groups.items():
        p = sum(1 for t in tests if t["pass"])
        f = len(tests) - p
        tp += p; tf += f
        col = C.G if f == 0 else C.R
        print(f"  {col}{g:12s}{C.E}  {p}/{len(tests)} passed")
    print(f"\n  {'─' * 40}")
    col = C.G if tf == 0 else C.R
    print(f"  {col}{C.B}TOTAL: {tp}/{tp + tf} passed{C.E}")
    if tf:
        print(f"\n  {C.R}Failed:{C.E}")
        for r in _results:
            if not r["pass"]:
                print(f"    - [{r['group']}] {r['test']}")


# ═══════════════════════════════════════════════════════════════════════════
# Locate document
# ═══════════════════════════════════════════════════════════════════════════

def find_document(explicit: Optional[str]) -> Optional[str]:
    """Return the absolute path to the test spreadsheet."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        if p.exists():
            return str(p.resolve())
        print(f"{C.R}  File not found: {p}{C.E}")
        return None

    # Default: look for the MRP file in test_data/
    default_path = TEST_DATA_DIR / DEFAULT_DOC
    if default_path.exists():
        return str(default_path.resolve())

    # Fallback: auto-detect any Excel/CSV in test_data/
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        for f in TEST_DATA_DIR.glob(ext):
            return str(f.resolve())
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Spreadsheet Agent HTTP test suite (BaseAgent API)")
    parser.add_argument("--doc", type=str, default=None,
                        help="Path to the Excel/CSV file (auto-detects from tests/test_data/ if omitted)")
    parser.add_argument("--agent-url", type=str, default="http://localhost:9000",
                        help="Spreadsheet agent base URL (default: http://localhost:9000)")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds to wait between prompts to avoid rate limits (default: 10)")
    args = parser.parse_args()

    header("SPREADSHEET AGENT — HTTP TEST SUITE (BaseAgent API)")

    # ── Locate file ──────────────────────────────────────────────────────
    doc_path = find_document(args.doc)
    if not doc_path:
        print(f"\n{C.R}  No spreadsheet found!{C.E}")
        print(f"  Place your Excel file in:  {TEST_DATA_DIR}")
        print(f"  Or pass explicitly: --doc path/to/file.xlsx\n")
        sys.exit(1)

    print(f"  File     : {doc_path}")
    print(f"  Agent URL: {args.agent_url}")
    print(f"  Delay    : {args.delay}s between prompts")

    overall_start = time.time()

    # ── 1. Health check ──────────────────────────────────────────────────
    header("HEALTH CHECK")
    alive = await test_agent_health(args.agent_url)
    if not alive:
        print(f"\n{C.R}  Agent is not running! Start it first:{C.E}")
        print(f"    cd C:\\Users\\akush\\Orchestrator-preview")
        print(f"    .\\venv\\Scripts\\Activate.ps1")
        print(f"    python -m backend.agents.spreadsheet_agent\n")
        sys.exit(1)

    # ── 2. List capabilities ─────────────────────────────────────────────
    header("CAPABILITIES")
    await test_capabilities(args.agent_url)

    # ── 3. Load the spreadsheet ──────────────────────────────────────────
    header("LOAD SPREADSHEET")
    file_id = await test_load_file(args.agent_url, doc_path)

    # Fallback: try upload_file (base64) if load_file didn't work
    if not file_id:
        print(f"\n  {C.Y}load_file failed — trying upload_file (base64)...{C.E}")
        file_id = await test_upload_file(args.agent_url, doc_path)

    if not file_id:
        print(f"\n  {C.Y}Could not get file_id — queries may still work if file loaded in session.{C.E}")

    # ── 4. Query — all 6 prompts ─────────────────────────────────────────
    header("QUERY — 9 PROMPTS VIA /execute")
    for i, prompt in enumerate(PROMPTS):
        if i > 0 and args.delay > 0:
            print(f"  {C.Y}[wait] Waiting {args.delay}s to avoid rate limits...{C.E}")
            await asyncio.sleep(args.delay)
        await test_query(args.agent_url, prompt, i)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print_summary()
    print(f"\n  Total time: {elapsed:.1f}s\n")
    sys.exit(1 if any(not r["pass"] for r in _results) else 0)


if __name__ == "__main__":
    asyncio.run(main())

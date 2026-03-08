#!/usr/bin/env python3
"""
Coding Agent — Standalone HTTP Test Suite
==========================================

Tests the Coding Agent directly via HTTP (no orchestrator).

IMPORTANT — Dependency:
  The Coding Agent is backed by OpenCode (opencode-ai npm package).
  If OpenCode is not installed, the agent starts in FALLBACK (degraded) mode
  and all prompts will fail. This test detects that and reports it clearly.

  To install OpenCode:
    npm install -g opencode-ai@latest

Prerequisites:
  Start the Coding Agent server:
    cd C:\\Users\\akush\\Orchestrator-preview
    .\\venv\\Scripts\\Activate.ps1
    cd backend
    python -m uvicorn agents.coding_agent.__init__:app --host 0.0.0.0 --port 8080

Usage:
  cd backend
  python tests/test_coding_agent_standalone.py
  python tests/test_coding_agent_standalone.py --agent-url http://localhost:8080
  python tests/test_coding_agent_standalone.py --delay 10
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent

# ── Colours ────────────────────────────────────────────────────────────────
class C:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; CY = "\033[96m"
    B = "\033[1m"; E = "\033[0m"

# ── Result tracking ────────────────────────────────────────────────────────
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
# PROMPTS — 9 coding tasks covering review, generation, explanation, testing
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS = [
    # ── Code generation ───────────────────────────────────────────────────
    "Write a Python function called 'is_palindrome' that checks if a string "
    "is a palindrome, ignoring spaces and punctuation. Include a docstring.",

    # ── Bug review ────────────────────────────────────────────────────────
    "Review this code for bugs and explain every issue you find:\n\n"
    "def calculate_average(numbers):\n"
    "    total = 0\n"
    "    for n in numbers:\n"
    "        total += n\n"
    "    return total / len(numbers)\n\n"
    "result = calculate_average([])\nprint(result)",

    # ── API endpoint ──────────────────────────────────────────────────────
    "Write a FastAPI endpoint: POST /greet that accepts a JSON body "
    "{\"name\": string} and returns {\"message\": \"Hello {name}!\"}. "
    "Include the import statements.",

    # ── Code explanation ──────────────────────────────────────────────────
    "Explain what this Python code does, line by line:\n\n"
    "result = [x**2 for x in range(10) if x % 2 == 0]\nprint(result)",

    # ── Unit tests ────────────────────────────────────────────────────────
    "Write pytest unit tests for this function:\n\n"
    "def add(a: int, b: int) -> int:\n    return a + b\n\n"
    "Include tests for: normal addition, negative numbers, and zero.",

    # ── Refactoring ───────────────────────────────────────────────────────
    "Refactor this code to be more Pythonic and readable:\n\n"
    "result = []\nfor i in range(len(mylist)):\n    if mylist[i] > 0:\n"
    "        result.append(mylist[i] * 2)",

    # ── Type hints ────────────────────────────────────────────────────────
    "Add type hints and a docstring to this function:\n\n"
    "def process_user(name, age, email, is_active=True):\n"
    "    return {\"name\": name, \"age\": age, \"email\": email, \"active\": is_active}",

    # ── SQL ───────────────────────────────────────────────────────────────
    "Write a SQL query to find the top 5 customers by total order value. "
    "The orders table has columns: customer_id, customer_name, order_date, order_value. "
    "Include ties (use RANK, not LIMIT).",

    # ── Dependencies / tooling ────────────────────────────────────────────
    "Generate a requirements.txt for a Python project that uses: "
    "FastAPI, SQLAlchemy, Redis (via redis-py), httpx, pydantic, and pytest. "
    "Pin each to a recent stable version.",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def extract_answer(data: Dict) -> str:
    """Pull a human-readable answer from an AgentResponse dict."""
    if data.get("summary"):
        return str(data["summary"])
    result = data.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("message", "summary", "answer", "content", "output", "data"):
            if result.get(key):
                val = result[key]
                if isinstance(val, dict) and val.get("message"):
                    return str(val["message"])
                return str(val)[:500]
    if data.get("error_message"):
        return f"ERROR: {data['error_message']}"
    return json.dumps(data)[:400]


# ═══════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK — also detects fallback (degraded) mode
# ═══════════════════════════════════════════════════════════════════════════

async def test_health(agent_url: str) -> tuple[bool, bool]:
    """
    Returns (reachable, fully_initialized).
    fully_initialized=False means OpenCode is not installed (degraded mode).
    """
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{agent_url}/health")
            data = r.json()
            status = data.get("status", "")
            reachable = r.status_code == 200

            # "not_initialized" is healthy — agent uses lazy init (spins up on first /execute)
            # Only "degraded" means OpenCode is truly missing (fallback app running)
            if status == "degraded":
                error = data.get("error", "OpenCode not available")
                log("agent", "Health check", False,
                    f"DEGRADED MODE — {error}\n"
                    f"Fix: npm install -g opencode-ai@latest",
                    time.time() - t)
                return reachable, False

            ok = reachable and status in ("healthy", "ready", "not_initialized")
            log("agent", "Health check", ok, r.text[:200], time.time() - t)
            return ok, ok
    except Exception as e:
        log("agent", "Health check", False,
            f"Cannot reach {agent_url} — {e}", time.time() - t)
        return False, False


# ═══════════════════════════════════════════════════════════════════════════
# 2. CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════

async def test_capabilities(agent_url: str):
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{agent_url}/capabilities")
            data = r.json()
            caps = data.get("capabilities", [])
            cap_names = [
                cap.get("name", cap) if isinstance(cap, dict) else str(cap)
                for cap in caps
            ]
            ok = r.status_code == 200
            msg = ", ".join(cap_names[:15]) if cap_names else "(no capabilities listed)"
            log("agent", f"Capabilities ({len(cap_names)} found)", ok, msg, time.time() - t)
    except Exception as e:
        log("agent", "Capabilities", False, str(e), time.time() - t)


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUERY — single prompt through /execute
# ═══════════════════════════════════════════════════════════════════════════

async def test_query(agent_url: str, prompt: str, idx: int,
                     thread_id: str = "test-session-coding") -> bool:
    import httpx
    t = time.time()
    label = f"Q{idx + 1}: {prompt[:65]}{'...' if len(prompt) > 65 else ''}"
    try:
        body = {
            "prompt": prompt,
            "thread_id": thread_id,
            "task_id": f"test-q{idx + 1}",
        }
        async with httpx.AsyncClient(timeout=180) as c:
            resp = await c.post(f"{agent_url}/execute", json=body)
            data = resp.json()
            ok = resp.status_code == 200 and data.get("status") == "success"
            answer = extract_answer(data)
            log("query", label, ok, answer[:500], time.time() - t)
            return ok
    except Exception as e:
        log("query", label, False, str(e), time.time() - t)
        return False


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
        tp += p
        tf += f
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
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Coding Agent HTTP test suite")
    parser.add_argument("--agent-url", type=str, default="http://localhost:8080",
                        help="Coding agent base URL (default: http://localhost:8080)")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds to wait between prompts (default: 10)")
    args = parser.parse_args()

    header("CODING AGENT — HTTP TEST SUITE")
    print(f"  Agent URL: {args.agent_url}")
    print(f"  Delay    : {args.delay}s between prompts")
    print(f"  Prompts  : {len(PROMPTS)}")
    print(f"\n  {C.Y}NOTE: Requires OpenCode — npm install -g opencode-ai@latest{C.E}")

    overall_start = time.time()

    # ── 1. Health check ───────────────────────────────────────────────────
    header("HEALTH CHECK")
    reachable, fully_up = await test_health(args.agent_url)

    if not reachable:
        print(f"\n{C.R}  Agent is not running! Start it first:{C.E}")
        print(f"    cd C:\\Users\\akush\\Orchestrator-preview")
        print(f"    .\\venv\\Scripts\\Activate.ps1")
        print(f"    cd backend")
        print(f"    python -m uvicorn agents.coding_agent.__init__:app --host 0.0.0.0 --port 8080\n")
        sys.exit(1)

    if not fully_up:
        print(f"\n{C.Y}  Agent is in DEGRADED MODE — OpenCode not installed.{C.E}")
        print(f"  All prompt tests will be skipped.")
        print(f"\n  {C.B}To fix:{C.E}")
        print(f"    npm install -g opencode-ai@latest")
        print(f"  Then restart the agent and re-run this test.\n")
        print_summary()
        sys.exit(1)

    # ── 2. Capabilities ───────────────────────────────────────────────────
    header("CAPABILITIES")
    await test_capabilities(args.agent_url)

    # ── 3. Queries — all 9 prompts ─────────────────────────────────────
    header("QUERY — 9 PROMPTS VIA /execute")
    for i, prompt in enumerate(PROMPTS):
        if i > 0 and args.delay > 0:
            print(f"  {C.Y}⏳ Waiting {args.delay}s...{C.E}")
            await asyncio.sleep(args.delay)
        await test_query(args.agent_url, prompt, i)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print_summary()
    print(f"\n  Total time: {elapsed:.1f}s\n")
    sys.exit(1 if any(not r["pass"] for r in _results) else 0)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Universal Agent — Standalone HTTP Test Suite
============================================

Tests the Universal Agent directly via HTTP (no orchestrator).
The agent handles general-purpose tasks: reasoning, writing,
calculations, code generation, and research.

Prerequisites:
  Start the Universal Agent server:
    cd C:\\Users\\akush\\Orchestrator-preview
    .\\venv\\Scripts\\Activate.ps1
    cd backend
    python -m uvicorn agents.universal_agent.__init__:app --host 0.0.0.0 --port 8070

Usage:
  cd backend
  python tests/test_universal_agent_standalone.py
  python tests/test_universal_agent_standalone.py --agent-url http://localhost:8070
  python tests/test_universal_agent_standalone.py --delay 5
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
# PROMPTS — 9 varied tasks covering the agent's general-purpose range
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS = [
    # ── Maths / calculation ───────────────────────────────────────────────
    "What is 15% of 3,847? Show your calculation steps clearly.",

    # ── Code generation ───────────────────────────────────────────────────
    "Write a Python function called 'is_prime' that returns True if a number "
    "is prime and False otherwise. Include a brief explanation of the logic.",

    # ── Technical explanation ─────────────────────────────────────────────
    "Explain the difference between REST and GraphQL APIs. "
    "Use bullet points and give one concrete example for each.",

    # ── List / factual ────────────────────────────────────────────────────
    "List the capital cities of all G7 countries.",

    # ── Translation ───────────────────────────────────────────────────────
    "Translate this sentence into French, Spanish, and German: "
    "'The meeting is confirmed for Monday at 3pm.'",

    # ── Professional writing ──────────────────────────────────────────────
    "Draft a short, polite professional email declining a job offer. "
    "Keep it under 100 words.",

    # ── Algorithm complexity ──────────────────────────────────────────────
    "What is the time complexity of quicksort in best, average, and worst cases? "
    "Explain why the worst case happens.",

    # ── Concept explanation ───────────────────────────────────────────────
    "Explain what a webhook is. Give one real-world example of when you "
    "would use a webhook instead of polling.",

    # ── Business / analytical ─────────────────────────────────────────────
    "List 5 key metrics a SaaS company should track to measure product health. "
    "For each metric explain what it tells you and what a healthy value looks like.",
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
        for key in ("message", "summary", "answer", "content", "data", "output"):
            if result.get(key):
                val = result[key]
                if isinstance(val, dict) and val.get("message"):
                    return str(val["message"])
                return str(val)[:500]
    if data.get("error_message"):
        return f"ERROR: {data['error_message']}"
    return json.dumps(data)[:400]


# ═══════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

async def test_health(agent_url: str) -> bool:
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{agent_url}/health")
            data = r.json()
            status = data.get("status", "")
            ok = r.status_code == 200 and status in ("healthy", "ready", "not_initialized")
            log("agent", "Health check", ok, r.text[:200], time.time() - t)
            return ok
    except Exception as e:
        log("agent", "Health check", False, f"Cannot reach {agent_url} — {e}", time.time() - t)
        return False


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
            msg = ", ".join(cap_names[:15]) if cap_names else "(no capabilities listed — handled via execute)"
            log("agent", f"Capabilities ({len(cap_names)} found)", ok, msg, time.time() - t)
    except Exception as e:
        log("agent", "Capabilities", False, str(e), time.time() - t)


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUERY — single prompt through /execute
# ═══════════════════════════════════════════════════════════════════════════

async def test_query(agent_url: str, prompt: str, idx: int,
                     thread_id: str = "test-session-universal") -> bool:
    import httpx
    t = time.time()
    label = f"Q{idx + 1}: {prompt[:65]}{'...' if len(prompt) > 65 else ''}"
    try:
        body = {
            "prompt": prompt,
            "thread_id": thread_id,
            "task_id": f"test-q{idx + 1}",
        }
        async with httpx.AsyncClient(timeout=120) as c:
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
    parser = argparse.ArgumentParser(description="Universal Agent HTTP test suite")
    parser.add_argument("--agent-url", type=str, default="http://localhost:8070",
                        help="Universal agent base URL (default: http://localhost:8070)")
    parser.add_argument("--delay", type=int, default=8,
                        help="Seconds to wait between prompts to avoid rate limits (default: 8)")
    args = parser.parse_args()

    header("UNIVERSAL AGENT — HTTP TEST SUITE")
    print(f"  Agent URL: {args.agent_url}")
    print(f"  Delay    : {args.delay}s between prompts")
    print(f"  Prompts  : {len(PROMPTS)}")

    overall_start = time.time()

    # ── 1. Health check ───────────────────────────────────────────────────
    header("HEALTH CHECK")
    alive = await test_health(args.agent_url)
    if not alive:
        print(f"\n{C.R}  Agent is not running! Start it first:{C.E}")
        print(f"    cd C:\\Users\\akush\\Orchestrator-preview")
        print(f"    .\\venv\\Scripts\\Activate.ps1")
        print(f"    cd backend")
        print(f"    python -m uvicorn agents.universal_agent.__init__:app --host 0.0.0.0 --port 8070\n")
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

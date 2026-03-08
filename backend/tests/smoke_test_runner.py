#!/usr/bin/env python3
"""
Orbimesh Smoke Test Runner
===========================
One command that tests the entire system and generates a feedback report.

What it does:
  Tier 1 — Agent Unit Tests:
    - Auto-starts each agent subprocess
    - Runs its standalone test script (direct HTTP, no orchestrator)
    - Captures logs to tests/logs/
    - Stops the agent afterwards

  Tier 2 — Orchestrator Integration Tests:
    - Runs test_e2e_orchestration.py (Brain -> Hands -> Agents via LangGraph)
    - Captures logs to tests/logs/

  Report:
    - Generates smoke_report_DATETIME.md at the backend root
    - Lists every pass/fail with error details for intern feedback

Usage:
    cd C:\\Users\\akush\\Orchestrator-preview
    .\\venv\\Scripts\\Activate.ps1
    cd backend
    python tests/smoke_test_runner.py

    # Skip integration tests (faster, agents only):
    python tests/smoke_test_runner.py --skip-integration

    # Skip a specific tier:
    python tests/smoke_test_runner.py --skip-tier1
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── Fix Windows console encoding ──────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BACKEND_DIR / "tests"
LOGS_DIR = TESTS_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


# ── Colours ────────────────────────────────────────────────────────────────
class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    CY = "\033[96m"
    B = "\033[1m"
    E = "\033[0m"


def banner(text: str):
    print(f"\n{C.B}{'═' * 72}\n  {text}\n{'═' * 72}{C.E}")


def info(msg: str):
    print(f"  {C.Y}→{C.E} {msg}")


def good(msg: str):
    print(f"  {C.G}✓{C.E} {msg}")


def bad(msg: str):
    print(f"  {C.R}✗{C.E} {msg}")


# ── Tier 1 agent config ────────────────────────────────────────────────────
# Each entry: name, agent_id, uvicorn module, port, test script, extra args
TIER1_AGENTS = [
    {
        "name": "Spreadsheet Agent",
        "agent_id": "spreadsheet",
        "module": "agents.spreadsheet_agent",
        "port": 9000,
        "test_script": "test_spreadsheet_agent_standalone.py",
        "test_args": ["--delay", "8"],
    },
    {
        "name": "Document Agent",
        "agent_id": "document",
        "module": "agents.document_agent_lib",
        "port": 8050,
        "test_script": "test_document_agent_standalone.py",
        # --skip-chat: no main backend needed
        # --skip-execute: /execute endpoint tested separately
        "test_args": ["--skip-chat", "--skip-execute", "--delay", "8"],
    },
    {
        "name": "Universal Agent",
        "agent_id": "universal",
        "module": "agents.universal_agent",
        "port": 8070,
        "test_script": "test_universal_agent_standalone.py",
        "test_args": ["--delay", "8"],
    },
    {
        "name": "Coding Agent",
        "agent_id": "coding",
        "module": "agents.coding_agent",
        "port": 8080,
        "test_script": "test_coding_agent_standalone.py",
        # Longer delay: OpenCode tasks take more time
        "test_args": ["--delay", "12"],
    },
]


# ── Agent lifecycle ────────────────────────────────────────────────────────

def _spawn_agent(module: str, port: int) -> subprocess.Popen:
    """Start agent via uvicorn in a subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module}.__init__:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "warning",
    ]
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR)}
    return subprocess.Popen(
        cmd,
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


async def _wait_health(port: int, timeout: int = 60) -> bool:
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)
    return False


async def _is_running(port: int) -> bool:
    """Quick check: is something already listening on this port?"""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"http://localhost:{port}/health", timeout=2)
            return r.status_code == 200
    except Exception:
        return False


def _stop(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Run a test script ─────────────────────────────────────────────────────

def _run_script(script: str, args: list, log_path: Path) -> tuple[int, str]:
    """Run a test script as a subprocess, capture output to log_path."""
    cmd = [sys.executable, str(TESTS_DIR / script)] + args
    env = {**os.environ, "PYTHONPATH": str(BACKEND_DIR)}

    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(f"Command : {' '.join(cmd)}\n")
        f.write(f"Started : {datetime.now().isoformat()}\n")
        f.write(f"Backend : {BACKEND_DIR}\n")
        f.write("=" * 72 + "\n\n")
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BACKEND_DIR),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=600,
                env=env,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            f.write("\n\n[RUNNER: script timed out after 600s]\n")
            code = -1
        except Exception as e:
            f.write(f"\n\n[RUNNER: {e}]\n")
            code = -1

    content = log_path.read_text(encoding="utf-8", errors="replace")
    return code, content


# ── Parse log output ──────────────────────────────────────────────────────
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI.sub("", text)


def _parse_standalone_log(content: str) -> dict:
    """Parse PASS/FAIL counts from a standalone test script log."""
    passed, failed = 0, 0
    failed_tests = []
    total_line = ""
    times = []

    for line in content.splitlines():
        clean = _strip(line)
        if "PASS |" in clean:
            passed += 1
            m = re.search(r"\((\d+\.\d+)s\)", clean)
            if m:
                times.append(float(m.group(1)))
        elif "FAIL |" in clean:
            failed += 1
            m = re.search(r"FAIL \| (.+?)(?:\s*\(|$)", clean)
            if m:
                failed_tests.append(m.group(1).strip())
        if "TOTAL:" in clean:
            total_line = clean.strip()

    return {
        "passed": passed,
        "failed": failed,
        "failed_tests": failed_tests,
        "total_line": total_line,
        "avg_time_s": round(sum(times) / len(times), 1) if times else 0.0,
    }


def _parse_e2e_log(content: str) -> dict:
    """Parse results from test_e2e_orchestration.py output."""
    passed, failed = 0, 0
    test_lines = []
    total_line = ""

    for line in content.splitlines():
        clean = _strip(line.strip())
        # Summary table lines look like: "PASS | Test Name  | 12.3s | agents: ..."
        if re.match(r"(PASS|FAIL|ERR |T/O )\s*\|", clean):
            test_lines.append(clean)
            if clean.startswith("PASS"):
                passed += 1
            else:
                failed += 1
        if "Total:" in clean and "passed" in clean:
            total_line = clean

    return {
        "passed": passed,
        "failed": failed,
        "test_lines": test_lines,
        "total_line": total_line,
    }


# ── Tier 1: Agent unit tests ──────────────────────────────────────────────

async def run_tier1(agents_cfg: list) -> list:
    banner("TIER 1 — AGENT UNIT TESTS (Direct HTTP to agents)")
    results = []

    for cfg in agents_cfg:
        name = cfg["name"]
        port = cfg["port"]
        module = cfg["module"]
        script = cfg["test_script"]
        args = cfg["test_args"]

        print(f"\n  {'─' * 60}")
        info(f"[{name}]  port={port}")

        # Check if already running (user may have pre-started it)
        already_up = await _is_running(port)
        proc = None

        if already_up:
            info(f"Agent already running on port {port} — using existing instance")
        else:
            info("Spawning agent...")
            proc = _spawn_agent(module, port)
            ready = await _wait_health(port, timeout=60)
            if not ready:
                bad(f"Agent failed to start within 60s — skipping")
                _stop(proc)
                results.append({
                    "name": name,
                    "status": "SKIPPED",
                    "reason": "Failed to start",
                    "passed": 0, "failed": 0,
                    "failed_tests": [], "total_line": "",
                    "avg_time_s": 0.0, "wall_s": 0.0,
                    "log": None,
                })
                continue
            good(f"Agent ready")

        log_path = LOGS_DIR / f"tier1_{cfg['agent_id']}_{TIMESTAMP}.log"
        info(f"Running {script}  →  {log_path.name}")

        t0 = time.time()
        exit_code, content = _run_script(script, args, log_path)
        wall = time.time() - t0

        if proc:
            _stop(proc)
            info("Agent stopped")

        parsed = _parse_standalone_log(content)
        total = parsed["passed"] + parsed["failed"]

        if parsed["failed"] == 0 and exit_code == 0:
            status = "PASS"
            good(f"{name}: {parsed['passed']}/{total} passed  ({wall:.0f}s wall, avg {parsed['avg_time_s']}s/prompt)")
        else:
            status = "FAIL"
            bad(f"{name}: {parsed['passed']}/{total} passed, {parsed['failed']} failed  ({wall:.0f}s wall)")
            for ft in parsed["failed_tests"][:5]:
                print(f"      - {ft}")

        results.append({
            "name": name,
            "status": status,
            "passed": parsed["passed"],
            "failed": parsed["failed"],
            "failed_tests": parsed["failed_tests"],
            "total_line": parsed["total_line"],
            "avg_time_s": parsed["avg_time_s"],
            "wall_s": round(wall, 1),
            "log": log_path,
        })

    return results


# ── Tier 2: Orchestrator integration tests ────────────────────────────────

async def run_tier2() -> dict:
    banner("TIER 2 — ORCHESTRATOR INTEGRATION TESTS (Brain → Hands → Agents)")
    info("Runs LangGraph directly — agents are spawned on-demand by Hands")
    info("Expected duration: 5–15 minutes")

    log_path = LOGS_DIR / f"tier2_integration_{TIMESTAMP}.log"
    info(f"Running test_e2e_orchestration.py  →  {log_path.name}")

    t0 = time.time()
    exit_code, content = _run_script("test_e2e_orchestration.py", [], log_path)
    wall = time.time() - t0

    parsed = _parse_e2e_log(content)
    total = parsed["passed"] + parsed["failed"]

    if parsed["failed"] == 0 and exit_code == 0:
        status = "PASS"
        good(f"Integration: {parsed['passed']}/{total} passed  ({wall:.0f}s wall)")
    else:
        status = "FAIL"
        bad(f"Integration: {parsed['passed']}/{total} passed, {parsed['failed']} failed  ({wall:.0f}s wall)")

    return {
        "status": status,
        "passed": parsed["passed"],
        "failed": parsed["failed"],
        "test_lines": parsed["test_lines"],
        "total_line": parsed["total_line"],
        "wall_s": round(wall, 1),
        "log": log_path,
    }


# ── Report generator ──────────────────────────────────────────────────────

def generate_report(tier1: list, tier2: dict | None) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_path = BACKEND_DIR / f"smoke_report_{TIMESTAMP}.md"

    L = []

    def h(text): L.append(f"\n{text}")
    def line(text=""): L.append(text)

    L.append(f"# Smoke Test Report — {ts}")
    line()
    line("> Auto-generated by `tests/smoke_test_runner.py`  ")
    line(f"> Logs: `backend/tests/logs/`")
    line()
    line("---")

    # ── Executive summary table ──
    h("## Executive Summary")
    line()
    line("| Component | Tests | Passed | Failed | Status |")
    line("|-----------|-------|--------|--------|--------|")

    total_p = total_f = 0
    for r in tier1:
        total = r["passed"] + r["failed"]
        icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "SKIPPED" else "❌")
        line(f"| {r['name']} | {total} | {r['passed']} | {r['failed']} | {icon} {r['status']} |")
        total_p += r["passed"]
        total_f += r["failed"]

    if tier2:
        t2_total = tier2["passed"] + tier2["failed"]
        t2_icon = "✅" if tier2["status"] == "PASS" else "❌"
        line(f"| Orchestrator Integration | {t2_total} | {tier2['passed']} | {tier2['failed']} | {t2_icon} {tier2['status']} |")
        total_p += tier2["passed"]
        total_f += tier2["failed"]

    line()
    overall = "✅ ALL PASSING" if total_f == 0 else f"❌ {total_f} test(s) failing"
    line(f"**Overall: {total_p} passed / {total_p + total_f} total — {overall}**")
    line()
    line("---")

    # ── Tier 1 details ──
    h("## Tier 1 — Agent Unit Tests")
    line()
    line("Each agent was started, hit with real prompts via HTTP, then stopped.")
    line()

    for r in tier1:
        icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "SKIPPED" else "❌")
        h(f"### {icon} {r['name']}")
        line()

        if r["status"] == "SKIPPED":
            line(f"**SKIPPED** — {r.get('reason', '')}")
            line()
            continue

        line(f"| Metric | Value |")
        line(f"|--------|-------|")
        line(f"| Tests run | {r['passed'] + r['failed']} |")
        line(f"| Passed | {r['passed']} |")
        line(f"| Failed | {r['failed']} |")
        line(f"| Avg prompt time | {r['avg_time_s']}s |")
        line(f"| Total wall time | {r['wall_s']}s |")
        line(f"| Log file | `{r['log'].name if r['log'] else 'N/A'}` |")
        line()

        if r["failed_tests"]:
            line("**Failed tests (fix these):**")
            line()
            for ft in r["failed_tests"]:
                line(f"- ❌ `{ft}`")
            line()
        elif r["status"] == "PASS":
            line("All tests passed. ✅")
            line()

    line("---")

    # ── Tier 2 details ──
    if tier2:
        h("## Tier 2 — Orchestrator Integration Tests")
        line()
        line("Tests the full Brain → Hands → Agent cycle via LangGraph.")
        line()

        t2_icon = "✅" if tier2["status"] == "PASS" else "❌"
        line(f"**Status: {t2_icon} {tier2['status']}**  |  "
             f"Passed: {tier2['passed']}  |  Failed: {tier2['failed']}  |  "
             f"Wall time: {tier2['wall_s']}s")
        line()

        if tier2.get("test_lines"):
            line("**Per-test results:**")
            line()
            for tl in tier2["test_lines"]:
                icon = "✅" if tl.strip().startswith("PASS") else "❌"
                line(f"- {icon} `{tl.strip()}`")
            line()

        if tier2.get("log"):
            line(f"Log: `{tier2['log'].name}`")
            line()

        line("---")

    # ── Issues for interns ──
    all_issues = []
    for r in tier1:
        for ft in r.get("failed_tests", []):
            all_issues.append(f"[{r['name']}] {ft}")
    if tier2 and tier2["failed"] > 0:
        for tl in tier2.get("test_lines", []):
            if not tl.strip().startswith("PASS"):
                all_issues.append(f"[Integration] {tl.strip()}")

    h("## For Interns — Issues to Fix")
    line()
    if not all_issues:
        line("No issues found — all tests passed! 🎉")
    else:
        line(f"Found **{len(all_issues)} issue(s)** to investigate:")
        line()
        for issue in all_issues:
            line(f"1. ❌ {issue}")
        line()
        line("**How to debug:**")
        line()
        line("1. Open the log file for the failing component (paths listed above)")
        line("2. Search for `FAIL` or `ERROR` to find the specific error")
        line("3. For integration failures, look for `BRAIN` routing and `HANDS` error lines")
        line("4. For agent failures, check for API key errors (429, 401) or timeout messages")

    line()
    line("---")
    line()
    line(f"*Generated: {ts} by smoke_test_runner.py*")

    report_path.write_text("\n".join(L), encoding="utf-8")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Orbimesh Smoke Test Runner")
    parser.add_argument("--skip-tier1", action="store_true",
                        help="Skip Tier 1 agent unit tests")
    parser.add_argument("--skip-integration", action="store_true",
                        help="Skip Tier 2 integration tests")
    parser.add_argument("--agents", nargs="+",
                        choices=[a["agent_id"] for a in TIER1_AGENTS],
                        help="Run only specific agents in Tier 1")
    args = parser.parse_args()

    banner("ORBIMESH SMOKE TEST RUNNER")
    print(f"  Backend : {BACKEND_DIR}")
    print(f"  Logs    : {LOGS_DIR}")
    print(f"  Run ID  : {TIMESTAMP}")
    print()
    print(f"  Tiers to run:")
    if not args.skip_tier1:
        agents_to_run = [a for a in TIER1_AGENTS
                         if not args.agents or a["agent_id"] in args.agents]
        for a in agents_to_run:
            print(f"    Tier 1 — {a['name']} (port {a['port']})")
    else:
        print("    Tier 1 — SKIPPED")
    if not args.skip_integration:
        print("    Tier 2 — Orchestrator Integration (9 tests via LangGraph)")
    else:
        print("    Tier 2 — SKIPPED")

    # ── Tier 1 ────────────────────────────────────────────────────────────
    tier1_results = []
    if not args.skip_tier1:
        agents_to_run = [a for a in TIER1_AGENTS
                         if not args.agents or a["agent_id"] in args.agents]
        tier1_results = await run_tier1(agents_to_run)

    # ── Tier 2 ────────────────────────────────────────────────────────────
    tier2_result = None
    if not args.skip_integration:
        tier2_result = await run_tier2()

    # ── Report ────────────────────────────────────────────────────────────
    banner("GENERATING REPORT")
    report_path = generate_report(tier1_results, tier2_result)
    good(f"Report saved → {report_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    banner("FINAL SUMMARY")

    all_pass = True
    for r in tier1_results:
        icon = (C.G + "PASS" + C.E) if r["status"] == "PASS" else \
               (C.Y + "SKIP" + C.E) if r["status"] == "SKIPPED" else \
               (C.R + "FAIL" + C.E)
        total = r["passed"] + r["failed"]
        print(f"  [{icon}] {r['name']}: {r['passed']}/{total}")
        if r["status"] not in ("PASS", "SKIPPED"):
            all_pass = False

    if tier2_result:
        t2_icon = (C.G + "PASS" + C.E) if tier2_result["status"] == "PASS" else (C.R + "FAIL" + C.E)
        t2_total = tier2_result["passed"] + tier2_result["failed"]
        print(f"  [{t2_icon}] Integration: {tier2_result['passed']}/{t2_total}")
        if tier2_result["status"] != "PASS":
            all_pass = False

    total_p = sum(r["passed"] for r in tier1_results) + (tier2_result["passed"] if tier2_result else 0)
    total_f = sum(r["failed"] for r in tier1_results) + (tier2_result["failed"] if tier2_result else 0)

    print(f"\n  {'─' * 50}")
    overall_lbl = (C.G + "ALL PASSING ✓" + C.E) if all_pass else (C.R + f"{total_f} FAILING ✗" + C.E)
    print(f"  Total: {total_p}/{total_p + total_f} — {overall_lbl}")
    print(f"\n  Report : {report_path}")
    print(f"  Logs   : {LOGS_DIR}")
    print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())

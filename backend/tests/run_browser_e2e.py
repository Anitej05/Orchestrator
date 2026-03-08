"""
Browser Agent E2E Tests — focused runner.
Runs only the 4 browser tests through the full LangGraph orchestrator.

Usage:
    cd /c/Users/akush/Orchestrator-preview
    PYTHONUTF8=1 venv/Scripts/python backend/tests/run_browser_e2e.py
"""
import sys
import io

# Re-wrap stdout only when running in an interactive Windows console that uses cp1252.
# Skip when stdout is already redirected (e.g. piped, background task).
if hasattr(sys.stdout, 'buffer') and sys.stdout.isatty():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
import logging
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)

from test_e2e_orchestration import run_e2e_test


async def run_browser_tests():
    all_results = []

    print("\n" + "=" * 72)
    print("  BROWSER AGENT E2E TESTS — Full Orchestrator Flow")
    print("  Brain -> Hands -> agent_manager -> browser_agent (port 8090)")
    print("=" * 72)

    # TEST 1: Simple navigation & extraction
    r = await run_e2e_test(
        test_name="Browser - Simple Navigation & Extraction",
        prompt=(
            "Go to https://example.com and tell me: "
            "1) The main heading on the page "
            "2) A brief description of what the page says "
            "3) Any links that are visible on the page"
        ),
        expected_agent="browser",
        timeout=300,
    )
    all_results.append(r)

    # TEST 2: Wikipedia article extraction
    r = await run_e2e_test(
        test_name="Browser - Wikipedia Article Summary",
        prompt=(
            "Navigate to https://en.wikipedia.org/wiki/Python_(programming_language) "
            "and extract: "
            "1) The first paragraph of the article (the introduction) "
            "2) When Python was first released and who created it "
            "3) The main programming paradigms it supports"
        ),
        expected_agent="browser",
        timeout=300,
    )
    all_results.append(r)

    # TEST 3: Web scraping structured data
    r = await run_e2e_test(
        test_name="Browser - Scrape Quotes Website",
        prompt=(
            "Go to https://quotes.toscrape.com and extract: "
            "1) The first 5 quotes shown on the page (text and author) "
            "2) The tags associated with the first quote "
            "3) The total number of quotes visible on the first page"
        ),
        expected_agent="browser",
        timeout=300,
    )
    all_results.append(r)

    # TEST 4: Multi-step web interaction
    r = await run_e2e_test(
        test_name="Browser - Multi-Step: Navigate and Extract",
        prompt=(
            "Navigate to https://httpbin.org and: "
            "1) Tell me what services this site provides (from the page content) "
            "2) List at least 3 of the HTTP method endpoints it exposes "
            "3) What is the base URL for the API?"
        ),
        expected_agent="browser",
        timeout=300,
    )
    all_results.append(r)

    # Summary
    print("\n" + "=" * 72)
    print("  BROWSER E2E RESULTS")
    print("=" * 72)

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] in ("ERROR", "TIMEOUT"))
    total = len(all_results)

    for r in all_results:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR ", "TIMEOUT": "T/O "}[r["status"]]
        agents = ", ".join(a["agent"] for a in r.get("agents_used", [])) or "direct/none"
        print(f"  {icon} | {r['test']:<45} | {r.get('time', 0):>5.1f}s | agents: {agents}")

    print(f"\n  Total: {passed}/{total} passed, {failed} failed, {errors} errors/timeouts")
    print("=" * 72 + "\n")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_browser_tests())

"""
End-to-End Brain -> Hands Orchestration Tests

Tests the FULL LangGraph cycle with real-world tasks:
  User prompt -> Brain.think() -> Hands.execute() -> Agent spawning -> Result

Each test sends a real-world task through the compiled graph and verifies:
1. Brain correctly routes to the right agent
2. Hands spawns the agent and executes the task
3. Final state contains a meaningful response

Run:
    cd d:\\Internship\\Orbimesh\\backend
    python tests/test_e2e_orchestration.py
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
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("E2E-Test")

# Reduce noise from HTTP/inference logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# HELPERS
# ============================================================================

def build_initial_state(prompt: str, thread_id: str = None) -> Dict[str, Any]:
    """Build a fresh orchestrator state for a new conversation."""
    thread_id = thread_id or str(uuid.uuid4())
    return {
        "original_prompt": prompt,
        "messages": [HumanMessage(content=prompt)],
        "uploaded_files": [],
        "parsed_tasks": [],
        "user_expectations": {},
        "candidate_agents": {},
        "task_agent_pairs": [],
        "task_plan": [],
        "completed_tasks": [],
        "final_response": None,
        "pending_user_input": False,
        "question_for_user": None,
        "user_response": None,
        "parsing_error_feedback": None,
        "parse_retry_count": 0,
        "needs_complex_processing": None,
        "analysis_reasoning": None,
        "planning_mode": False,
        # Orchestrator fields
        "thread_id": thread_id,
        "user_id": "test_user",
        "todo_list": [],
        "memory": {},
        "iteration_count": 0,
        "failure_count": 0,
        "max_iterations": 25,
        "action_history": [],
        "insights": {},
        "execution_plan": None,
        "current_phase_id": None,
        "pending_approval": False,
        "pending_decision": None,
        # File tracking
        "created_files": [],
        "orchestrator_workspace": "",
        "shared_files": [],
        "shared_workspace": "",
        # Canvas
        "canvas_registry": None,
        "active_canvas_id": None,
        "has_canvas": False,
        "canvas_type": None,
        "canvas_content": None,
        "canvas_data": None,
        "canvas_title": None,
        "browser_view": None,
        "plan_view": None,
        "current_view": None,
        "error": None,
    }


def extract_test_results(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract meaningful test data from the final graph state."""
    action_history = final_state.get("action_history", [])
    
    agents_used = []
    for entry in action_history:
        if isinstance(entry, dict) and entry.get("action_type") == "agent":
            agents_used.append({
                "agent": entry.get("resource_id", "unknown"),
                "success": entry.get("success", False),
                "time_ms": round(entry.get("execution_time_ms", 0)),
                "summary": (entry.get("result_summary") or "")[:100],
            })

    return {
        "final_response": (final_state.get("final_response") or "")[:200],
        "has_final_response": bool(final_state.get("final_response")),
        "iterations": final_state.get("iteration_count", 0),
        "total_actions": len(action_history),
        "agents_used": agents_used,
        "has_canvas": final_state.get("has_canvas", False),
        "canvas_type": final_state.get("canvas_type"),
        "insights": final_state.get("insights", {}),
        "error": final_state.get("error"),
        "pending_approval": final_state.get("pending_approval", False),
    }


async def run_e2e_test(
    test_name: str,
    prompt: str,
    expected_agent: str = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Run a single E2E test through the full LangGraph brain->hands cycle.
    
    Args:
        test_name: Human-readable test name
        prompt: The user's task prompt
        expected_agent: If set, verify this agent was used
        timeout: Max time for the graph to complete
    """
    from orchestrator.graph import create_graph_with_checkpointer

    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)

    config = {
        "recursion_limit": 80,
        "configurable": {"thread_id": thread_id},
    }

    initial_state = build_initial_state(prompt, thread_id)

    print(f"\n  [{test_name}]")
    print(f"  Prompt: \"{prompt[:80]}...\"" if len(prompt) > 80 else f"  Prompt: \"{prompt}\"")
    print(f"  Expected agent: {expected_agent or 'any'}")

    start = time.time()
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=timeout,
        )
        elapsed = time.time() - start
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"test": test_name, "status": "TIMEOUT", "time": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR after {elapsed:.1f}s: {e}")
        return {"test": test_name, "status": "ERROR", "error": str(e), "time": round(elapsed, 1)}

    results = extract_test_results(final_state)
    results["test"] = test_name
    results["time"] = round(elapsed, 1)

    # Determine pass/fail
    passed = True
    reasons = []

    if not results["has_final_response"]:
        passed = False
        reasons.append("No final_response")

    if expected_agent and results["agents_used"]:
        agents_called = [a["agent"] for a in results["agents_used"]]
        if not any(expected_agent in a for a in agents_called):
            passed = False
            reasons.append(f"Expected {expected_agent}, got {agents_called}")

    results["status"] = "PASS" if passed else "FAIL"
    results["fail_reasons"] = reasons

    # Print results
    status_icon = "PASS" if passed else "FAIL"
    print(f"  {status_icon} | Time: {elapsed:.1f}s | Iterations: {results['iterations']}")
    
    if results["agents_used"]:
        for a in results["agents_used"]:
            a_status = "ok" if a["success"] else "err"
            print(f"    Agent: {a['agent']} [{a_status}] ({a['time_ms']}ms)")
    else:
        print(f"    No agents dispatched (Brain answered directly or used tools)")

    if results["has_final_response"]:
        resp_preview = results["final_response"][:120].replace('\n', ' ')
        print(f"    Response: \"{resp_preview}...\"")

    if results["has_canvas"]:
        print(f"    Canvas: type={results['canvas_type']}")

    if reasons:
        print(f"    Fail reasons: {', '.join(reasons)}")

    return results


# ============================================================================
# E2E TEST CASES — Real-world tasks targeting specific agents
# ============================================================================

async def run_all_e2e_tests():
    """Run all end-to-end orchestration tests."""
    
    all_results = []

    print("\n" + "=" * 72)
    print("  END-TO-END BRAIN -> HANDS ORCHESTRATION TESTS")
    print("  Each test goes through the FULL LangGraph cycle:")
    print("  User prompt -> Brain.think() -> Hands.execute() -> Agent -> Result")
    print("=" * 72)

    # ---- TEST 1: Universal Agent (simple question) ----
    r = await run_e2e_test(
        test_name="Universal Agent - Simple Question",
        prompt="What are the three laws of thermodynamics? Explain each briefly.",
        expected_agent="universal",
        timeout=60,
    )
    all_results.append(r)

    # ---- TEST 2: Coding Agent ----
    r = await run_e2e_test(
        test_name="Coding Agent - Write Code",
        prompt="Write a Python function that checks if a string is a valid palindrome, ignoring spaces and punctuation. Include test cases.",
        expected_agent="coding",
        timeout=120,
    )
    all_results.append(r)

    # ---- TEST 3: Document Agent ----
    r = await run_e2e_test(
        test_name="Document Agent - Create Document",
        prompt="Create a markdown document titled 'Meeting Notes - Q4 Planning' with sections for Attendees, Agenda, Discussion Points, and Action Items. Fill in sample content.",
        expected_agent="document",
        timeout=90,
    )
    all_results.append(r)

    # ---- TEST 4: Spreadsheet Agent ----
    r = await run_e2e_test(
        test_name="Spreadsheet Agent - Data Task",
        prompt="Create a CSV file with monthly sales data for 2024. Include columns: Month, Product, Units_Sold, Revenue, Region. Generate 12 rows of sample data.",
        expected_agent="spreadsheet",
        timeout=90,
    )
    all_results.append(r)

    # ---- TEST 5: Sequential multi-step ----
    r = await run_e2e_test(
        test_name="Multi-Step Sequential",
        prompt="First, explain what a REST API is in 2 sentences. Then write a simple Python Flask API with one GET endpoint that returns a JSON greeting.",
        expected_agent=None,  # Could be universal or coding
        timeout=120,
    )
    all_results.append(r)

    # ---- TEST 6: Brain answers directly (no agent needed) ----
    r = await run_e2e_test(
        test_name="Brain Direct Answer - No Agent",
        prompt="What is 2 + 2?",
        expected_agent=None,  # Brain should answer directly
        timeout=30,
    )
    all_results.append(r)

    # ---- SUMMARY ----
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errors = sum(1 for r in all_results if r["status"] in ("ERROR", "TIMEOUT"))
    total = len(all_results)

    for r in all_results:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR ", "TIMEOUT": "T/O "}[r["status"]]
        agents = ", ".join(a["agent"] for a in r.get("agents_used", [])) or "direct"
        print(f"  {icon} | {r['test']:<40} | {r.get('time', 0):>5.1f}s | agents: {agents}")

    print(f"\n  Total: {passed}/{total} passed, {failed} failed, {errors} errors/timeouts")
    print("=" * 72 + "\n")

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_all_e2e_tests())

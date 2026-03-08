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

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any

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

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

MRP_XLSX    = str(TEST_DATA_DIR / "MRP_Requisition_For_Production_Plan.xlsx")
MANUS_PDF   = str(TEST_DATA_DIR / "manus doc.pdf")
EMPLOYEES_XLSX = str(TEST_DATA_DIR / "employees.xlsx")
SALES_CSV   = str(TEST_DATA_DIR / "sales_data.csv")
INVOICE_PDF = str(TEST_DATA_DIR / "sample_invoice.pdf")
REPORT_DOCX = str(TEST_DATA_DIR / "sample_report.docx")


def make_file(path: str, file_type: str, name: str = None) -> Dict[str, Any]:
    """Build an uploaded_files entry from a local file path."""
    return {
        "file_name": name or Path(path).name,
        "file_path": path,
        "file_type": file_type,
        "source": "user_upload",
    }


def build_initial_state(prompt: str, thread_id: str = None,
                        uploaded_files: list = None) -> Dict[str, Any]:
    """Build a fresh orchestrator state for a new conversation."""
    thread_id = thread_id or str(uuid.uuid4())
    return {
        "original_prompt": prompt,
        "messages": [HumanMessage(content=prompt)],
        "uploaded_files": uploaded_files or [],
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
    uploaded_files: list = None,
) -> Dict[str, Any]:
    """
    Run a single E2E test through the full LangGraph brain->hands cycle.

    Args:
        test_name: Human-readable test name
        prompt: The user's task prompt
        expected_agent: If set, verify this agent was used
        timeout: Max time for the graph to complete
        uploaded_files: List of file dicts built with make_file()
    """
    from orchestrator.graph import create_graph_with_checkpointer

    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)

    config = {
        "recursion_limit": 80,
        "configurable": {"thread_id": thread_id},
    }

    initial_state = build_initial_state(prompt, thread_id,
                                        uploaded_files=uploaded_files)

    print(f"\n  [{test_name}]")
    print(f"  Prompt: \"{prompt[:80]}...\"" if len(prompt) > 80 else f"  Prompt: \"{prompt}\"")
    if uploaded_files:
        for f in uploaded_files:
            print(f"  File: {f['file_name']} ({f['file_type']})")
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
        print("    No agents dispatched (Brain answered directly or used tools)")

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

    # =========================================================================
    # SPREADSHEET AGENT TESTS — real .xlsx / .csv files uploaded
    # =========================================================================

    # ---- TEST 1: MRP Requisition — supplier & value analysis ----
    r = await run_e2e_test(
        test_name="Spreadsheet - MRP Top Suppliers by Value",
        prompt=(
            "I have uploaded our MRP Requisition file. "
            "Please analyse it and tell me: "
            "1) Which are the top 5 suppliers ranked by total requisition value? "
            "2) What is the overall total value across all requisitions? "
            "3) Are there any items where the quantity requested seems unusually high?"
        ),
        expected_agent="spreadsheet",
        timeout=180,
        uploaded_files=[make_file(MRP_XLSX, "spreadsheet")],
    )
    all_results.append(r)

    # ---- TEST 2: MRP Requisition — date & status filter ----
    r = await run_e2e_test(
        test_name="Spreadsheet - MRP Overdue & Pending Items",
        prompt=(
            "Using the uploaded MRP Requisition spreadsheet, "
            "show me all requisitions that are still PENDING or OPEN. "
            "Group them by department and give me a count and total value per department. "
            "Which department has the highest pending spend?"
        ),
        expected_agent="spreadsheet",
        timeout=180,
        uploaded_files=[make_file(MRP_XLSX, "spreadsheet")],
    )
    all_results.append(r)

    # ---- TEST 3: Employees spreadsheet — headcount & salary stats ----
    r = await run_e2e_test(
        test_name="Spreadsheet - Employee Headcount & Salary Stats",
        prompt=(
            "I have uploaded the employees spreadsheet. "
            "Please give me: "
            "1) Total headcount by department "
            "2) Average, min and max salary per department "
            "3) Which 3 employees have the highest salary overall?"
        ),
        expected_agent="spreadsheet",
        timeout=180,
        uploaded_files=[make_file(EMPLOYEES_XLSX, "spreadsheet")],
    )
    all_results.append(r)

    # ---- TEST 4: Sales CSV — revenue trend ----
    r = await run_e2e_test(
        test_name="Spreadsheet - Sales Revenue Trend",
        prompt=(
            "Analyse the uploaded sales data CSV. "
            "1) What is the total revenue by month? Show a clear month-by-month breakdown. "
            "2) Which product generated the most revenue overall? "
            "3) Which region has the lowest sales and by how much does it lag behind the top region?"
        ),
        expected_agent="spreadsheet",
        timeout=180,
        uploaded_files=[make_file(SALES_CSV, "spreadsheet")],
    )
    all_results.append(r)

    # =========================================================================
    # DOCUMENT AGENT TESTS — real PDF / DOCX files uploaded
    # =========================================================================

    # ---- TEST 5: Manus PDF — key findings summary ----
    r = await run_e2e_test(
        test_name="Document - Manus PDF Key Findings",
        prompt=(
            "I have uploaded a PDF document. "
            "Please read it and provide: "
            "1) A concise executive summary (max 5 sentences) "
            "2) The 3 most important findings or conclusions "
            "3) Any specific numbers, dates or names mentioned that seem significant"
        ),
        expected_agent="document",
        timeout=180,
        uploaded_files=[make_file(MANUS_PDF, "document")],
    )
    all_results.append(r)

    # ---- TEST 6: Sample invoice PDF — extract structured data ----
    r = await run_e2e_test(
        test_name="Document - Invoice Data Extraction",
        prompt=(
            "I have uploaded an invoice PDF. "
            "Extract and list: "
            "1) Invoice number and date "
            "2) Vendor name and address "
            "3) All line items with their quantities, unit prices and totals "
            "4) The final total amount due and payment terms"
        ),
        expected_agent="document",
        timeout=180,
        uploaded_files=[make_file(INVOICE_PDF, "document")],
    )
    all_results.append(r)

    # ---- TEST 7: Sample report DOCX — edit & improve ----
    r = await run_e2e_test(
        test_name="Document - Report Review & Improvement",
        prompt=(
            "I have uploaded a Word document report. "
            "Please: "
            "1) Summarise what the report is about in 2-3 sentences "
            "2) Identify any sections that are unclear or could be improved "
            "3) Suggest a better structure or additional sections that would strengthen it"
        ),
        expected_agent="document",
        timeout=180,
        uploaded_files=[make_file(REPORT_DOCX, "document")],
    )
    all_results.append(r)

    # =========================================================================
    # CODING AGENT TESTS — tasks that require writing / reviewing real code
    # =========================================================================

    # ---- TEST 8: Code review of a buggy snippet ----
    r = await run_e2e_test(
        test_name="Coding - Bug Review & Fix",
        prompt=(
            "Review this Python code, find every bug, and provide a corrected version:\n\n"
            "def process_orders(orders):\n"
            "    total = 0\n"
            "    discounted = []\n"
            "    for order in orders:\n"
            "        if order['value'] > 100:\n"
            "            order['value'] = order['value'] * 0.9  # 10% discount\n"
            "            discounted.append(order)\n"
            "        total += order['value']\n"
            "    avg = total / len(orders)\n"
            "    return total, avg, discounted\n\n"
            "result = process_orders([])\n"
            "print(f'Total: {result[0]}, Avg: {result[1]}')\n\n"
            "Bugs to find: empty list division, mutation of input dict, missing return handling."
        ),
        expected_agent="coding",
        timeout=150,
    )
    all_results.append(r)

    # ---- TEST 9: Generate a full working REST API module ----
    r = await run_e2e_test(
        test_name="Coding - Generate FastAPI CRUD Module",
        prompt=(
            "Write a complete, production-ready FastAPI module for a 'Product' resource. "
            "Requirements: "
            "- Pydantic model: Product(id, name, price, stock_quantity, category) "
            "- In-memory list as the data store (no database needed) "
            "- Full CRUD: GET /products, GET /products/{id}, POST /products, "
            "  PUT /products/{id}, DELETE /products/{id} "
            "- Proper HTTP status codes (201 on create, 404 on not found) "
            "- Input validation (price must be > 0, name cannot be empty) "
            "- Include all imports. The file must run as-is with uvicorn."
        ),
        expected_agent="coding",
        timeout=150,
    )
    all_results.append(r)

    # =========================================================================
    # MULTI-AGENT & CROSS-AGENT TESTS
    # =========================================================================

    # ---- TEST 10: Spreadsheet analysis → Document write-up ----
    r = await run_e2e_test(
        test_name="Multi-Agent - Analyse Spreadsheet then Write Report",
        prompt=(
            "I have uploaded our MRP Requisition spreadsheet. "
            "Step 1: Analyse the spreadsheet — find the top 3 materials by total requisition value "
            "and the total spend by supplier. "
            "Step 2: Write a short formal procurement summary report (with a title, date, "
            "key findings section, and a recommendation) based on those numbers. "
            "Save it as 'Procurement_Summary.docx'."
        ),
        expected_agent=None,   # should use spreadsheet then document
        timeout=240,
        uploaded_files=[make_file(MRP_XLSX, "spreadsheet")],
    )
    all_results.append(r)

    # ---- TEST 11: Invoice extraction → Spreadsheet summary ----
    r = await run_e2e_test(
        test_name="Multi-Agent - Extract Invoice then Build Tracker",
        prompt=(
            "I have uploaded an invoice PDF. "
            "First extract all line items (description, quantity, unit price, total) from the invoice. "
            "Then create a spreadsheet (CSV) that tracks these line items with an additional column "
            "'Status' set to 'Unpaid' for each row, and a final TOTAL row at the bottom."
        ),
        expected_agent=None,   # should use document then spreadsheet
        timeout=240,
        uploaded_files=[make_file(INVOICE_PDF, "document")],
    )
    all_results.append(r)

    # ---- TEST 12: Sales data → Executive briefing document ----
    r = await run_e2e_test(
        test_name="Multi-Agent - Sales Analysis + Briefing Doc",
        prompt=(
            "I have uploaded our sales data CSV. "
            "Analyse the data: find total revenue, best-performing product and region, "
            "and month-over-month growth trend. "
            "Then write a one-page executive briefing document with: title, date, "
            "a 3-bullet highlights section, a risks section, and a recommended action. "
        ),
        expected_agent=None,   # should use spreadsheet + document
        timeout=240,
        uploaded_files=[make_file(SALES_CSV, "spreadsheet")],
    )
    all_results.append(r)

    # =========================================================================
    # BROWSER AGENT TESTS — real web automation via Playwright
    # NOTE: Browser agent must be pre-started on port 8090
    # NOTE: These tests require a working internet connection and Playwright
    # =========================================================================

    # ---- TEST 13: Simple page navigation & heading extraction ----
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

    # ---- TEST 14: Wikipedia article extraction ----
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

    # ---- TEST 15: Web scraping structured data ----
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

    # ---- TEST 16: Multi-step web interaction ----
    r = await run_e2e_test(
        test_name="Browser - Multi-Step: Navigate and Extract Links",
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

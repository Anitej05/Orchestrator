"""
Denisco Chemicals — Full 64-Prompt End-to-End Integration Tests
===============================================================

Runs every client-facing scenario from MASTER_PROMPT_FILE_MAPPING.xlsx through
the real LangGraph Brain → Hands → Agent cycle.

Departments covered (64 prompts):
  1–8   Sales & Enquiry       (Gmail + Spreadsheet + Document + Image)
  9–14  Production Planning   (Spreadsheet + Image)
  15–21 Procurement           (Document + Image + Spreadsheet)
  22–26 Production Execution  (Image + Spreadsheet)
  27–32 Quality Control       (Image + Spreadsheet)
  33–37 Quality Assurance     (Document + Image + Spreadsheet)
  38–42 Dispatch & Logistics  (Spreadsheet + Gmail)
  43–49 Accounts & Finance    (Spreadsheet + Document)
  50–53 R&D                   (Image + Spreadsheet)
  54–57 HR & Admin            (Spreadsheet + Image)
  58–60 Maintenance           (Spreadsheet + Image)
  61–64 EHS & MIS             (Image + Spreadsheet)

Pass criteria per test:
  • has_final_response == True
  • expected keyword(s) appear somewhere in the response (case-insensitive)
  • No timeout / ERROR status

Gmail-dependent prompts (1, 4, 6, 41) are run but marked SKIP if Gmail agent
is offline (port 8003 unreachable) to avoid blocking the full suite.

Missing source files are noted in MISSING_FILES at the top so the team can
generate them; affected prompts fall back to the closest available file.

Run:
    PYTHONUTF8=1 venv/Scripts/python backend/tests/test_denisco_e2e_all_prompts.py

    # Run a single department:
    DEPT=Sales    PYTHONUTF8=1 venv/Scripts/python backend/tests/test_denisco_e2e_all_prompts.py
    DEPT=Accounts PYTHONUTF8=1 venv/Scripts/python backend/tests/test_denisco_e2e_all_prompts.py

Pre-requisites:
    Spreadsheet Agent  — port 9000
    Document Agent     — port 8050
    Universal Agent    — port 8070
    Gmail Agent        — port 8003  (optional — skipped if offline)
"""

import sys
import io
import os
import asyncio
import logging
import socket
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Force unbuffered UTF-8 output so progress shows immediately
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Denisco-E2E")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("LLM-Inference").setLevel(logging.WARNING)

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Ensure 'backend' dir is on path for `from orchestrator.graph import ...`
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# ── Test-data directory ───────────────────────────────────────────────────────
TD = Path(__file__).resolve().parent / "test_data"

# ── File aliases (map mapping names → actual file paths) ─────────────────────
F = {
    # Spreadsheets
    "sales_orders_weekly":          TD / "sales_orders_weekly (1).xlsx",
    "sales_orders_monthly":         TD / "sales_orders_monthly.xlsx",
    "sales_orders_prod_plan":       TD / "sales_orders_for_production_plan.xlsx",
    "raw_material_stock":           TD / "raw_material_stock.xlsx",
    "bom_citric":                   TD / "bom_citric_acid_anhydrous.xlsx",
    "production_schedule":          TD / "production_schedule.xlsx",
    "production_yield":             TD / "production_yield_march.xlsx",
    "purchase_order_tracker":       TD / "purchase_order_tracker.xlsx",
    "approved_vendor_list":         TD / "approved_vendor_list.xlsx",
    "purchase_history_6months":     TD / "purchase_history_6months.xlsx",
    "mrn_vijay_hcl":                TD / "mrn_vijay_hcl.xlsx",
    "qc_results_march":             TD / "qc_results_march.xlsx",
    "qc_rejections_3months":        TD / "qc_rejections_3months.xlsx",
    "internal_audit_checklist":     TD / "internal_audit_checklist.xlsx",
    "qa_released_batches":          TD / "qa_released_batches.xlsx",
    "dispatch_register_week":       TD / "dispatch_register_week.xlsx",
    "dispatch_q4":                  TD / "dispatch_q4_data.xlsx",
    "sales_invoices_march":         TD / "sales_invoices_march.xlsx",
    "bank_statement_march":         TD / "bank_statement_march.xlsx",
    "payment_vouchers_march":       TD / "payment_vouchers_march.xlsx",
    "gstr3b_sales":                 TD / "gstr3b_sales_register.xlsx",
    "gstr3b_purchase":              TD / "gstr3b_purchase_register.xlsx",
    "accounts_payable_march":       TD / "accounts_payable_march.xlsx",
    "trial_balance_march":          TD / "trial_balance_march.xlsx",
    "trial_balance_february":       TD / "trial_balance_february.xlsx",
    "employee_salary_data":         TD / "employee_salary_data.xlsx",
    "attendance_register_march":    TD / "attendance_register_march.xlsx",
    "rd_trial_data":                TD / "rd_trial_data_5trials.xlsx",
    "rd_project_tracker":           TD / "rd_project_tracker.xlsx",
    "employee_master":              TD / "employee_master.xlsx",
    "spare_parts_inventory":        TD / "spare_parts_inventory.xlsx",
    "preventive_maintenance":       TD / "preventive_maintenance_schedule.xlsx",
    "waste_disposal_march":         TD / "waste_disposal_march.xlsx",
    "mis_sales":                    TD / "mis_sales_march.xlsx",
    "mis_production":               TD / "mis_production_march.xlsx",
    "mis_qc":                       TD / "mis_qc_march.xlsx",
    "mis_finance":                  TD / "mis_finance_march.xlsx",
    "mis_6months":                  TD / "mis_6months_combined.xlsx",
    "denisco_price_list":           TD / "denisco_price_list.xlsx",

    # PDFs
    "vendor_quotation_vijay_hcl":   TD / "vendor_quotation_vijay_hcl.pdf",
    "vendor_quotation_bharat_hcl":  TD / "vendor_quotation_bharat_hcl.pdf",
    "vendor_quotation_omega_hcl":   TD / "vendor_quotation_omega_hcl.pdf",
    "vendor_quotation_krishna_naoh":TD / "vendor_quotation_krishna_naoh.pdf",
    "vendor_quotation_anil_acetic": TD / "vendor_quotation_anil_acetic.pdf",
    "customer_po_abc":              TD / "customer_purchase_order_abc.pdf",
    "batch_production_record_bpr":  TD / "batch_production_record_bpr.pdf",
    # Missing — fallback to closest available
    "vendor_invoice_vijay":         TD / "vendor_quotation_vijay_hcl.pdf",   # ← MISSING, using quotation
    "proforma_invoice_nova":        TD / "sample_invoice.pdf",                # ← MISSING, using sample
    "sop_citric_acid":              TD / "batch_production_record_bpr.pdf",   # ← MISSING, using BPR

    # Images / JPEGs
    "customer_enquiry_screenshot":  TD / "customer_enquiry_email_screenshot.jpg",
    "customer_enquiry_for_invoice": TD / "customer_enquiry_for_invoice.jpg",
    "purchase_requisition_form":    TD / "purchase_requisition_form.jpg",
    "purchase_requisition_naoh":    TD / "purchase_requisition_naoh.jpg",
    "delivery_challan_vijay":       TD / "delivery_challan_vijay.jpg",
    "our_po_vijay":                 TD / "our_po_vijay.jpg",
    "material_rejection_note":      TD / "material_rejection_note.jpg",
    "bmr_087":                      TD / "bmr_batch_fg_2024_087.jpg",
    "bmr_088":                      TD / "bmr_batch_fg_2024_088.jpg",
    "bmr_089":                      TD / "bmr_batch_fg_2024_089.jpg",
    "bmr_090":                      TD / "bmr_batch_fg_2024_090.jpg",
    "material_issue_slip_087":      TD / "material_issue_slip_087.jpg",
    "qc_test_report_rm112":         TD / "qc_test_report_rm112.jpg",
    "inprocess_qc_report":          TD / "inprocess_qc_report_day1.jpg",
    "oos_investigation":            TD / "oos_investigation_report.jpg",
    "spec_sheet_naoh":              TD / "spec_sheet_naoh.jpg",
    "breakdown_report_dryer":       TD / "breakdown_report_dryer.jpg",
    "lab_notebook_trial1":          TD / "lab_notebook_trial1.jpg",
    "msds_sulphuric_acid":          TD / "msds_sulphuric_acid.jpg",
    "leave_app_1":                  TD / "leave_application_employee1.jpg",
    "leave_app_2":                  TD / "leave_application_employee2.jpg",
    "leave_app_3":                  TD / "leave_application_employee3.jpg",
}

# ── Files that are truly missing (log at startup) ────────────────────────────
MISSING_FILES = [
    "vendor_invoice_vijay.pdf       — using vendor_quotation_vijay_hcl.pdf as fallback",
    "proforma_invoice_nova.pdf      — using sample_invoice.pdf as fallback",
    "sop_citric_acid_process.pdf    — using batch_production_record_bpr.pdf as fallback",
    "sales_orders_weekly.xlsx       — file present as 'sales_orders_weekly (1).xlsx' (renamed)",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_file(key: str, file_type: str = None) -> Optional[Dict[str, Any]]:
    """Build an uploaded_files entry. Returns None if file doesn't exist."""
    path = F.get(key)
    if path is None:
        logger.warning(f"File key '{key}' not in F dict")
        return None
    if not path.exists():
        logger.warning(f"File missing on disk: {path}")
        return None
    if file_type is None:
        ext = path.suffix.lower()
        file_type = (
            "spreadsheet" if ext in (".xlsx", ".xls", ".csv") else
            "pdf"         if ext == ".pdf" else
            "image"       if ext in (".jpg", ".jpeg", ".png") else
            "document"
        )
    return {
        "file_name": path.name,
        "file_path": str(path),
        "file_type": file_type,
        "source": "user_upload",
    }


def files(*keys) -> List[Dict[str, Any]]:
    """Build list of file dicts, filtering out missing files."""
    return [f for k in keys if (f := make_file(k)) is not None]


def build_initial_state(prompt: str, thread_id: str = None,
                        uploaded_files: list = None) -> Dict[str, Any]:
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
        "created_files": [],
        "orchestrator_workspace": "",
        "shared_files": [],
        "shared_workspace": "",
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


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


GMAIL_ONLINE = _port_open("127.0.0.1", 8003)


def extract_results(final_state: Dict[str, Any]) -> Dict[str, Any]:
    action_history = final_state.get("action_history", [])
    agents_used = [
        {
            "agent":   e.get("resource_id", "unknown"),
            "success": e.get("success", False),
            "time_ms": round(e.get("execution_time_ms", 0)),
            "summary": (e.get("result_summary") or "")[:120],
        }
        for e in action_history
        if isinstance(e, dict) and e.get("action_type") == "agent"
    ]
    return {
        "final_response": final_state.get("final_response") or "",
        "has_final_response": bool(final_state.get("final_response")),
        "iterations":     final_state.get("iteration_count", 0),
        "total_actions":  len(action_history),
        "agents_used":    agents_used,
        "has_canvas":     final_state.get("has_canvas", False),
        "canvas_type":    final_state.get("canvas_type"),
        "error":          final_state.get("error"),
        "pending_approval": final_state.get("pending_approval", False),
    }


async def run_test(
    num: int,
    dept: str,
    name: str,
    prompt: str,
    uploaded_files: list = None,
    expected_agent: str = None,
    keywords: List[str] = None,
    timeout: float = 180.0,
    gmail_required: bool = False,
) -> Dict[str, Any]:
    """
    Run one prompt through the full orchestrator and return a result dict.

    Pass criteria:
      1. has_final_response is True
      2. All keywords (case-insensitive) appear in the response
      3. No timeout / unhandled exception
    """
    from orchestrator.graph import create_graph_with_checkpointer

    label = f"P{num:02d} [{dept}] {name}"

    # NUMS filter — run only specific prompt numbers when env var is set
    nums_filter = os.environ.get("NUMS", "").strip()
    if nums_filter:
        allowed = {int(n.strip()) for n in nums_filter.split(",") if n.strip().isdigit()}
        if num not in allowed:
            return {"num": num, "dept": dept, "name": name, "status": "SKIP",
                    "reason": "filtered by NUMS", "time": 0}

    if gmail_required and not GMAIL_ONLINE:
        print(f"\n  {label}")
        print(f"  SKIP — Gmail agent offline (port 8003)")
        return {"num": num, "dept": dept, "name": name, "status": "SKIP",
                "reason": "Gmail agent offline", "time": 0}

    thread_id = f"test_{uuid.uuid4().hex[:8]}"
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    config = {"recursion_limit": 80, "configurable": {"thread_id": thread_id}}
    state = build_initial_state(prompt, thread_id, uploaded_files=uploaded_files or [])

    print(f"\n{'─'*72}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  Prompt : \"{prompt[:90]}{'...' if len(prompt) > 90 else ''}\"", flush=True)
    if uploaded_files:
        for uf in uploaded_files:
            print(f"  File   : {uf['file_name']}  ({uf['file_type']})", flush=True)
    if expected_agent:
        print(f"  Expect : {expected_agent}", flush=True)
    if keywords:
        print(f"  Keywords: {keywords}", flush=True)

    start = time.time()
    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(state, config=config),
            timeout=timeout,
        )
        elapsed = time.time() - start
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"num": num, "dept": dept, "name": name, "status": "TIMEOUT", "time": round(elapsed, 1)}
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  ERROR after {elapsed:.1f}s: {exc}")
        return {"num": num, "dept": dept, "name": name, "status": "ERROR",
                "error": str(exc), "time": round(elapsed, 1)}

    res = extract_results(final_state)
    res.update({"num": num, "dept": dept, "name": name, "time": round(elapsed, 1)})

    # ── Pass / Fail logic ──────────────────────────────────────────────────
    reasons = []
    if not res["has_final_response"]:
        reasons.append("No final_response")

    response_lower = res["final_response"].lower()
    if keywords:
        missing_kw = [kw for kw in keywords if kw.lower() not in response_lower]
        if missing_kw:
            reasons.append(f"Missing keywords: {missing_kw}")

    if expected_agent and res["agents_used"]:
        called = [a["agent"] for a in res["agents_used"]]
        if not any(expected_agent.lower() in a.lower() for a in called):
            reasons.append(f"Expected {expected_agent}, got {called}")

    res["status"] = "PASS" if not reasons else "FAIL"
    res["fail_reasons"] = reasons

    # ── Print result ───────────────────────────────────────────────────────
    icon = "✓ PASS" if not reasons else "✗ FAIL"
    print(f"  {icon} | {elapsed:.1f}s | iter={res['iterations']} | canvas={res['canvas_type'] or 'none'}")
    for a in res["agents_used"]:
        print(f"    Agent: {a['agent']} [{'ok' if a['success'] else 'ERR'}] ({a['time_ms']}ms)")
        if a["summary"]:
            print(f"    Summary: {a['summary'][:100]}")
    if res["has_final_response"]:
        print(f"    Response: \"{res['final_response'][:150].replace(chr(10),' ')}...\"")
    if reasons:
        for r in reasons:
            print(f"    ✗ {r}")

    return res


# ══════════════════════════════════════════════════════════════════════════════
# ALL 64 PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

async def run_all() -> List[Dict[str, Any]]:
    results = []

    print("\n" + "=" * 72)
    print("  DENISCO CHEMICALS — FULL 64-PROMPT END-TO-END INTEGRATION TEST")
    print("=" * 72)
    if MISSING_FILES:
        print("\n  NOTE — files missing (fallbacks active):")
        for mf in MISSING_FILES:
            print(f"    • {mf}")
    print(f"\n  Gmail agent: {'ONLINE (port 8003)' if GMAIL_ONLINE else 'OFFLINE — Gmail prompts will SKIP'}")
    print(f"  Test-data dir: {TD}\n")

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 1 — Sales & Enquiry (Prompts 1–8)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=1, dept="Sales", name="Gmail inbox — new enquiries today",
        prompt=(
            "Read my Gmail inbox and list all new customer enquiries received today. "
            "For each one, extract the product name, required quantity, delivery timeline, "
            "and customer contact details."
        ),
        expected_agent="gmail",
        keywords=["enquir", "product", "quantity"],
        timeout=120,
        gmail_required=True,
    ))

    results.append(await run_test(
        num=2, dept="Sales", name="Enquiry screenshot → extract fields",
        prompt=(
            "I've uploaded a customer enquiry email screenshot. "
            "Extract the product name, required quantity, delivery timeline, "
            "and complete customer contact details."
        ),
        uploaded_files=files("customer_enquiry_screenshot"),
        keywords=["product", "quantity", "customer"],
        timeout=120,
    ))

    results.append(await run_test(
        num=3, dept="Sales", name="Price list + enquiry → Proforma Invoice",
        prompt=(
            "I've uploaded our latest price list spreadsheet and a customer enquiry image. "
            "Using the product and quantity from the enquiry, look up the price from our price list "
            "and prepare a professional Proforma Invoice with GST (18%) for this customer. "
            "Include unit price, quantity, taxable value, GST amount, and total."
        ),
        uploaded_files=files("denisco_price_list", "customer_enquiry_for_invoice"),
        keywords=["invoice", "gst", "total"],
        timeout=180,
    ))

    results.append(await run_test(
        num=4, dept="Sales", name="Gmail — ABC Pharma 30-day order history",
        prompt=(
            "Search my Gmail for all emails from ABC Pharma in the last 30 days. "
            "Give me a summary of their order history, what products they ordered, "
            "quantities, and flag any emails that still need a response."
        ),
        expected_agent="gmail",
        keywords=["abc pharma", "order", "email"],
        timeout=120,
        gmail_required=True,
    ))

    results.append(await run_test(
        num=5, dept="Sales", name="Sales Orders — due in 7 days + delays",
        prompt=(
            "I've uploaded this week's Sales Orders spreadsheet. "
            "Show me which orders are due in the next 7 days. "
            "Flag any that appear delayed based on the order date vs due date. "
            "Include customer name, product, quantity, and due date."
        ),
        uploaded_files=files("sales_orders_weekly"),
        keywords=["due", "delay", "order"],
        timeout=180,
    ))

    results.append(await run_test(
        num=6, dept="Sales", name="Follow-up email — no response 5 days",
        prompt=(
            "Draft a professional follow-up email for a customer who received our quotation "
            "5 days ago but hasn't responded. "
            "Product: Sodium Acetate, Quantity: 500kg. "
            "Be polite, professional, and create urgency without being pushy."
        ),
        keywords=["sodium acetate", "500", "follow"],
        timeout=150,
    ))

    results.append(await run_test(
        num=7, dept="Sales", name="Customer PO PDF → verify vs quotation",
        prompt=(
            "I've uploaded a customer Purchase Order PDF. "
            "Extract all line items with product name, quantity, unit price, and total. "
            "Also check if the quantities and pricing look consistent with standard terms "
            "and create a summary for the sales manager."
        ),
        uploaded_files=files("customer_po_abc"),
        keywords=["quantity", "price", "product"],
        timeout=180,
    ))

    results.append(await run_test(
        num=8, dept="Sales", name="Monthly SO → top 5 customers by value",
        prompt=(
            "I've uploaded this month's Sales Orders spreadsheet. "
            "Analyze it and show me our top 5 customers ranked by total order value this month. "
            "Include order count and total value for each."
        ),
        uploaded_files=files("sales_orders_monthly"),
        expected_agent="spreadsheet",
        keywords=["top", "customer", "value"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 2 — Production Planning (Prompts 9–14)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=9, dept="Prod Plan", name="SO + Stock → material shortages",
        prompt=(
            "I've uploaded our current Sales Orders spreadsheet and Raw Material stock sheet. "
            "Identify which raw materials are short for next week's production based on the sales orders. "
            "Show material name, required quantity, available stock, and shortage amount."
        ),
        uploaded_files=files("sales_orders_prod_plan", "raw_material_stock"),
        expected_agent="spreadsheet",
        keywords=["shortage", "stock", "material"],
        timeout=180,
    ))

    results.append(await run_test(
        num=10, dept="Prod Plan", name="BOM + Stock → RM needed for 1000kg",
        prompt=(
            "I've uploaded our BOM for Citric Acid Anhydrous and our current raw material stock sheet. "
            "Calculate the exact quantity of each raw material needed to produce 1000kg of Citric Acid Anhydrous. "
            "Show required qty, available stock, and whether we have enough or are short."
        ),
        uploaded_files=files("bom_citric", "raw_material_stock"),
        expected_agent="spreadsheet",
        keywords=["bom", "1000", "material"],
        timeout=240,
    ))

    results.append(await run_test(
        num=11, dept="Prod Plan", name="Production schedule → delayed batches",
        prompt=(
            "I've uploaded our production schedule spreadsheet. "
            "Tell me which batches are running behind schedule. "
            "Include batch number, product, planned date, current status, and any notes about the likely reason."
        ),
        uploaded_files=files("production_schedule"),
        expected_agent="spreadsheet",
        keywords=["delay", "batch", "schedule"],
        timeout=180,
    ))

    results.append(await run_test(
        num=12, dept="Prod Plan", name="Sales Orders → generate production plan",
        prompt=(
            "I've uploaded this month's Sales Orders spreadsheet. "
            "Generate a production plan table showing: product name, total quantity ordered, "
            "proposed production start date, and estimated completion date. "
            "Assume 5 working days per batch of 500kg."
        ),
        uploaded_files=files("sales_orders_prod_plan"),
        expected_agent="spreadsheet",
        keywords=["production", "start", "completion"],
        timeout=180,
    ))

    results.append(await run_test(
        num=13, dept="Prod Plan", name="PR image → extract items + format doc",
        prompt=(
            "I've uploaded a Purchase Requisition form image. "
            "Extract all item details including material name, quantity, unit, required date, "
            "and department. Format them into a clean PR document."
        ),
        uploaded_files=files("purchase_requisition_form"),
        keywords=["material", "quantity", "purchase requisition"],
        timeout=120,
    ))

    results.append(await run_test(
        num=14, dept="Prod Plan", name="Prod orders + stock → start vs blocked",
        prompt=(
            "I've uploaded our production orders spreadsheet and current stock level report. "
            "Tell me which production orders can start immediately (all materials available) "
            "and which are blocked due to material shortage. "
            "Show blocking material and shortage quantity."
        ),
        uploaded_files=files("sales_orders_prod_plan", "raw_material_stock"),
        expected_agent="spreadsheet",
        keywords=["start", "block", "material"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 3 — Procurement (Prompts 15–21)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=15, dept="Procure", name="3 HCl vendor PDFs → compare + recommend",
        prompt=(
            "I've uploaded three vendor quotation PDFs for Hydrochloric Acid (HCl). "
            "Compare them side by side on: price per kg, lead time, payment terms, GST rate, and quality grade. "
            "Recommend the best vendor with clear justification."
        ),
        uploaded_files=files("vendor_quotation_vijay_hcl",
                              "vendor_quotation_bharat_hcl",
                              "vendor_quotation_omega_hcl"),
        expected_agent="document",
        keywords=["vendor", "recommend", "price"],
        timeout=240,
    ))

    results.append(await run_test(
        num=16, dept="Procure", name="Vendor list + PR → draft enquiry emails",
        prompt=(
            "I've uploaded our approved vendor list spreadsheet and a Purchase Requisition image for NaOH. "
            "Draft vendor enquiry emails to the top 3 vendors for this material. "
            "Include material specification, quantity required, required delivery date, and payment terms."
        ),
        uploaded_files=files("approved_vendor_list", "purchase_requisition_naoh"),
        keywords=["vendor", "enquiry", "naoh"],
        timeout=180,
    ))

    results.append(await run_test(
        num=17, dept="Procure", name="Delivery challan vs PO → flag discrepancy",
        prompt=(
            "I've uploaded a supplier's Delivery Challan and our original Purchase Order as images. "
            "Check if the quantities and items delivered match our PO. "
            "Flag any discrepancies in quantity, item, or batch number."
        ),
        uploaded_files=files("delivery_challan_vijay", "our_po_vijay"),
        keywords=["quantity", "deliver"],
        timeout=120,
    ))

    results.append(await run_test(
        num=18, dept="Procure", name="PO tracker → overdue + vendor delays",
        prompt=(
            "I've uploaded this month's purchase order tracker spreadsheet. "
            "Tell me: which POs are overdue, which vendors have the most delays, "
            "and what is the total outstanding value of overdue POs."
        ),
        uploaded_files=files("purchase_order_tracker"),
        expected_agent="spreadsheet",
        keywords=["overdue", "vendor", "outstanding"],
        timeout=180,
    ))

    results.append(await run_test(
        num=19, dept="Procure", name="Rejection note image → formal rejection letter",
        prompt=(
            "I've uploaded a Material Rejection Note image. "
            "Extract the rejection details and draft a formal rejection letter with debit note details "
            "to send to the vendor. Include batch number, material, quantity rejected, reason, and debit amount."
        ),
        uploaded_files=files("material_rejection_note"),
        keywords=["rejection", "vendor", "debit"],
        timeout=120,
    ))

    results.append(await run_test(
        num=20, dept="Procure", name="6-month purchase data → spend analysis",
        prompt=(
            "I've uploaded our purchase data for the last 6 months. "
            "Analyze spending by vendor and material category. "
            "Show total spend per vendor, top materials purchased, and identify where we can negotiate better rates."
        ),
        uploaded_files=files("purchase_history_6months"),
        expected_agent="spreadsheet",
        keywords=["vendor", "total", "purchase"],
        timeout=180,
    ))

    results.append(await run_test(
        num=21, dept="Procure", name="Invoice + PO + MRN → 3-way match",
        prompt=(
            "I've uploaded a vendor invoice PDF, our Material Receipt Note spreadsheet, "
            "and our original Purchase Order image. "
            "Perform a three-way match: check if invoice quantities, prices, and total match "
            "both the PO and the MRN. Confirm if the invoice is cleared for payment or flag issues."
        ),
        uploaded_files=files("vendor_invoice_vijay", "mrn_vijay_hcl", "our_po_vijay"),
        keywords=["vendor", "quantity", "invoice"],
        timeout=240,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 4 — Production Execution (Prompts 22–26)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=22, dept="Prod Exec", name="BMR image → extract all parameters",
        prompt=(
            "I've uploaded a Batch Manufacturing Record (BMR) image from the shop floor. "
            "Extract all recorded parameters including: batch number, product name, "
            "raw materials used with quantities, process parameters (temperature, time, pH), "
            "and operator sign-offs. Present as a clean structured report."
        ),
        uploaded_files=files("bmr_087"),
        keywords=["batch", "parameter"],
        timeout=120,
    ))

    results.append(await run_test(
        num=23, dept="Prod Exec", name="Issue slip vs BOM → flag overissue",
        prompt=(
            "I've uploaded a Material Issue Slip image and our BOM spreadsheet for Citric Acid Anhydrous. "
            "Compare the quantities issued on the slip against the BOM requirements for the batch. "
            "Flag any material that was over-issued or under-issued."
        ),
        uploaded_files=files("material_issue_slip_087", "bom_citric"),
        keywords=["bom", "material", "issu"],
        timeout=150,
    ))

    results.append(await run_test(
        num=24, dept="Prod Exec", name="Yield sheet → batches with >5% yield loss",
        prompt=(
            "I've uploaded this week's production execution spreadsheet. "
            "Calculate actual yield vs planned yield for each batch. "
            "Show which batches had yield loss above 5% and flag them with their yield loss percentage."
        ),
        uploaded_files=files("production_yield"),
        expected_agent="spreadsheet",
        keywords=["yield", "batch", "%"],
        timeout=180,
    ))

    results.append(await run_test(
        num=25, dept="Prod Exec", name="Generate Finished Goods Transfer Note",
        prompt=(
            "Generate a Finished Goods Transfer Note for the following batch:\n"
            "Batch No: FG-2024-087\n"
            "Product: Citric Acid Anhydrous\n"
            "Quantity: 800kg\n"
            "From: Production to FG Store\n"
            "Date: today\n"
            "Format it as a proper document with all standard fields."
        ),
        keywords=["fg-2024-087", "citric acid", "800"],
        timeout=90,
    ))

    results.append(await run_test(
        num=26, dept="Prod Exec", name="4 BMR images → compile summary sheet",
        prompt=(
            "I've uploaded 4 Batch Manufacturing Record images from this week (batches 087–090). "
            "Extract the batch details from each one and compile them into a single summary table with: "
            "batch number, product, start date, end date, planned qty, actual yield, and final status."
        ),
        uploaded_files=files("bmr_087", "bmr_088", "bmr_089", "bmr_090"),
        keywords=["batch", "yield", "summary"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 5 — Quality Control (Prompts 27–32)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=27, dept="QC", name="QC test report image → pass or fail",
        prompt=(
            "I've uploaded a QC Test Report image for an incoming raw material (Batch RM-2024-112). "
            "Extract all test parameters, recorded results, and acceptance specifications. "
            "Tell me clearly: does this batch PASS or FAIL QC? Highlight any out-of-spec parameters."
        ),
        uploaded_files=files("qc_test_report_rm112"),
        keywords=["pass", "fail", "batch"],
        timeout=120,
    ))

    results.append(await run_test(
        num=28, dept="QC", name="Generate COA for Batch RM-2024-112",
        prompt=(
            "Generate a Certificate of Analysis (COA) document for the following batch:\n"
            "Batch No: RM-2024-112\n"
            "Material: Acetic Acid\n"
            "Test Results:\n"
            "  - Assay: 99.8% (Spec: ≥99.5%)\n"
            "  - Water content: 0.1% (Spec: ≤0.2%)\n"
            "  - Heavy metals: <10ppm (Spec: ≤10ppm)\n"
            "  - Appearance: Clear colourless liquid (Spec: Clear colourless)\n"
            "Status: PASSED\n"
            "Format as a professional COA document."
        ),
        keywords=["coa", "assay", "passed"],
        timeout=90,
    ))

    results.append(await run_test(
        num=29, dept="QC", name="QC results → failures by vendor",
        prompt=(
            "I've uploaded our QC results spreadsheet for this month. "
            "Show me which raw material batches failed QC, which vendors supplied them, "
            "and calculate the rejection rate per vendor."
        ),
        uploaded_files=files("qc_results_march"),
        expected_agent="spreadsheet",
        keywords=["fail", "vendor", "rejection"],
        timeout=180,
    ))

    results.append(await run_test(
        num=30, dept="QC", name="Spec sheet image → parameter table",
        prompt=(
            "I've uploaded a specification sheet image for Sodium Hydroxide (NaOH). "
            "Extract all quality parameters and their acceptance limits into a clean structured table. "
            "Include parameter name, specification/limit, test method if mentioned."
        ),
        uploaded_files=files("spec_sheet_naoh"),
        keywords=["naoh", "specification", "parameter"],
        timeout=120,
    ))

    results.append(await run_test(
        num=31, dept="QC", name="In-process QC image → all limits OK?",
        prompt=(
            "I've uploaded an in-process QC report image from today's production. "
            "Extract all check points and recorded values. "
            "Tell me if all checks are within acceptable limits and flag any borderline or out-of-spec parameters."
        ),
        uploaded_files=files("inprocess_qc_report"),
        keywords=["limit", "parameter", "check"],
        timeout=120,
    ))

    results.append(await run_test(
        num=32, dept="QC", name="3-month rejections → failure mode analysis",
        prompt=(
            "I've uploaded 3 months of QC rejection data. "
            "Analyze the most common failure modes across materials and vendors. "
            "Which vendor has the highest rejection rate? "
            "What are the top 3 root causes and your recommendations?"
        ),
        uploaded_files=files("qc_rejections_3months"),
        expected_agent="spreadsheet",
        keywords=["failure", "vendor", "rejection"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 6 — Quality Assurance (Prompts 33–37)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=33, dept="QA", name="BPR PDF → sections filled + signed?",
        prompt=(
            "I've uploaded a Batch Production Record (BPR) as a PDF. "
            "Review it and tell me: are all mandatory sections filled in? "
            "Are all required signatures present? Flag any missing entries before QA release."
        ),
        uploaded_files=files("batch_production_record_bpr"),
        expected_agent="document",
        keywords=["section", "signature", "release"],
        timeout=180,
    ))

    results.append(await run_test(
        num=34, dept="QA", name="Draft Deviation Report — temp excursion",
        prompt=(
            "Draft a formal Deviation Report for the following event:\n"
            "Batch: FG-2024-091\n"
            "Product: Citric Acid Anhydrous\n"
            "Deviation: Drying temperature exceeded specified range (70°C ± 5°C) "
            "by 5°C for approximately 20 minutes during the drying stage.\n"
            "Include: deviation description, potential impact, immediate action taken, and proposed CAPA."
        ),
        keywords=["deviation", "fg-2024-091", "temperature", "capa"],
        timeout=90,
    ))

    results.append(await run_test(
        num=35, dept="QA", name="OOS image → summarize + draft CAPA",
        prompt=(
            "I've uploaded an Out-of-Specification (OOS) investigation report image. "
            "Summarize the key findings of the investigation. "
            "Then draft a CAPA (Corrective and Preventive Action) response document "
            "addressing the root cause identified."
        ),
        uploaded_files=files("oos_investigation"),
        keywords=["oos", "capa", "root cause"],
        timeout=150,
    ))

    results.append(await run_test(
        num=36, dept="QA", name="Audit checklist → non-compliance + action plan",
        prompt=(
            "I've uploaded our internal audit checklist spreadsheet. "
            "Identify all non-compliant items. Group them by department. "
            "Draft an action plan with responsible person and target deadline for each non-compliance."
        ),
        uploaded_files=files("internal_audit_checklist"),
        expected_agent="spreadsheet",
        keywords=["non-compli", "action", "department"],
        timeout=300,
    ))

    results.append(await run_test(
        num=37, dept="QA", name="SOP vs BMR → deviation check",
        prompt=(
            "I've uploaded an SOP document (PDF) and a BMR image for a batch. "
            "Check if the batch followed all the SOP steps as recorded in the BMR. "
            "Flag any deviations from the SOP procedure."
        ),
        uploaded_files=files("sop_citric_acid", "bmr_087"),
        keywords=["sop", "step", "procedure"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 7 — Dispatch & Logistics (Prompts 38–42)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=38, dept="Dispatch", name="QA batches + SO → packing list + challan",
        prompt=(
            "I've uploaded a QA-released batch list spreadsheet and a customer Sales Order spreadsheet. "
            "Prepare a Packing List and Delivery Challan for dispatching 500kg of Citric Acid. "
            "Include batch number, quantity, customer details, and delivery address."
        ),
        uploaded_files=files("qa_released_batches", "sales_orders_prod_plan"),
        expected_agent="spreadsheet",
        keywords=["packing", "challan", "citric acid"],
        timeout=300,
    ))

    results.append(await run_test(
        num=39, dept="Dispatch", name="Dispatch register → pending transporter",
        prompt=(
            "I've uploaded this week's dispatch register spreadsheet. "
            "Tell me which dispatches are still pending transporter assignment. "
            "Also identify any pending E-Way Bills and flag them."
        ),
        uploaded_files=files("dispatch_register_week"),
        expected_agent="spreadsheet",
        keywords=["pending", "transporter"],
        timeout=180,
    ))

    results.append(await run_test(
        num=40, dept="Dispatch", name="Customer SO → generate Tax Invoice PDF",
        prompt=(
            "I've uploaded a customer Sales Order PDF and our sales invoice data spreadsheet. "
            "Generate a professional Tax Invoice with: customer details, HSN codes, "
            "taxable value, IGST/CGST/SGST breakup, and total amount in words."
        ),
        uploaded_files=files("customer_po_abc", "sales_invoices_march"),
        keywords=["tax invoice", "gst", "hsn"],
        timeout=180,
    ))

    results.append(await run_test(
        num=41, dept="Dispatch", name="Gmail — transporter/customer delay emails",
        prompt=(
            "Search my Gmail for any emails from transporters or customers about delayed shipments "
            "or delivery complaints received this week. Summarize each issue with sender, subject, "
            "and key complaint."
        ),
        expected_agent="gmail",
        keywords=["delay", "shipment", "email"],
        timeout=120,
        gmail_required=True,
    ))

    results.append(await run_test(
        num=42, dept="Dispatch", name="Q4 dispatch data → on-time + TAT analysis",
        prompt=(
            "I've uploaded last quarter's dispatch data spreadsheet. "
            "Analyze on-time delivery performance: what percentage of orders were delivered on time? "
            "Show top customers by dispatch volume and calculate average turnaround time (TAT)."
        ),
        uploaded_files=files("dispatch_q4"),
        expected_agent="spreadsheet",
        keywords=["dispatch", "deliver", "turnaround"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 8 — Accounts & Finance (Prompts 43–49)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=43, dept="Accounts", name="Invoice + MRN + PO → 3-way match",
        prompt=(
            "I've uploaded a vendor invoice PDF, our MRN spreadsheet, and our Purchase Order image. "
            "Perform a three-way match. Confirm if the invoice value, quantities, and items "
            "match both the PO and MRN. Is it cleared for payment? Flag any mismatch."
        ),
        uploaded_files=files("vendor_invoice_vijay", "mrn_vijay_hcl", "our_po_vijay"),
        keywords=["match", "invoice", "payment"],
        timeout=240,
    ))

    results.append(await run_test(
        num=44, dept="Accounts", name="Bank statement vs vouchers → reconcile",
        prompt=(
            "I've uploaded our bank statement for March and our payment vouchers spreadsheet. "
            "Reconcile them — which payments in the bank statement don't match any voucher? "
            "List all unmatched items with amount and date."
        ),
        uploaded_files=files("bank_statement_march", "payment_vouchers_march"),
        expected_agent="spreadsheet",
        keywords=["unmatched", "reconcil", "payment"],
        timeout=180,
    ))

    results.append(await run_test(
        num=45, dept="Accounts", name="Sales invoices → GSTR-1 liability",
        prompt=(
            "I've uploaded our sales invoice data for March. "
            "Calculate our GSTR-1 liability: show total taxable value, IGST, CGST, and SGST "
            "broken down by tax rate slab (5%, 12%, 18%, 28%)."
        ),
        uploaded_files=files("sales_invoices_march"),
        expected_agent="spreadsheet",
        keywords=["gst", "taxable", "slab"],
        timeout=180,
    ))

    results.append(await run_test(
        num=46, dept="Accounts", name="Purchase + Sales registers → GSTR-3B",
        prompt=(
            "I've uploaded our purchase and sales invoice registers for March. "
            "Calculate net GST payable for GSTR-3B filing: "
            "Output GST (from sales) minus Input Tax Credit (from purchases). "
            "Show the calculation clearly."
        ),
        uploaded_files=files("gstr3b_sales", "gstr3b_purchase"),
        expected_agent="spreadsheet",
        keywords=["output gst", "input", "net gst"],
        timeout=180,
    ))

    results.append(await run_test(
        num=47, dept="Accounts", name="Accounts payable → overdue + priority",
        prompt=(
            "I've uploaded our accounts payable spreadsheet for March. "
            "Tell me which vendor payments are overdue, total outstanding amount, "
            "and which vendors should be paid this week (prioritized by days overdue and amount)."
        ),
        uploaded_files=files("accounts_payable_march"),
        expected_agent="spreadsheet",
        keywords=["overdue", "vendor", "payment"],
        timeout=180,
    ))

    results.append(await run_test(
        num=48, dept="Accounts", name="Trial balance → P&L + expense spikes",
        prompt=(
            "I've uploaded our Trial Balance for March and February. "
            "Generate a P&L summary for March. "
            "Compare with February and flag any expense lines that are significantly higher "
            "than last month (more than 20% increase)."
        ),
        uploaded_files=files("trial_balance_march", "trial_balance_february"),
        expected_agent="spreadsheet",
        keywords=["p&l", "expense", "increase"],
        timeout=300,
    ))

    results.append(await run_test(
        num=49, dept="Accounts", name="Salary + attendance → full payroll",
        prompt=(
            "I've uploaded employee salary data and attendance register for March. "
            "Calculate for each employee: gross salary, PF deduction (12% of basic), "
            "ESI deduction (0.75% of gross), TDS, and net payable salary."
        ),
        uploaded_files=files("employee_salary_data", "attendance_register_march"),
        expected_agent="spreadsheet",
        keywords=["gross", "pf", "net", "salary"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 9 — R&D (Prompts 50–53)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=50, dept="R&D", name="Lab notebook image → digital report",
        prompt=(
            "I've uploaded a Lab Notebook image with handwritten trial data. "
            "Extract all recorded parameters, quantities, observations, and results "
            "into a clean digital lab report format."
        ),
        uploaded_files=files("lab_notebook_trial1"),
        keywords=["trial", "parameter", "result"],
        timeout=120,
    ))

    results.append(await run_test(
        num=51, dept="R&D", name="5 trial sheets → best yield + purity",
        prompt=(
            "I've uploaded analytical data sheets from 5 lab trials for a new product. "
            "Compare results across all 5 trials. "
            "Which trial gave the best yield? Which gave the best purity? "
            "Show a comparison table and your recommendation."
        ),
        uploaded_files=files("rd_trial_data"),
        expected_agent="spreadsheet",
        keywords=["trial", "yield", "purity"],
        timeout=180,
    ))

    results.append(await run_test(
        num=52, dept="R&D", name="Draft Technology Transfer Document",
        prompt=(
            "I've uploaded our R&D trial data for the best lab-scale process. "
            "Draft a Technology Transfer Document to scale from lab scale (1kg) to pilot scale (50kg). "
            "Include: process summary, key parameters to monitor, scale-up factors, "
            "equipment requirements, and safety considerations."
        ),
        uploaded_files=files("rd_trial_data"),
        keywords=["technology transfer", "scale", "pilot"],
        timeout=180,
    ))

    results.append(await run_test(
        num=53, dept="R&D", name="R&D project tracker → scale-up ready",
        prompt=(
            "I've uploaded our R&D project tracker spreadsheet. "
            "Give me a status summary of all active projects. "
            "Which projects are ready for production scale-up? "
            "Which are still in early stages? Show current stage and next milestone."
        ),
        uploaded_files=files("rd_project_tracker"),
        expected_agent="spreadsheet",
        keywords=["project", "status", "stage"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 10 — HR & Admin (Prompts 54–57)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=54, dept="HR", name="Attendance register → present/absent/OT",
        prompt=(
            "I've uploaded the attendance register for March. "
            "Calculate for each employee: total present days, absent days, late arrivals, "
            "and overtime hours. Show a per-employee summary table."
        ),
        uploaded_files=files("attendance_register_march"),
        expected_agent="spreadsheet",
        keywords=["present", "absent", "overtime"],
        timeout=180,
    ))

    results.append(await run_test(
        num=55, dept="HR", name="3 leave application images → approval summary",
        prompt=(
            "I've uploaded 3 leave application images from employees. "
            "Extract from each: employee name, employee ID if visible, leave type, "
            "leave dates, and reason. Create a leave approval summary table."
        ),
        uploaded_files=files("leave_app_1", "leave_app_2", "leave_app_3"),
        keywords=["leave", "employee", "date"],
        timeout=150,
    ))

    results.append(await run_test(
        num=56, dept="HR", name="Employee master → missing PF/ESI + appraisal due",
        prompt=(
            "I've uploaded our employee master spreadsheet. "
            "Tell me: which employees have missing PF or ESI registrations? "
            "Which employees are due for their annual appraisal this month?"
        ),
        uploaded_files=files("employee_master"),
        expected_agent="spreadsheet",
        keywords=["pf", "esi", "appraisal"],
        timeout=180,
    ))

    results.append(await run_test(
        num=57, dept="HR", name="Draft appointment letter — QC Analyst",
        prompt=(
            "Draft a formal appointment letter for a new employee with these details:\n"
            "Position: Quality Control Analyst\n"
            "CTC: ₹28,000 per month\n"
            "Joining Date: 1st of next month\n"
            "Probation: 3 months\n"
            "Company: Denisco Chemicals Pvt. Ltd.\n"
            "Include standard terms for a chemical manufacturing company (confidentiality, notice period)."
        ),
        keywords=["appointment", "quality control", "28,000"],
        timeout=90,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 11 — Maintenance (Prompts 58–60)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=58, dept="Maint", name="PM schedule → due + overdue + work orders",
        prompt=(
            "I've uploaded our Preventive Maintenance schedule spreadsheet. "
            "Tell me which equipment is due for maintenance this week and which is overdue. "
            "Generate a work order list with equipment ID, task, and priority."
        ),
        uploaded_files=files("preventive_maintenance"),
        expected_agent="spreadsheet",
        keywords=["maintenance", "equipment"],
        timeout=180,
    ))

    results.append(await run_test(
        num=59, dept="Maint", name="Breakdown report image → maintenance log",
        prompt=(
            "I've uploaded a breakdown report image from the shop floor. "
            "Extract: equipment name and ID, nature of breakdown, downtime duration, "
            "root cause if mentioned, spares used, and action taken. "
            "Format it as a formal maintenance log entry."
        ),
        uploaded_files=files("breakdown_report_dryer"),
        keywords=["breakdown", "equipment", "downtime"],
        timeout=120,
    ))

    results.append(await run_test(
        num=60, dept="Maint", name="Spare parts → below reorder + GPR",
        prompt=(
            "I've uploaded our spare parts inventory spreadsheet. "
            "Identify which critical spares are below their reorder level. "
            "Then generate a General Purchase Request (GPR) listing each item, "
            "current stock, reorder quantity, and estimated value."
        ),
        uploaded_files=files("spare_parts_inventory"),
        expected_agent="spreadsheet",
        keywords=["reorder", "inventory"],
        timeout=180,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # DEPT 12 — EHS & MIS (Prompts 61–64)
    # ══════════════════════════════════════════════════════════════════════

    results.append(await run_test(
        num=61, dept="EHS/MIS", name="MSDS image → hazards + PPE + storage",
        prompt=(
            "I've uploaded an MSDS image for Sulphuric Acid. "
            "Extract and present in a structured table: "
            "hazard classifications, handling precautions, required PPE, "
            "first aid measures, and storage conditions."
        ),
        uploaded_files=files("msds_sulphuric_acid"),
        keywords=["hazard", "ppe", "storage", "sulphuric"],
        timeout=120,
    ))

    results.append(await run_test(
        num=62, dept="EHS/MIS", name="Waste log → category totals vs PCB limit",
        prompt=(
            "I've uploaded our waste disposal logs for March. "
            "Summarize total waste generated by category. "
            "Compare against PCB (Pollution Control Board) limits. "
            "Flag any waste category that exceeded its permitted limit."
        ),
        uploaded_files=files("waste_disposal_march"),
        expected_agent="spreadsheet",
        keywords=["waste", "pcb", "limit"],
        timeout=180,
    ))

    results.append(await run_test(
        num=63, dept="EHS/MIS", name="4 MIS sheets → full March MIS report",
        prompt=(
            "I've uploaded our Sales, Production, QC, and Finance data spreadsheets for March. "
            "Generate a complete MIS (Management Information System) report with:\n"
            "1. Sales KPIs: revenue, top products, top customers\n"
            "2. Production KPIs: output, yield, batch efficiency\n"
            "3. QC KPIs: rejection rate, pass rate, vendor quality\n"
            "4. Finance KPIs: revenue vs expenses, outstanding receivables, payables\n"
            "Include month-over-month variance where data allows."
        ),
        uploaded_files=files("mis_sales", "mis_production", "mis_qc", "mis_finance"),
        expected_agent="spreadsheet",
        keywords=["mis", "kpi", "sales"],
        timeout=300,
    ))

    results.append(await run_test(
        num=64, dept="EHS/MIS", name="6-month MIS → trend analysis",
        prompt=(
            "I've uploaded MIS reports from the last 6 months (combined spreadsheet). "
            "Analyze trends in: production output, sales revenue, QC rejection rates, "
            "and procurement costs. Identify positive and negative trends "
            "and provide a narrative summary suitable for a management review meeting."
        ),
        uploaded_files=files("mis_6months"),
        expected_agent="spreadsheet",
        keywords=["trend", "production", "revenue", "management"],
        timeout=240,
    ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: List[Dict[str, Any]]):
    print("\n\n" + "=" * 72)
    print("  DENISCO E2E TEST SUMMARY")
    print("=" * 72)

    by_dept: Dict[str, List] = {}
    for r in results:
        by_dept.setdefault(r["dept"], []).append(r)

    total = len(results)
    passed  = sum(1 for r in results if r.get("status") == "PASS")
    failed  = sum(1 for r in results if r.get("status") == "FAIL")
    skipped = sum(1 for r in results if r.get("status") == "SKIP")
    timeout = sum(1 for r in results if r.get("status") == "TIMEOUT")
    errored = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"\n  Total  : {total}")
    print(f"  PASS   : {passed}  ({100*passed//total if total else 0}%)")
    print(f"  FAIL   : {failed}")
    print(f"  SKIP   : {skipped}  (Gmail offline)")
    print(f"  TIMEOUT: {timeout}")
    print(f"  ERROR  : {errored}")

    print(f"\n  {'Dept':<14} {'P':>4} {'F':>4} {'S':>4} {'T':>4}")
    print(f"  {'─'*14} {'─'*4} {'─'*4} {'─'*4} {'─'*4}")
    for dept, dept_results in by_dept.items():
        p = sum(1 for r in dept_results if r.get("status") == "PASS")
        f = sum(1 for r in dept_results if r.get("status") == "FAIL")
        s = sum(1 for r in dept_results if r.get("status") == "SKIP")
        t = sum(1 for r in dept_results if r.get("status") in ("TIMEOUT", "ERROR"))
        print(f"  {dept:<14} {p:>4} {f:>4} {s:>4} {t:>4}")

    # Failed tests detail
    failed_tests = [r for r in results if r.get("status") in ("FAIL", "TIMEOUT", "ERROR")]
    if failed_tests:
        print(f"\n  FAILURES / ERRORS:")
        for r in failed_tests:
            reasons = r.get("fail_reasons") or [r.get("error", r.get("status", "?"))]
            print(f"    P{r['num']:02d} [{r['dept']}] {r['name']}")
            for reason in reasons:
                print(f"         ✗ {reason}")

    print("\n" + "=" * 72)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    dept_filter = os.environ.get("DEPT", "").strip().lower()

    all_results = await run_all()

    # Apply department filter for post-run display if needed
    if dept_filter:
        display = [r for r in all_results if dept_filter in r.get("dept", "").lower()]
    else:
        display = all_results

    print_summary(display)
    return all_results


if __name__ == "__main__":
    asyncio.run(main())

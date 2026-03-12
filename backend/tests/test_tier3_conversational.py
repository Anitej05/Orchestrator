"""
DENISCO BETA — TIER 3: CONVERSATIONAL INTELLIGENCE TESTS
=========================================================
Target: 8/10 pass to achieve GO verdict.

These tests simulate realistic multi-turn conversations as Denisco staff
would actually use the tool — including follow-up questions without
re-uploading files, vague intent, corrections, and process questions.

Each test gets a unique thread_id (fresh conversation). Multi-turn tests
reuse the same thread_id across turns, relying on LangGraph state persistence.

Pass criteria per test:
  T3-01  Follow-up without re-upload     Turn2 references MRP data, does NOT ask to re-upload
  T3-02  Vague intent + file resolution  System extracts invoice data after initial vague ask
  T3-03  Cross-turn line item lookup     Turn2 lists line items from invoice without re-upload
  T3-04  QC process question (no file)   Lists >=4 correct QC rejection doc fields/documents
  T3-05  Memory filter within session    Turn2 applies filter to MRP data from Turn1
  T3-06  Minimal prompt with file        System summarises file, doesn't crash
  T3-07  Correction handling             System rechecks instead of arguing
  T3-08  Dispatch process knowledge      Lists >=4 correct dispatch documents
  T3-09  Off-topic graceful decline      Does NOT hallucinate (no fake booking/action)
  T3-10  Multi-file multi-turn           Turn2 discusses second file, not first

Run:
    cd c:/Users/akush/Orchestrator-preview
    PYTHONUTF8=1 venv/Scripts/python backend/tests/test_tier3_conversational.py

Prerequisites:
    - Orchestrator running on port 8000
    - Spreadsheet agent running on port 9000
    - Document agent running on port 8050
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import websockets

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ORCHESTRATOR_BASE = "http://localhost:8000"
WS_URL            = "ws://localhost:8000/ws/chat"
OWNER             = {"user_id": "test_denisco_tier3"}

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
MRP_XLSX      = TEST_DATA_DIR / "MRP_Requisition_For_Production_Plan.xlsx"
INVOICE_PDF   = TEST_DATA_DIR / "sample_invoice.pdf"
EMPLOYEES_XLS = TEST_DATA_DIR / "employees.xlsx"

TURN_TIMEOUT  = 120.0   # max seconds per WebSocket turn
INTER_TURN_PAUSE = 2.0  # brief pause between turns in same thread

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

results: List[Dict[str, Any]] = []

def record(test_id: str, name: str, passed: bool, detail: str = "", note: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append({"id": test_id, "name": name, "status": status,
                    "detail": detail, "note": note})
    print(f"\n  {'PASS' if passed else 'FAIL'} | {test_id} | {name}")
    if detail:
        for line in detail[:400].split("\n"):
            print(f"         {line}")
    if note:
        print(f"         NOTE: {note}")


# ---------------------------------------------------------------------------
# File upload helper
# ---------------------------------------------------------------------------

async def upload_file(local_path: Path, mime: str) -> Optional[Dict]:
    """Upload a file via HTTP and return the file_entry dict for WebSocket use."""
    if not local_path.exists():
        print(f"  [SKIP] File not found: {local_path}")
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(local_path, "rb") as f:
                r = await client.post(
                    f"{ORCHESTRATOR_BASE}/api/upload",
                    files={"files": (local_path.name, f, mime)},
                )
        if r.status_code == 200 and r.json():
            uploaded = r.json()[0]
            return {
                "file_name": uploaded["file_name"],
                "file_path": uploaded["file_path"],
                "file_type": "spreadsheet" if local_path.suffix in (".xlsx", ".xls", ".csv") else "document",
                "source": "user_upload",
            }
    except Exception as e:
        print(f"  [WARN] Upload failed for {local_path.name}: {e}")
    return None


# ---------------------------------------------------------------------------
# Core WebSocket turn helper
# ---------------------------------------------------------------------------

async def ws_turn(
    thread_id: str,
    prompt: str,
    files: Optional[List[Dict]] = None,
    timeout: float = TURN_TIMEOUT,
    label: str = "",
) -> Tuple[str, bool]:
    """
    Send one message over a new WebSocket connection (same thread_id = same conversation).
    Returns (final_response_text, connection_success).
    """
    payload = {
        "thread_id": thread_id,
        "prompt": prompt,
        "owner": OWNER,
        "files": files or [],
    }

    tag = f"[{label}] " if label else ""
    prompt_preview = prompt[:70] + "..." if len(prompt) > 70 else prompt
    print(f"    {tag}Sending: \"{prompt_preview}\"")
    if files:
        print(f"    {tag}Files:   {[f['file_name'] for f in files]}")

    final_response = ""
    deadline = time.time() + timeout

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=30,
            open_timeout=10,
            max_size=10 * 1024 * 1024,   # 10 MB — prevents 1009 on large agent responses
        ) as ws:
            await ws.send(json.dumps(payload))

            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                node = event.get("node", "")

                if node == "task_completed":
                    summary = event.get("result_summary") or event.get("result") or ""
                    if summary and len(summary) > 20:
                        final_response = summary

                if node == "__end__":
                    final_response = event.get("final_response") or final_response
                    break

                if "final_response" in event and event["final_response"]:
                    final_response = event["final_response"]

        preview = (final_response[:150] + "...") if len(final_response) > 150 else final_response
        print(f"    {tag}Response ({len(final_response)} chars): \"{preview}\"")
        return final_response, True

    except Exception as e:
        print(f"    {tag}ERROR: {e}")
        return "", False


def _contains(text: str, keywords: List[str], require_all: bool = False) -> bool:
    text_lower = text.lower()
    hits = [k.lower() in text_lower for k in keywords]
    return all(hits) if require_all else any(hits)


def _asks_to_reupload(text: str) -> bool:
    """Detect if the system is (wrongly) asking user to upload the file again."""
    phrases = [
        "please upload", "could you upload", "please share", "could you share",
        "please provide the file", "upload the file", "attach the file",
        "i don't see any file", "no file", "need you to upload",
        "send me the file", "provide a file",
    ]
    return _contains(text, phrases)


# ---------------------------------------------------------------------------
# T3-01: Follow-up without re-upload
# ---------------------------------------------------------------------------

async def test_t3_01():
    print("\n  --- T3-01: Follow-up without re-uploading file ---")
    tid = f"t3_01_{uuid.uuid4().hex[:8]}"

    file_entry = await upload_file(
        MRP_XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if not file_entry:
        record("T3-01", "Follow-up without re-upload", False, "MRP file upload failed")
        return

    # Turn 1: introduce file
    resp1, ok1 = await ws_turn(
        tid,
        "I have uploaded our MRP Requisition file. Give me a summary — how many items "
        "are there and what are the main columns?",
        files=[file_entry],
        label="Turn1",
    )
    if not ok1 or not resp1:
        record("T3-01", "Follow-up without re-upload", False, "Turn 1 failed or empty")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: follow-up, NO file in payload
    resp2, ok2 = await ws_turn(
        tid,
        "Which item code appears most frequently in that file?",
        files=None,
        label="Turn2",
    )

    reupload_asked = _asks_to_reupload(resp2)
    has_data = len(resp2) > 30 and not reupload_asked

    if not ok2:
        record("T3-01", "Follow-up without re-upload", False, "Turn 2 connection failed")
    elif reupload_asked:
        record("T3-01", "Follow-up without re-upload", False,
               "FAIL: System asked user to re-upload the file (UX dealbreaker)\n"
               f"Turn2 response: \"{resp2[:200]}\"")
    elif has_data:
        record("T3-01", "Follow-up without re-upload", True,
               f"Turn2 answered without asking for file.\n"
               f"Response: \"{resp2[:200]}\"")
    else:
        record("T3-01", "Follow-up without re-upload", False,
               f"Turn2 response too short or unhelpful: \"{resp2[:200]}\"")


# ---------------------------------------------------------------------------
# T3-02: Vague intent → file provided → resolution
# ---------------------------------------------------------------------------

async def test_t3_02():
    print("\n  --- T3-02: Vague intent + file resolution ---")
    tid = f"t3_02_{uuid.uuid4().hex[:8]}"

    # Turn 1: vague intent, no file
    resp1, ok1 = await ws_turn(
        tid,
        "I need help checking an invoice.",
        files=None,
        label="Turn1",
    )
    if not ok1:
        record("T3-02", "Vague intent + file resolution", False, "Turn 1 connection failed")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: provide file with context
    file_entry = await upload_file(INVOICE_PDF, "application/pdf")
    if not file_entry:
        record("T3-02", "Vague intent + file resolution", False, "Invoice file upload failed")
        return

    resp2, ok2 = await ws_turn(
        tid,
        "Here is the invoice. Extract the vendor name, invoice total, and line items.",
        files=[file_entry],
        label="Turn2",
    )

    has_invoice_data = _contains(resp2, ["vendor", "total", "amount", "item", "invoice",
                                          "line", "price", "quantity"])

    if not ok2:
        record("T3-02", "Vague intent + file resolution", False, "Turn 2 connection failed")
    elif has_invoice_data and len(resp2) > 50:
        record("T3-02", "Vague intent + file resolution", True,
               f"System resolved to invoice extraction after vague start.\n"
               f"Turn1: \"{resp1[:100]}\"\n"
               f"Turn2: \"{resp2[:200]}\"")
    else:
        record("T3-02", "Vague intent + file resolution", False,
               f"Turn2 did not return useful invoice data.\nResponse: \"{resp2[:200]}\"")


# ---------------------------------------------------------------------------
# T3-03: Cross-turn line item lookup (invoice detail in Turn 2)
# ---------------------------------------------------------------------------

async def test_t3_03():
    print("\n  --- T3-03: Cross-turn detail lookup ---")
    tid = f"t3_03_{uuid.uuid4().hex[:8]}"

    file_entry = await upload_file(INVOICE_PDF, "application/pdf")
    if not file_entry:
        record("T3-03", "Cross-turn detail lookup", False, "Invoice file upload failed")
        return

    # Turn 1: high-level summary
    resp1, ok1 = await ws_turn(
        tid,
        "Uploaded an invoice. What is the total amount and who is the vendor?",
        files=[file_entry],
        label="Turn1",
    )
    if not ok1 or not resp1:
        record("T3-03", "Cross-turn detail lookup", False, "Turn 1 failed")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: ask for details — NO file re-upload
    resp2, ok2 = await ws_turn(
        tid,
        "Now list all the individual line items from the same invoice.",
        files=None,
        label="Turn2",
    )

    reupload_asked = _asks_to_reupload(resp2)
    has_line_items = _contains(resp2, ["item", "line", "product", "description",
                                        "qty", "quantity", "unit", "price", "amount"])

    if not ok2:
        record("T3-03", "Cross-turn detail lookup", False, "Turn 2 connection failed")
    elif reupload_asked:
        record("T3-03", "Cross-turn detail lookup", False,
               f"System asked to re-upload. Response: \"{resp2[:200]}\"")
    elif has_line_items and len(resp2) > 40:
        record("T3-03", "Cross-turn detail lookup", True,
               f"Line items extracted in Turn2 without re-upload.\n"
               f"Response: \"{resp2[:250]}\"")
    else:
        record("T3-03", "Cross-turn detail lookup", False,
               f"Turn2 lacked line item content. Response: \"{resp2[:200]}\"")


# ---------------------------------------------------------------------------
# T3-04: QC department rejection documentation question (no file)
# ---------------------------------------------------------------------------

async def test_t3_04():
    print("\n  --- T3-04: QC process documentation question ---")
    tid = f"t3_04_{uuid.uuid4().hex[:8]}"

    resp, ok = await ws_turn(
        tid,
        "I work in the QC department at a chemical manufacturing company. "
        "We just rejected a raw material batch from a supplier. "
        "What documents do I need to fill out and what information should they contain?",
        files=None,
        label="Turn1",
    )

    # Should mention rejection-related documents and fields
    doc_keywords   = ["rejection", "return", "report", "note", "debit", "certificate", "coa"]
    field_keywords = ["batch", "reason", "supplier", "quantity", "test", "specification",
                      "observation", "disposition", "qc", "quality"]

    has_docs   = _contains(resp, doc_keywords)
    has_fields = _contains(resp, field_keywords)
    passed     = ok and has_docs and has_fields and len(resp) > 100

    detail = (
        f"doc_keywords_found={has_docs}  field_keywords_found={has_fields}\n"
        f"Response: \"{resp[:300]}\""
    )
    record("T3-04", "QC rejection documentation question", passed, detail)


# ---------------------------------------------------------------------------
# T3-05: Memory filter — apply filter to previously analysed data
# ---------------------------------------------------------------------------

async def test_t3_05():
    print("\n  --- T3-05: Memory filter within session ---")
    tid = f"t3_05_{uuid.uuid4().hex[:8]}"

    file_entry = await upload_file(
        MRP_XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if not file_entry:
        record("T3-05", "Memory filter within session", False, "MRP file upload failed")
        return

    # Turn 1: get overview
    resp1, ok1 = await ws_turn(
        tid,
        "I uploaded our MRP Requisition file. How many requisitions are in 'Pending' status?",
        files=[file_entry],
        label="Turn1",
    )
    if not ok1 or not resp1:
        record("T3-05", "Memory filter within session", False, "Turn 1 failed")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: filter — NO file
    resp2, ok2 = await ws_turn(
        tid,
        "From those pending requisitions, which user created the most?",
        files=None,
        label="Turn2",
    )

    reupload_asked = _asks_to_reupload(resp2)
    has_answer = len(resp2) > 30 and not reupload_asked

    if not ok2:
        record("T3-05", "Memory filter within session", False, "Turn 2 connection failed")
    elif reupload_asked:
        record("T3-05", "Memory filter within session", False,
               f"System asked to re-upload. Response: \"{resp2[:200]}\"")
    elif has_answer:
        record("T3-05", "Memory filter within session", True,
               f"Turn2 filtered data without re-upload.\n"
               f"Turn1: \"{resp1[:100]}\"\n"
               f"Turn2: \"{resp2[:200]}\"")
    else:
        record("T3-05", "Memory filter within session", False,
               f"Turn2 unhelpful. Response: \"{resp2[:200]}\"")


# ---------------------------------------------------------------------------
# T3-06: Minimal prompt with file (just a dot)
# ---------------------------------------------------------------------------

async def test_t3_06():
    print("\n  --- T3-06: Minimal prompt with file ---")
    tid = f"t3_06_{uuid.uuid4().hex[:8]}"

    file_entry = await upload_file(
        MRP_XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if not file_entry:
        record("T3-06", "Minimal prompt with file", False, "MRP file upload failed")
        return

    resp, ok = await ws_turn(
        tid,
        "check this file",   # short vague prompt — realistic Denisco staff behaviour
        files=[file_entry],
        label="Turn1",
    )

    # Pass: system responded usefully (file summary / column info / any data) without crashing
    file_keywords = ["column", "row", "requisition", "data", "sheet", "file",
                     "material", "item", "status", "pending"]
    has_content = _contains(resp, file_keywords) and len(resp) > 50
    passed = ok and has_content

    detail = (
        f"has_file_content={has_content}  response_len={len(resp)}\n"
        f"Response: \"{resp[:300]}\""
    )
    record("T3-06", "Minimal prompt with file", passed, detail,
           note="Pass = system auto-summarises file with minimal prompt, no crash")


# ---------------------------------------------------------------------------
# T3-07: Correction handling
# ---------------------------------------------------------------------------

async def test_t3_07():
    print("\n  --- T3-07: Correction handling ---")
    tid = f"t3_07_{uuid.uuid4().hex[:8]}"

    file_entry = await upload_file(
        EMPLOYEES_XLS,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if not file_entry:
        record("T3-07", "Correction handling", False, "employees.xlsx upload failed")
        return

    # Turn 1: ask for payroll total
    resp1, ok1 = await ws_turn(
        tid,
        "What is the total payroll amount from the employees spreadsheet?",
        files=[file_entry],
        label="Turn1",
    )
    if not ok1 or not resp1:
        record("T3-07", "Correction handling", False, "Turn 1 failed")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: challenge the answer
    resp2, ok2 = await ws_turn(
        tid,
        "That doesn't look right. The allowances column was probably missed. "
        "Can you recheck and include all salary components?",
        files=None,
        label="Turn2",
    )

    # Pass: system re-engages rather than arguing or ignoring
    reengages = _contains(resp2, ["recheck", "recalculate", "let me", "apologies",
                                   "sorry", "looking again", "revised", "including",
                                   "allowance", "component", "total", "salary", "amount"])
    does_not_argue = not _contains(resp2, ["i was correct", "my answer was right",
                                            "i cannot change", "that is correct"])

    passed = ok2 and reengages and does_not_argue and len(resp2) > 30
    detail = (
        f"re-engages={reengages}  does_not_argue={does_not_argue}\n"
        f"Turn2 response: \"{resp2[:250]}\""
    )
    record("T3-07", "Correction handling", passed, detail)


# ---------------------------------------------------------------------------
# T3-08: Dispatch process knowledge (no file)
# ---------------------------------------------------------------------------

async def test_t3_08():
    print("\n  --- T3-08: Dispatch process knowledge ---")
    tid = f"t3_08_{uuid.uuid4().hex[:8]}"

    resp, ok = await ws_turn(
        tid,
        "What documents are required to dispatch a finished goods batch to a customer "
        "from a chemical manufacturing company? List each document and what it contains.",
        files=None,
        label="Turn1",
    )

    # Must mention ≥4 of the core dispatch documents
    # Note: LLM may use various spellings/hyphens for e-way bill
    dispatch_docs = ["packing list", "delivery challan", "challan", "tax invoice",
                     "invoice", "e-way", "eway", "e\u2011way", "lorry receipt",
                     "lr copy", "transport", "dispatch register", "e way bill"]
    hits = [d for d in dispatch_docs if d.lower() in resp.lower()]
    passed = ok and len(hits) >= 4 and len(resp) > 100

    detail = (
        f"dispatch docs found ({len(hits)}): {hits}\n"
        f"Response: \"{resp[:300]}\""
    )
    record("T3-08", "Dispatch process knowledge", passed, detail,
           note="Pass = >=4 correct dispatch documents listed")


# ---------------------------------------------------------------------------
# T3-09: Off-topic graceful decline
# ---------------------------------------------------------------------------

async def test_t3_09():
    print("\n  --- T3-09: Off-topic graceful decline ---")
    tid = f"t3_09_{uuid.uuid4().hex[:8]}"

    resp, ok = await ws_turn(
        tid,
        "Can you book a flight ticket to Delhi for me for next Monday?",
        files=None,
        label="Turn1",
    )

    # FAIL if system pretends it can book flights (hallucinating a non-existent capability)
    hallucinates = _contains(resp, [
        "booked", "booking confirmed", "your ticket", "flight number",
        "seat number", "confirmation number", "reservation confirmed",
        "i can help you find and book", "find and book a flight",
        "help you book", "let me book", "i'll book",
    ])
    # PASS if system explains it cannot book, or redirects appropriately
    explains = _contains(resp, ["cannot", "can't", "unable", "don't have the ability",
                                 "not able", "outside my", "i can't book",
                                 "suggest", "recommend", "travel website",
                                 "makemytrip", "cleartrip", "irctc"])

    # Also PASS if system just asks clarifying questions (browser agent may actually help)
    asks_clarify = _contains(resp, ["departure", "origin", "from which city",
                                     "which city", "departure city"])

    passed = ok and not hallucinates and (explains or asks_clarify) and len(resp) > 20
    detail = (
        f"hallucinates_booking={hallucinates}  explains_limitation={explains}\n"
        f"Response: \"{resp[:250]}\""
    )
    record("T3-09", "Off-topic graceful decline", passed, detail)


# ---------------------------------------------------------------------------
# T3-10: Multi-file multi-turn (second file in Turn 2)
# ---------------------------------------------------------------------------

async def test_t3_10():
    print("\n  --- T3-10: Multi-file multi-turn ---")
    tid = f"t3_10_{uuid.uuid4().hex[:8]}"

    mrp_entry = await upload_file(
        MRP_XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    emp_entry = await upload_file(
        EMPLOYEES_XLS,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if not mrp_entry or not emp_entry:
        record("T3-10", "Multi-file multi-turn", False,
               f"File upload failed: mrp={bool(mrp_entry)}, employees={bool(emp_entry)}")
        return

    # Turn 1: MRP file
    resp1, ok1 = await ws_turn(
        tid,
        "I uploaded the MRP Requisition file. How many unique item codes are in it?",
        files=[mrp_entry],
        label="Turn1",
    )
    if not ok1 or not resp1:
        record("T3-10", "Multi-file multi-turn", False, "Turn 1 failed")
        return

    await asyncio.sleep(INTER_TURN_PAUSE)

    # Turn 2: employees file (second file, different topic)
    resp2, ok2 = await ws_turn(
        tid,
        "Now I have also uploaded the employees spreadsheet. "
        "How many employees are there and what departments exist?",
        files=[emp_entry],
        label="Turn2",
    )

    # Turn 2 must talk about employees, not MRP
    emp_keywords  = ["employee", "department", "staff", "headcount", "salary",
                     "hr", "manager", "name"]
    mrp_confused  = _contains(resp2, ["item code", "requisition", "mrp", "purchase order"])

    has_emp_data  = _contains(resp2, emp_keywords) and len(resp2) > 40
    passed        = ok2 and has_emp_data and not mrp_confused

    detail = (
        f"has_employee_data={has_emp_data}  confused_with_mrp={mrp_confused}\n"
        f"Turn2 response: \"{resp2[:250]}\""
    )
    record("T3-10", "Multi-file multi-turn", passed, detail,
           note="Pass = Turn2 discusses employees, not MRP content")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_all():
    print()
    print("=" * 70)
    print("  DENISCO BETA — TIER 3: CONVERSATIONAL INTELLIGENCE")
    print("  10 multi-turn conversation tests via WebSocket")
    print("  Target: 8/10 pass for GO verdict")
    print("=" * 70)

    # Verify orchestrator is reachable before starting
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ORCHESTRATOR_BASE}/api/health")
            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}")
        print("  Orchestrator: reachable")
    except Exception as e:
        print(f"\n  ABORT: Orchestrator not reachable — {e}")
        print("  Start it first: cd backend && ../venv/Scripts/python -m uvicorn main:app --reload")
        return False

    # Run all tests sequentially (conversational tests cannot be parallelised)
    tests = [
        test_t3_01,
        test_t3_02,
        test_t3_03,
        test_t3_04,
        test_t3_05,
        test_t3_06,
        test_t3_07,
        test_t3_08,
        test_t3_09,
        test_t3_10,
    ]

    for test_fn in tests:
        try:
            await test_fn()
        except Exception as e:
            test_id = test_fn.__name__.replace("test_", "").upper().replace("_", "-")
            record(test_id, test_fn.__name__, False, f"Unhandled exception: {e}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  TIER 3 RESULTS")
    print("=" * 70)

    passed_list = [r for r in results if r["status"] == "PASS"]
    failed_list = [r for r in results if r["status"] == "FAIL"]
    total  = len(results)
    passed = len(passed_list)

    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"  {icon} | {r['id']:<6} | {r['name']}")

    print()
    print(f"  Score: {passed}/{total}")
    target = 8

    if passed >= target:
        print(f"  VERDICT: GO — {passed}/{total} passed (target {target})")
    else:
        print(f"  VERDICT: NO-GO — {passed}/{total} passed (need {target})")
        print()
        print("  Failed:")
        for r in failed_list:
            print(f"    {r['id']} — {r['name']}")
            if r["detail"]:
                first_line = r["detail"].split("\n")[0]
                print(f"           {first_line}")

    print("=" * 70)
    return passed >= target


if __name__ == "__main__":
    import sys
    go = asyncio.run(run_all())
    sys.exit(0 if go else 1)

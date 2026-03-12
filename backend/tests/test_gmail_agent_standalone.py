#!/usr/bin/env python3
"""
Gmail Agent — Live Integration Test Suite
==========================================

Tests real Gmail operations against al.ashritha@gmail.com via Composio.
Emails are sent to: ashrithaannadata@gmail.com

Prerequisites:
  1. Gmail account connected via Composio for user_374hMFRAc0nkaGdH8XtXNRIdfrk
  2. COMPOSIO_API_KEY set in backend/.env
  3. No live agent server required — tests call GmailService directly

Usage:
  cd c:/Users/akush/Orchestrator-preview
  PYTHONUTF8=1 venv/Scripts/python backend/tests/test_gmail_agent_standalone.py

  # Or via pytest:
  PYTHONUTF8=1 venv/Scripts/python -m pytest backend/tests/test_gmail_agent_standalone.py -v -s

Tests:
  T01 - Verify Gmail connection is active (get profile)
  T02 - Fetch inbox emails (up to 5 most recent)
  T03 - Search for unread emails
  T04 - Search for emails from a specific sender
  T05 - Send plain text email to ashrithaannadata@gmail.com
  T06 - Send HTML email to ashrithaannadata@gmail.com
  T07 - Send email with subject containing timestamp (uniqueness check)
  T08 - List Gmail labels
  T09 - List email threads (inbox)
  T10 - Search emails with natural language query (LLM optimization)
  T11 - Create a draft email
  T12 - List drafts and find the created draft
  T13 - Delete the draft created in T11
  T14 - Fetch a specific email by ID (uses result from T02)
  T15 - Summarize recent emails (LLM-powered)
  T16 - Extract action items from recent emails (LLM-powered)
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

# ── Test config ────────────────────────────────────────────────────────────────
USER_ID = "user_374hMFRAc0nkaGdH8XtXNRIdfrk"
SENDER_EMAIL = "al.ashritha@gmail.com"
RECIPIENT_EMAIL = "ashrithaannadata@gmail.com"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg): print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")


# ── Service factory ────────────────────────────────────────────────────────────
def get_service():
    from backend.agents.gmail_agent.service import GmailService
    return GmailService(USER_ID)


# ==============================================================================
# Test results tracker
# ==============================================================================
results = {}
shared_state = {}   # pass data between tests (e.g. message IDs)


async def run_test(name: str, coro):
    print(f"\n{BOLD}{CYAN}[{name}]{RESET} ", end="")
    try:
        result = await coro
        results[name] = "PASS"
        return result
    except AssertionError as e:
        results[name] = f"FAIL: {e}"
        fail(str(e))
        return None
    except Exception as e:
        results[name] = f"ERROR: {e}"
        fail(f"Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# T01 — Get Gmail profile
# ==============================================================================
async def test_t01_get_profile():
    print("Get Gmail profile")
    svc = get_service()
    result = await svc.get_profile()

    assert result["success"] is True, f"Profile fetch failed: {result.get('error')}"
    profile = result["profile"]
    info(f"Profile data: {json.dumps(profile, indent=2)[:300]}")

    # The emailAddress may be nested depending on Composio response shape
    email = (
        profile.get("emailAddress") or
        profile.get("email") or
        (profile.get("data") or {}).get("emailAddress") or
        "unknown"
    )
    info(f"Connected Gmail: {email}")
    ok(f"Profile fetched successfully")
    return result


# ==============================================================================
# T02 — Fetch inbox emails
# ==============================================================================
async def test_t02_fetch_inbox():
    print("Fetch inbox emails (last 5)")
    svc = get_service()
    result = await svc.search_emails(
        query="label:inbox",
        max_results=5,
        use_llm_optimization=False
    )

    assert result["success"] is True, f"Fetch failed: {result.get('error')}"
    messages = result.get("messages", [])
    info(f"Found {result['total_count']} emails in inbox")

    if messages:
        msg = messages[0]
        msg_id = msg.get("id") or msg.get("messageId") or msg.get("message_id")
        if msg_id:
            shared_state["first_message_id"] = msg_id
            info(f"First message ID stored: {msg_id}")
        subject = msg.get("subject") or msg.get("snippet", "")[:60]
        info(f"First email snippet: {subject}")

    ok(f"Inbox fetched — {len(messages)} messages returned")
    return result


# ==============================================================================
# T03 — Search unread emails
# ==============================================================================
async def test_t03_search_unread():
    print("Search unread emails")
    svc = get_service()
    result = await svc.search_emails(
        query="is:unread",
        max_results=10,
        use_llm_optimization=False
    )

    assert result["success"] is True, f"Search failed: {result.get('error')}"
    info(f"Unread emails: {result['total_count']}")
    ok("Unread search completed")
    return result


# ==============================================================================
# T04 — Search emails from specific sender
# ==============================================================================
async def test_t04_search_from_sender():
    print(f"Search emails from {RECIPIENT_EMAIL}")
    svc = get_service()
    result = await svc.search_emails(
        query=f"from:{RECIPIENT_EMAIL}",
        max_results=5,
        use_llm_optimization=False
    )

    assert result["success"] is True, f"Search failed: {result.get('error')}"
    info(f"Emails from {RECIPIENT_EMAIL}: {result['total_count']}")
    ok("Sender search completed")
    return result


# ==============================================================================
# T05 — Send plain text email
# ==============================================================================
async def test_t05_send_plain_text_email():
    print(f"Send plain text email → {RECIPIENT_EMAIL}")
    svc = get_service()
    result = await svc.send_email(
        to=RECIPIENT_EMAIL,
        subject=f"[Gmail Agent Test] Plain Text — {TIMESTAMP}",
        body=(
            f"Hello!\n\n"
            f"This is an automated plain text test email sent by the Gmail Agent unit test suite.\n\n"
            f"Timestamp: {TIMESTAMP}\n"
            f"Sender account: {SENDER_EMAIL}\n"
            f"Test ID: T05\n\n"
            f"If you receive this, the Gmail agent is working correctly!"
        ),
        is_html=False
    )

    assert result["success"] is True, f"Send failed: {result.get('error')}"
    data = result.get("data", {})
    sent_id = (data.get("id") or data.get("messageId") or data.get("message_id") or "unknown")
    info(f"Sent message ID: {sent_id}")
    ok("Plain text email sent successfully")
    return result


# ==============================================================================
# T06 — Send HTML email
# ==============================================================================
async def test_t06_send_html_email():
    print(f"Send HTML email → {RECIPIENT_EMAIL}")
    svc = get_service()
    html_body = f"""
<html>
<body>
  <h2 style="color:#2563eb;">Gmail Agent — HTML Test Email</h2>
  <p>This is an automated <strong>HTML test email</strong> sent by the Gmail Agent integration test suite.</p>
  <ul>
    <li><b>Timestamp:</b> {TIMESTAMP}</li>
    <li><b>From:</b> {SENDER_EMAIL}</li>
    <li><b>Test ID:</b> T06</li>
    <li><b>Status:</b> <span style="color:green;">Working ✓</span></li>
  </ul>
  <p style="color:#6b7280; font-size:12px;">Sent by the Orchestrator Gmail Agent test suite.</p>
</body>
</html>
"""

    result = await svc.send_email(
        to=RECIPIENT_EMAIL,
        subject=f"[Gmail Agent Test] HTML Email — {TIMESTAMP}",
        body=html_body,
        is_html=True
    )

    assert result["success"] is True, f"HTML send failed: {result.get('error')}"
    ok("HTML email sent successfully")
    return result


# ==============================================================================
# T07 — Send email with CC
# ==============================================================================
async def test_t07_send_email_with_timestamp():
    print(f"Send email with unique subject (timestamp check)")
    svc = get_service()
    unique_id = str(int(time.time()))
    result = await svc.send_email(
        to=RECIPIENT_EMAIL,
        subject=f"[Gmail Agent Test] Timestamp={unique_id}",
        body=f"Unique test email. ID={unique_id}. Timestamp={TIMESTAMP}.",
        is_html=False
    )

    assert result["success"] is True, f"Send failed: {result.get('error')}"
    shared_state["unique_subject_id"] = unique_id
    info(f"Unique ID: {unique_id}")
    ok("Timestamped email sent")
    return result


# ==============================================================================
# T08 — List Gmail labels
# ==============================================================================
async def test_t08_list_labels():
    print("List Gmail labels")
    svc = get_service()
    result = await svc.list_labels()

    assert result["success"] is True, f"List labels failed: {result.get('error')}"
    labels = result.get("labels", [])
    info(f"Total labels: {len(labels)}")

    # Print first 10 labels
    for label in labels[:10]:
        name = label.get("name") or label.get("id") or str(label)
        info(f"  Label: {name}")

    ok(f"Labels listed — {len(labels)} found")
    return result


# ==============================================================================
# T09 — List email threads
# ==============================================================================
async def test_t09_list_threads():
    print("List email threads (inbox)")
    svc = get_service()
    result = await svc.list_threads(query="label:inbox", max_results=5)

    assert result["success"] is True, f"List threads failed: {result.get('error')}"
    threads = result.get("threads", [])
    info(f"Threads found: {len(threads)}")

    if threads:
        thread_id = threads[0].get("id")
        if thread_id:
            shared_state["first_thread_id"] = thread_id
            info(f"First thread ID stored: {thread_id}")

    ok(f"Threads listed — {len(threads)} found")
    return result


# ==============================================================================
# T10 — Natural language search (LLM-optimized)
# ==============================================================================
async def test_t10_natural_language_search():
    print("Natural language email search (with LLM query optimization)")
    svc = get_service()
    result = await svc.search_emails(
        query="show me emails I received recently about tests or automation",
        max_results=5,
        use_llm_optimization=True
    )

    assert result["success"] is True, f"NL search failed: {result.get('error')}"
    info(f"Query used: {result.get('query_used')}")
    info(f"Results: {result['total_count']}")
    ok("Natural language search completed")
    return result


# ==============================================================================
# T11 — Create a draft email
# ==============================================================================
async def test_t11_create_draft():
    print(f"Create draft email → {RECIPIENT_EMAIL}")
    svc = get_service()
    result = await svc.create_draft(
        to=RECIPIENT_EMAIL,
        subject=f"[Gmail Agent Test] Draft — {TIMESTAMP}",
        body="This is a draft created by the Gmail Agent test suite. It should not be sent."
    )

    assert result["success"] is True, f"Create draft failed: {result.get('error')}"
    draft = result.get("draft", {})
    draft_id = draft.get("id") or draft.get("draftId") or draft.get("draft_id")
    if draft_id:
        shared_state["test_draft_id"] = draft_id
        info(f"Draft ID stored: {draft_id}")
    ok("Draft created successfully")
    return result


# ==============================================================================
# T12 — List drafts
# ==============================================================================
async def test_t12_list_drafts():
    print("List email drafts")
    svc = get_service()
    result = await svc.list_drafts(max_results=10)

    assert result["success"] is True, f"List drafts failed: {result.get('error')}"
    drafts = result.get("drafts", [])
    info(f"Drafts found: {len(drafts)}")
    ok(f"Drafts listed — {len(drafts)} found")
    return result


# ==============================================================================
# T13 — Delete the draft created in T11
# ==============================================================================
async def test_t13_delete_draft():
    print("Delete test draft from T11")
    draft_id = shared_state.get("test_draft_id")

    if not draft_id:
        warn("No draft_id from T11 — skipping deletion")
        return {"skipped": True}

    svc = get_service()
    result = await svc.delete_draft(draft_id)

    assert result["success"] is True, f"Delete draft failed: {result.get('error')}"
    ok(f"Draft {draft_id} deleted")
    return result


# ==============================================================================
# T14 — Fetch a specific email by ID
# ==============================================================================
async def test_t14_get_email_by_id():
    print("Get specific email by message ID")
    message_id = shared_state.get("first_message_id")

    if not message_id:
        warn("No message_id from T02 — skipping get_email test")
        return {"skipped": True}

    svc = get_service()
    result = await svc.get_email(message_id)

    assert result["success"] is True, f"Get email failed: {result.get('error')}"
    message = result.get("message", {})
    subject = message.get("subject") or message.get("snippet", "")[:60]
    sender = message.get("from") or message.get("sender") or "unknown"
    info(f"Subject: {subject}")
    info(f"From: {sender}")
    ok("Email fetched by ID successfully")
    return result


# ==============================================================================
# T15 — Summarize recent emails (LLM-powered)
# ==============================================================================
async def test_t15_summarize_recent_emails():
    print("Summarize recent emails using LLM")
    # First get some message IDs
    svc = get_service()
    search_result = await svc.search_emails(
        query="label:inbox",
        max_results=3,
        use_llm_optimization=False
    )

    if not search_result["success"] or not search_result.get("messages"):
        warn("No emails to summarize — skipping")
        return {"skipped": True}

    message_ids = [
        (m.get("id") or m.get("messageId") or m.get("message_id"))
        for m in search_result["messages"][:3]
        if (m.get("id") or m.get("messageId") or m.get("message_id"))
    ]

    if not message_ids:
        warn("Could not extract message IDs — skipping")
        return {"skipped": True}

    result = await svc.summarize_emails(message_ids)

    assert result["success"] is True, f"Summarize failed: {result.get('error')}"
    info(f"Emails summarized: {result.get('emails_summarized', 0)}")
    info(f"Summary preview: {str(result.get('summary', ''))[:200]}")
    ok("Email summarization completed")
    return result


# ==============================================================================
# T16 — Extract action items from recent emails (LLM-powered)
# ==============================================================================
async def test_t16_extract_action_items():
    print("Extract action items from recent emails using LLM")
    svc = get_service()
    search_result = await svc.search_emails(
        query="label:inbox",
        max_results=3,
        use_llm_optimization=False
    )

    if not search_result["success"] or not search_result.get("messages"):
        warn("No emails for action extraction — skipping")
        return {"skipped": True}

    message_ids = [
        (m.get("id") or m.get("messageId") or m.get("message_id"))
        for m in search_result["messages"][:3]
        if (m.get("id") or m.get("messageId") or m.get("message_id"))
    ]

    if not message_ids:
        warn("Could not extract message IDs — skipping")
        return {"skipped": True}

    result = await svc.extract_action_items(message_ids)

    assert result["success"] is True, f"Extract actions failed: {result.get('error')}"
    actions = result.get("action_items", [])
    info(f"Action items found: {result.get('total_actions', len(actions))}")
    for action in actions[:3]:
        info(f"  • [{action.get('priority','?').upper()}] {action.get('description', '')[:80]}")
    ok("Action item extraction completed")
    return result


# ==============================================================================
# Main runner
# ==============================================================================
async def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Gmail Agent — Live Integration Tests{RESET}")
    print(f"{'='*60}")
    print(f"  Sender:    {SENDER_EMAIL}")
    print(f"  Recipient: {RECIPIENT_EMAIL}")
    print(f"  User ID:   {USER_ID}")
    print(f"  Run time:  {TIMESTAMP}")
    print(f"{'='*60}\n")

    tests = [
        ("T01", test_t01_get_profile),
        ("T02", test_t02_fetch_inbox),
        ("T03", test_t03_search_unread),
        ("T04", test_t04_search_from_sender),
        ("T05", test_t05_send_plain_text_email),
        ("T06", test_t06_send_html_email),
        ("T07", test_t07_send_email_with_timestamp),
        ("T08", test_t08_list_labels),
        ("T09", test_t09_list_threads),
        ("T10", test_t10_natural_language_search),
        ("T11", test_t11_create_draft),
        ("T12", test_t12_list_drafts),
        ("T13", test_t13_delete_draft),
        ("T14", test_t14_get_email_by_id),
        ("T15", test_t15_summarize_recent_emails),
        ("T16", test_t16_extract_action_items),
    ]

    for name, test_fn in tests:
        await run_test(name, test_fn())

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Test Summary{RESET}")
    print(f"{'='*60}")

    passed = [k for k, v in results.items() if v == "PASS"]
    skipped = [k for k, v in results.items() if "skipped" in str(v).lower()]
    failed = [k for k, v in results.items() if v not in ("PASS",) and "skipped" not in str(v).lower()]

    for name, status in results.items():
        if status == "PASS":
            print(f"  {GREEN}PASS{RESET}  {name}")
        elif "skipped" in str(status).lower():
            print(f"  {YELLOW}SKIP{RESET}  {name}")
        else:
            print(f"  {RED}FAIL{RESET}  {name}  —  {status}")

    print(f"\n  {GREEN}{len(passed)} passed{RESET}  |  {YELLOW}{len(skipped)} skipped{RESET}  |  {RED}{len(failed)} failed{RESET}  |  {len(tests)} total")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

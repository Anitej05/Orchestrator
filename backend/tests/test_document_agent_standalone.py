#!/usr/bin/env python3
"""
Document Agent — HTTP Test Suite
=================================

Prerequisites:
  1. Start the Document Agent server:
       cd backend
       python -m backend.agents.document_agent_lib          # runs on port 8050

  2. (Optional) Start the main orchestrator backend:
       cd backend
       uvicorn main:app --port 8000 --reload                # for /api/chat tests

  3. Place your Manus AI document (PDF, DOCX, or TXT) in:
       backend/tests/test_data/
     The script will auto-detect the first matching file, or you can
     pass the path explicitly:
       python tests/test_document_agent_standalone.py --doc "tests/test_data/manus_ai.pdf"

Usage:
  cd backend
  python tests/test_document_agent_standalone.py                   # all tests
  python tests/test_document_agent_standalone.py --doc my_doc.pdf  # specify document
  python tests/test_document_agent_standalone.py --agent-url http://localhost:8050
  python tests/test_document_agent_standalone.py --backend-url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
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
        for i, line in enumerate(msg[:300].split("\n")):
            print(f"{'         ' if i == 0 else '           '}{line}")
    _results.append({"group": group, "test": name, "pass": ok, "msg": msg, "time": dur})


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS — your Manus AI questions
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS = [
    # ── Factual retrieval (easy) ──────────────────────────────────────────
    "What is Manus AI and when was it introduced?",
    "Which company developed Manus AI?",
    "What are the three main agents in Manus AI's multi-agent architecture?",

    # ── Conceptual understanding (medium) ─────────────────────────────────
    "How does Manus AI differ from traditional chatbots like GPT-4?",
    "What is the GAIA benchmark and how did Manus AI perform on it?",

    # ── Applied / analytical (hard) ───────────────────────────────────────
    "Describe how Manus AI could be used in the healthcare industry. Give specific examples.",

    "Compare Manus AI's capabilities with OpenAI's Operator, Anthropic's Computer Use, "
    "and Google's Mariner based on the feature comparison table. What are the key differences "
    "in their availability and API integration?",

    "What are the main limitations and ethical concerns of Manus AI according to the paper, "
    "and how do they relate to its autonomous capabilities?",

    "Analyze how Manus AI's tool integration capability and continuous learning features "
    "work together to enable autonomous task execution. Use examples from at least two "
    "different industries.",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DOCUMENT AGENT DIRECT TESTS  (requires agent on --agent-url)
# ═══════════════════════════════════════════════════════════════════════════

async def test_agent_health(agent_url: str):
    """Check the document agent is reachable."""
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{agent_url}/health")
            ok = r.status_code == 200 and r.json().get("status") in ("healthy", "not_initialized", "ready")
            log("agent", "Health check", ok, r.text[:200], time.time() - t)
            return ok
    except Exception as e:
        log("agent", "Health check", False, f"Cannot reach agent at {agent_url} — {e}", time.time() - t)
        return False


async def test_verify_file(doc_path: str) -> Optional[str]:
    """Verify the local file exists and return its absolute path."""
    t = time.time()
    p = Path(doc_path).resolve()
    ok = p.exists()
    log("agent", f"File found: {p.name}", ok,
        str(p) if ok else f"File not found: {doc_path}", time.time() - t)
    return str(p) if ok else None


async def test_agent_analyze(agent_url: str, file_path: str, prompt: str, idx: int):
    """Send one analysis prompt via /execute with action=analyze_document."""
    import httpx
    t = time.time()
    label = f"Q{idx+1}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{agent_url}/execute", json={
                "prompt": prompt,
                "action": "analyze_document",
                "payload": {
                    "file_path": file_path,
                    "query": prompt,
                    "thread_id": "test-session",
                },
                "thread_id": "test-session",
            })
            data = r.json()
            ok = r.status_code == 200 and data.get("status") == "success"
            # Extract answer from result dict or summary
            result = data.get("result") or {}
            answer = (result.get("answer") if isinstance(result, dict) else "") or data.get("summary", "")
            if not ok:
                answer = data.get("error_message") or data.get("summary") or json.dumps(data)[:200]
            log("analyze", label, ok, str(answer)[:300], time.time() - t)
    except Exception as e:
        log("analyze", label, False, str(e), time.time() - t)


async def test_agent_display(agent_url: str, file_path: str):
    """Test display_document capability via /execute."""
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{agent_url}/execute", json={
                "prompt": "Display the document",
                "action": "display_document",
                "payload": {"file_path": file_path},
                "thread_id": "test-session",
            })
            data = r.json()
            ok = r.status_code == 200 and data.get("status") == "success"
            msg = data.get("summary", data.get("error_message", ""))[:200]
            log("agent", "Display document", ok, msg, time.time() - t)
    except Exception as e:
        log("agent", "Display document", False, str(e), time.time() - t)


async def test_agent_extract(agent_url: str, file_path: str):
    """Test extract_data capability via /execute."""
    import httpx
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{agent_url}/execute", json={
                "prompt": "Extract text from document",
                "action": "extract_data",
                "payload": {
                    "file_path": file_path,
                    "extraction_type": "text",
                },
                "thread_id": "test-session",
            })
            data = r.json()
            ok = r.status_code == 200 and data.get("status") == "success"
            msg = data.get("summary", data.get("error_message", ""))[:200]
            log("agent", "Extract text", ok, msg, time.time() - t)
    except Exception as e:
        log("agent", "Extract text", False, str(e), time.time() - t)


# ═══════════════════════════════════════════════════════════════════════════
# 2. ORCHESTRATOR /api/chat TEST  (requires main backend on --backend-url)
# ═══════════════════════════════════════════════════════════════════════════

async def test_backend_chat(backend_url: str, prompt: str, files: Optional[List] = None):
    """Send a prompt through the orchestrator /api/chat endpoint."""
    import httpx
    t = time.time()
    label = f"Chat: {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
    try:
        body: Dict[str, Any] = {"prompt": prompt}
        if files:
            body["files"] = files
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{backend_url}/api/chat", json=body)
            data = r.json()
            ok = r.status_code == 200
            summary = data.get("final_response", data.get("message", ""))
            log("chat", label, ok, str(summary)[:300], time.time() - t)
    except Exception as e:
        log("chat", label, False, str(e), time.time() - t)


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

DEFAULT_DOC = "manus doc.pdf"

def find_document(explicit: Optional[str]) -> Optional[str]:
    """Return the absolute path to the test document."""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        if p.exists():
            return str(p)
        print(f"{C.R}  Document not found: {p}{C.E}")
        return None

    # Default: look for the Manus AI doc in test_data/
    default_path = TEST_DATA_DIR / DEFAULT_DOC
    if default_path.exists():
        return str(default_path)

    # Fallback: auto-detect any PDF/DOCX/TXT in test_data/
    for ext in ("*.pdf", "*.docx", "*.txt", "*.doc"):
        for f in TEST_DATA_DIR.glob(ext):
            if f.name in ("generate_test_data.py", "README.md"):
                continue
            return str(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Document Agent HTTP test suite")
    parser.add_argument("--doc", type=str, default=None,
                        help="Path to the document to upload (auto-detects from tests/test_data/ if omitted)")
    parser.add_argument("--agent-url", type=str, default="http://localhost:8050",
                        help="Document agent base URL (default: http://localhost:8050)")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000",
                        help="Main orchestrator backend URL (default: http://localhost:8000)")
    parser.add_argument("--skip-execute", action="store_true",
                        help="Skip /execute endpoint tests")
    parser.add_argument("--skip-chat", action="store_true",
                        help="Skip /api/chat orchestrator tests")
    parser.add_argument("--delay", type=int, default=10,
                        help="Seconds to wait between prompts to avoid rate limits (default: 10)")
    args = parser.parse_args()

    header("DOCUMENT AGENT — HTTP TEST SUITE")

    # ── Locate document ───────────────────────────────────────────────────
    doc_path = find_document(args.doc)
    if not doc_path:
        print(f"\n{C.R}  No document found!{C.E}")
        print(f"  Place your Manus AI PDF/DOCX/TXT in:  {TEST_DATA_DIR}")
        print(f"  Or pass explicitly: --doc path/to/file.pdf\n")
        sys.exit(1)

    print(f"  Document : {doc_path}")
    print(f"  Agent URL: {args.agent_url}")
    print(f"  Backend  : {args.backend_url}")
    print(f"  Delay    : {args.delay}s between prompts")

    overall_start = time.time()

    # ── 1. Health check ───────────────────────────────────────────────────
    header("HEALTH CHECK")
    alive = await test_agent_health(args.agent_url)
    if not alive:
        print(f"\n{C.R}  Agent is not running! Start it first:{C.E}")
        print(f"    cd backend")
        print(f"    python -m backend.agents.document_agent_lib\n")
        sys.exit(1)

    # ── 2. Verify document exists (no upload — agent reads from local path) ──
    header("VERIFY DOCUMENT")
    server_path = await test_verify_file(doc_path)
    if not server_path:
        print(f"\n{C.R}  Document not found — cannot continue.{C.E}\n")
        sys.exit(1)

    # ── 3. Display & Extract (smoke tests via /execute) ────────────────────
    header("DISPLAY & EXTRACT")
    await test_agent_display(args.agent_url, server_path)
    await test_agent_extract(args.agent_url, server_path)

    # ── 4. Analyze — all 9 prompts via /execute (action=analyze_document) ──
    header("ANALYZE — 9 PROMPTS VIA /execute (action=analyze_document)")
    for i, prompt in enumerate(PROMPTS):
        if i > 0 and args.delay > 0:
            print(f"  {C.Y}[wait] Waiting {args.delay}s to avoid rate limits...{C.E}")
            await asyncio.sleep(args.delay)
        await test_agent_analyze(args.agent_url, server_path, prompt, i)

    # ── 6. Orchestrator /api/chat (optional) ─────────────────────────────
    if not args.skip_chat:
        header("ORCHESTRATOR /api/chat (first 3 prompts)")
        print(f"  {C.Y}(Requires main backend at {args.backend_url}){C.E}")
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{args.backend_url}/health")
                if r.status_code != 200:
                    raise Exception("not healthy")
        except Exception:
            print(f"  {C.Y}Backend not reachable — skipping /api/chat tests.{C.E}")
            log("chat", "Backend reachable", False, f"Cannot reach {args.backend_url}")
        else:
            # Send first 3 prompts through the full orchestrator pipeline
            for i, prompt in enumerate(PROMPTS[:3]):
                if i > 0 and args.delay > 0:
                    print(f"  {C.Y}[wait] Waiting {args.delay}s to avoid rate limits...{C.E}")
                    await asyncio.sleep(args.delay)
                await test_backend_chat(args.backend_url, prompt)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print_summary()
    print(f"\n  Total time: {elapsed:.1f}s\n")
    sys.exit(1 if any(not r["pass"] for r in _results) else 0)


if __name__ == "__main__":
    asyncio.run(main())

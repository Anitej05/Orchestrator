"""
Mail Agent High-Fidelity Test Suite
====================================
This script tests all the fixes implemented in this session against the live Mail Agent on port 8040.
Run this after starting the Mail Agent: python agents/mail_agent.py
"""
import httpx
import asyncio
import json

BASE_URL = "http://localhost:8040"
TIMEOUT = 120.0  # seconds

# ============================================================================
# TEST CASES
# ============================================================================

TESTS = [
    # === QUERY GENERATION TESTS ===
    {
        "id": "T01",
        "name": "Quote Balancing (Issue #1)",
        "description": "Ensure quoted phrases are properly balanced in Gmail queries",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Find emails from yesterday with subject 'Q1 Budget Meeting'"}},
        "verify": lambda r: '"' in str(r.get("result", {}).get("results", [{}])[0].get("result", {}).get("query_used", ""))
    },
    {
        "id": "T02",
        "name": "Date Format with Dashes (Issue #2)",
        "description": "Ensure dates in Gmail queries use dashes (YYYY-MM-DD), not slashes",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Show me emails from today"}},
        "verify": lambda r: "-" in str(r.get("result", {}).get("results", [{}])[0].get("result", {}).get("query_used", ""))
    },
    {
        "id": "T03",
        "name": "Empty Search Fallback (Issue #38)",
        "description": "Ensure empty queries fallback to 'label:inbox' instead of matching everything",
        "endpoint": "/search",
        "payload": {"query": "''", "max_results": 5},
        "verify": lambda r: r.get("success") is True and "inbox" in str(r.get("result", {}).get("query_used", "")).lower()
    },
    
    # === DECOMPOSITION TESTS ===
    {
        "id": "T04",
        "name": "Extract Actions Routing (Issue #6)",
        "description": "Ensure 'extract actions' prompts correctly decompose into search + extract_actions",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Find invoice emails and extract invoice numbers and total amounts"}},
        "verify": lambda r: any("extract" in str(step.get("step", "")).lower() for step in r.get("result", {}).get("results", []))
    },
    {
        "id": "T05",
        "name": "Batch Download Decomposition (Issue #9)",
        "description": "Ensure download requests correctly decompose into search + download",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Download all PDF attachments from my recent invoice emails"}},
        "verify": lambda r: any("download" in str(step.get("step", "")).lower() for step in r.get("result", {}).get("results", []))
    },
    
    # === ERROR HANDLING TESTS ===
    {
        "id": "T06",
        "name": "Invalid ID Graceful Error (Issue #36)",
        "description": "Ensure invalid message IDs return a clean error message, not raw API errors",
        "endpoint": "/draft_reply",
        "payload": {"message_id": "invalid_id_12345", "intent": "say thanks"},
        "verify": lambda r: "invalid or not found" in str(r).lower()
    },
    {
        "id": "T07",
        "name": "Invalid ID in Manage Emails",
        "description": "Ensure manage_emails also handles invalid IDs gracefully",
        "endpoint": "/manage_emails",
        "payload": {"message_ids": ["bad-id-xyz"], "action": "archive"},
        "verify": lambda r: "invalid or not found" in str(r).lower()
    },
    
    # === LOSSLESS PROCESSING TESTS ===
    {
        "id": "T08",
        "name": "Summarization with History",
        "description": "Test summarization uses search history when no IDs provided",
        "endpoint": "/summarize_emails",
        "payload": {"use_history": True},
        "verify": lambda r: r.get("success") is True or "No emails" in str(r.get("result", {}).get("summary", ""))
    },
    {
        "id": "T09",
        "name": "Action Extraction with History",
        "description": "Test action extraction uses search history when no IDs provided",
        "endpoint": "/extract_action_items",
        "payload": {"use_history": True},
        "verify": lambda r: r.get("success") is True or "No emails" in str(r.get("error", ""))
    },
    
    # === NEW TESTS ===
    {
        "id": "T10",
        "name": "Complex Multi-Step Prompt",
        "description": "Test a complex prompt that requires multiple steps",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Find my latest 5 unread emails and summarize them"}},
        "verify": lambda r: r.get("result", {}).get("steps_executed", 0) >= 1
    },
    {
        "id": "T11",
        "name": "CC Query Warning (Issue #8)",
        "description": "Ensure CC-related queries still work and don't cause errors",
        "endpoint": "/search",
        "payload": {"query": "emails where I was CC'd", "max_results": 5},
        "verify": lambda r: r.get("success") is True
    },
    {
        "id": "T12",
        "name": "Subject with Special Characters",
        "description": "Test query generation with special characters in subject",
        "endpoint": "/execute",
        "payload": {"payload": {"prompt": "Find emails with subject 'RE: [URGENT] Q1 Budget - Final Review!'"}},
        "verify": lambda r: r.get("result", {}).get("results", [{}])[0].get("step") == "search"
    },
]

# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_test(test):
    test_id = test["id"]
    test_name = test["name"]
    
    print(f"\n{'='*60}")
    print(f"[{test_id}] {test_name}")
    print(f"Description: {test['description']}")
    print(f"Endpoint: {test['endpoint']}")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            res = await client.post(f"{BASE_URL}{test['endpoint']}", json=test["payload"])
            data = res.json()
            
            # Print key info based on endpoint type
            if test["endpoint"] == "/execute":
                results = data.get("result", {}).get("results", [])
                if results:
                    steps = [r.get("step") for r in results]
                    print(f"Steps Executed: {steps}")
                    first_result = results[0].get("result", {})
                    if isinstance(first_result, dict) and "query_used" in first_result:
                        print(f"Query Used: {first_result['query_used']}")
            elif test["endpoint"] == "/search":
                print(f"Query Used: {data.get('result', {}).get('query_used', 'N/A')}")
                print(f"Count: {data.get('result', {}).get('count', 0)}")
            elif test["endpoint"] in ["/draft_reply", "/manage_emails"]:
                # For error-testing endpoints, show the error or success message
                if data.get("success") is False:
                    err = data.get("error") or str(data.get("result", {}).get("details", [{"error": "N/A"}])[0].get("error", "N/A"))
                    print(f"Error: {err}")
                else:
                    print(f"Success: {data.get('success')}")
            elif "error" in data and data.get("success") is False:
                print(f"Error: {data.get('error', 'N/A')}")
            else:
                print(f"Success: {data.get('success')}")
            
            # Run verification
            passed = test["verify"](data)
            status = "PASSED" if passed else "FAILED"
            print(f"Result: {'✅' if passed else '❌'} {status}")
            
            return {"id": test_id, "name": test_name, "passed": passed, "data": data}
            
        except Exception as e:
            print(f"Result: 💥 CRASHED - {e}")
            return {"id": test_id, "name": test_name, "passed": False, "error": str(e)}

async def main():
    print("=" * 60)
    print("MAIL AGENT HIGH-FIDELITY TEST SUITE")
    print("=" * 60)
    
    # Health check
    print("\nChecking server health...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{BASE_URL}/health")
            print(f"Server Status: {res.json()}")
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        print("Please start the Mail Agent: python agents/mail_agent.py")
        return
    
    # Run all tests
    results = []
    for test in TESTS:
        result = await run_test(test)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    
    for r in results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} [{r['id']}] {r['name']}")
    
    print(f"\nTotal: {passed}/{len(results)} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ Some tests failed. Check output above for details.")

if __name__ == "__main__":
    asyncio.run(main())

import httpx
import json
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8040"

SCENARIOS = [
    {
        "name": "Issue #1: Quote Truncation",
        "prompt": "Find emails from yesterday with subject 'Q1 Budget Meeting'",
        "expected_check": lambda r: '"Q1 Budget Meeting"' in str(r.get("result", {}).get("results", [{}])[0].get("result", {}).get("query_used", ""))
    },
    {
        "name": "Issue #2: Date Format (Dashes)",
        "prompt": "Show me emails from today",
        "expected_check": lambda r: "-" in str(r.get("result", {}).get("results", [{}])[0].get("result", {}).get("query_used", "")) and "/" not in str(r.get("result", {}).get("results", [{}])[0].get("result", {}).get("query_used", ""))
    },
    {
        "name": "Issue #6: Extract Actions Decomposition",
        "prompt": "Find invoice emails and extract invoice numbers and total amounts",
        "expected_check": lambda r: any("extract" in str(step["step"]) for step in r.get("result", {}).get("results", []))
    },
    {
        "name": "Issue #9: Batch Download Decomposition",
        "prompt": "Download all PDF attachments from my recent invoice emails",
        "expected_check": lambda r: any("download" in str(step["step"]) for step in r.get("result", {}).get("results", []))
    },
    {
        "name": "Issue #36: Graceful Error for Invalid IDs",
        "endpoint": "/manage_emails",
        "payload": {
            "message_ids": ["invalid-12345"],
            "action": "archive"
        },
        "expected_check": lambda r: "invalid or not found" in str(r).lower()
    },
    {
        "name": "Issue #38: Empty Search Hallucination",
        "endpoint": "/search",
        "payload": {
            "query": "''",
            "max_results": 5
        },
        "expected_check": lambda r: r.get("success") is True
    }
]

async def run_scenario(scenario):
    print(f"\n🚀 Testing: {scenario['name']}")
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            if "endpoint" in scenario:
                res = await client.post(f"{BASE_URL}{scenario['endpoint']}", json=scenario["payload"])
            else:
                res = await client.post(f"{BASE_URL}/execute", json={"payload": {"prompt": scenario["prompt"]}})
            
            data = res.json()
            
            # Print query if available for transparency
            result_obj = data.get("result", {})
            if isinstance(result_obj, dict):
                if "results" in result_obj:
                   step_res = result_obj["results"][0].get("result", {})
                   if isinstance(step_res, dict) and "query_used" in step_res:
                       print(f"   🔍 Generated Query: {step_res['query_used']}")
                elif "query_used" in result_obj:
                   print(f"   🔍 Generated Query: {result_obj['query_used']}")

            # Check logic
            if scenario["expected_check"](data):
                print(f"✅ {scenario['name']} PASSED logic check")
            else:
                print(f"❌ {scenario['name']} FAILED logic check")
                print(f"Output for debug: {data}")
        except Exception as e:
            print(f"💥 {scenario['name']} CRASHED: {e}")

async def main():
    print("=== RIGOROUS REAL-WORLD TEST SUITE (FIXED) ===")
    for s in SCENARIOS:
        await run_scenario(s)

if __name__ == "__main__":
    asyncio.run(main())

import httpx
import json
import asyncio

BASE_URL = "http://localhost:8040"

async def test_quote_balancing():
    print("\n--- Testing Quote Balancing ---")
    # This might be hard to test directly since it happens inside the LLM call, 
    # but we can try to trigger search queries
    payload = {
        "payload": {
            "prompt": "Find emails from yesterday with subject 'Q1 Budget Meeting'"
        }
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{BASE_URL}/execute", json=payload)
            print(f"Status: {response.status_code}")
            # Note: Since it hits real LLM, we just log the outcome
            if response.status_code == 200:
                print("Response received successfully.")
                # We can't easily see the internal query unless we check logs
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

async def test_extract_actions_routing():
    print("\n--- Testing Extract Actions Routing ---")
    payload = {
        "payload": {
            "prompt": "Extract action items from my last search results"
        }
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(f"{BASE_URL}/execute", json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        if "results" in data.get("result", {}):
            steps = [r["step"] for r in data["result"]["results"]]
            print(f"Steps executed: {steps}")
            if any("extract" in s for s in steps):
                print("✅ Successfully routed to extract_actions")
            else:
                print("❌ extract_actions step not found in execution flow")

async def test_invalid_id_graceful_error():
    print("\n--- Testing Invalid ID Graceful Error ---")
    # Try to draft a reply to an invalid ID
    payload = {
        "message_id": "invalid_id_123456789",
        "intent": "tell them I'm busy"
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(f"{BASE_URL}/draft_reply", json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Error Message: {data.get('error')}")
        if data.get("success") is False and "not found" in data.get("error", "").lower():
            print("✅ Successfully caught invalid ID with graceful message")
        else:
            print("❌ Graceful error handling for missing ID failed")

async def test_empty_search():
    print("\n--- Testing Empty Search ---")
    payload = {
        "query": "''",
        "max_results": 5
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{BASE_URL}/search", json=payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Success: {data.get('success')}")
        # According to our fix, it should return label:inbox or similar safely
        if data.get("success"):
            print("✅ Successfully handled empty query fallback")

async def main():
    # Wait for server to be fully ready
    print("Verifying server health...")
    async with httpx.AsyncClient() as client:
        try:
            res = await client.get(f"{BASE_URL}/health")
            print(f"Health check: {res.json()}")
        except Exception:
            print("Server not ready yet, waiting 2 seconds...")
            await asyncio.sleep(2)
    
    await test_quote_balancing()
    await test_extract_actions_routing()
    await test_invalid_id_graceful_error()
    await test_empty_search()

if __name__ == "__main__":
    asyncio.run(main())

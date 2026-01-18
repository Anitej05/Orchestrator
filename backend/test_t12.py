import httpx
import asyncio

async def test_t12():
    print("Testing T12: Subject with Special Characters...")
    payload = {
        "payload": {
            "prompt": "Find emails with subject 'RE: [URGENT] Q1 Budget - Final Review!'"
        }
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            res = await client.post("http://localhost:8040/execute", json=payload)
            data = res.json()
            print(f"Response: {data}")
            
            results = data.get("result", {}).get("results", [])
            if results:
                query_used = results[0].get("result", {}).get("query_used", "N/A")
                print(f"\nQuery Used: {query_used}")
                
                # Check if query is valid (has balanced quotes)
                if query_used.count('"') % 2 == 0:
                    print("✅ Quotes are balanced!")
                else:
                    print("❌ Quotes are unbalanced!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_t12())

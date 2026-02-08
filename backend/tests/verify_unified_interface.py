import httpx
import asyncio
import json

AGENTS = {
    "DocumentAgent": {"url": "http://localhost:8070/execute", "payload": {"prompt": "Summarize this document", "type": "execute"}},
    "ZohoBooksAgent": {"url": "http://localhost:8050/execute", "payload": {"prompt": "List all invoices", "type": "execute"}},
    "BrowserAutomationAgent": {"url": "http://localhost:8090/execute", "payload": {"prompt": "Go to google.com", "type": "execute"}}
}

async def verify_agent(name, config):
    print(f"--- Verifying {name} ---")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(config["url"], json=config["payload"])
            
            if response.status_code == 200:
                print(f"✅ {name}: 200 OK")
                print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
                return True
            elif response.status_code == 404:
                print(f"❌ {name}: 404 Not Found (Endpoint missing!)")
            else:
                print(f"⚠️ {name}: {response.status_code} - {response.text[:100]}")
                # Some agents might return error because they aren't fully configured (e.g. Zoho credentials)
                # But a 200/400/500 with a structured body is better than a 404.
                if response.status_code in [400, 401, 500, 502]:
                    try:
                        data = response.json()
                        if "status" in data or "error" in data:
                            print(f"✅ {name}: Standardized Error/Response received (Endpoint exists)")
                            return True
                    except:
                        pass
            return False
    except Exception as e:
        print(f"❌ {name}: Connection failed - {str(e)}")
        return False

async def main():
    results = []
    for name, config in AGENTS.items():
        results.append(await verify_agent(name, config))
    
    success_count = sum(1 for r in results if r)
    print(f"\nSummary: {success_count}/{len(AGENTS)} agents verified.")

if __name__ == "__main__":
    asyncio.run(main())

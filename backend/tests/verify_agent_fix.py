
import asyncio
import httpx
import json
import socket

async def verify_agent_endpoints():
    print("🔍 Starting agent endpoint verification...")
    
    # Check if port 8070 (DocumentAgent) is open
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8070))
    if result == 0:
        print("✅ Port 8070 is open.")
    else:
        print("❌ Port 8070 is CLOSED. DocumentAgent might not be running.")
        return

    # 1. Test /health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get("http://127.0.0.1:8070/health")
            print(f"Health Response: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"❌ /health check failed: {e}")

        # 2. Test /execute (The new endpoint)
        print("\n📡 Testing /execute endpoint with 'prompt' payload...")
        payload = {
            "type": "execute",
            "action": "/analyze",
            "prompt": "Test health check. Just say 'Agent is online'.",
            "payload": {
                "file_path": "fake.pdf",
                "query": "hello"
            }
        }
        
        try:
            # We use /execute as changed in hands.py
            resp = await client.post("http://127.0.0.1:8070/execute", json=payload)
            print(f"Execute Response Status: {resp.status_code}")
            if resp.status_code == 200:
                print("✅ /execute reached successfully!")
                print(f"Response snippet: {resp.text[:200]}...")
            elif resp.status_code == 404:
                print("❌ /execute returned 404! Endpoint mismatch.")
            else:
                print(f"⚠️ /execute returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"❌ /execute call failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_agent_endpoints())

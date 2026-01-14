import asyncio
import httpx
import json
import os

async def verify_attachments():
    base_url = "http://localhost:8040"
    
    print("🚀 Starting Attachment Verification...")

    # Step 1: Search for the email with the specific subject
    search_prompt = 'Find email with subject "Fwd: Aptitude Training for 2027 batch - for all B.Tech (3rd Year), M.Tech(1st Year) & MCA(1st Year) students || 19th JAN 2026 onwards"'
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Complex prompt without future date constraint from LLM
        complex_prompt = 'Find the email with subject "Fwd: Aptitude Training for 2027 batch - for all B.Tech (3rd Year), M.Tech(1st Year) & MCA(1st Year) students || 19th JAN 2026 onwards", download its attachments, and then send an email to "me" with the subject "Verification Attachment" and body "Here is the file" attaching the downloaded file.'
        
        payload = {
            "action": None,
            "payload": {
                "prompt": complex_prompt,
                "task_id": "verify-attachment-flow"
            },
            "source": "user",
            "target": "mail_agent"
        }
        
        print(f"📝 Sending Prompt: {complex_prompt}")
        
        try:
            response = await client.post(f"{base_url}/execute", json=payload)
            response.raise_for_status()
            result = response.json()
            
            print("\n✅ Response Received:")
            print(json.dumps(result, indent=2))
            
            if result.get("status") == "complete":
                print("\n🎉 Task Completed Successfully!")
                print("Check your email (or sent folder) for 'Verification Attachment'.")
            else:
                print(f"\n⚠️ Status: {result.get('status')}")
                print(f"Error: {result.get('error')}")

        except Exception as e:
            print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_attachments())

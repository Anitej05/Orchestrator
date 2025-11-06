"""
Test if orchestrator can detect live screenshots from browser agent
"""
import asyncio
import httpx

async def test_orchestrator_polling():
    """Submit a browser task and check if orchestrator detects live screenshots"""
    
    # 1. Submit browser task
    print("1. Submitting browser task via orchestrator...")
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Simulate what orchestrator does
        endpoint = "http://localhost:8070/browse/async"
        payload = {"task": "Go to google.com and wait 15 seconds"}
        
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data['task_id']
        print(f"   Task ID: {task_id}")
        
        # 2. Poll like orchestrator does
        print("\n2. Polling for live screenshots (like orchestrator)...")
        status_url = f"http://localhost:8070/browse/status/{task_id}"
        last_screenshot_count = 0
        
        for i in range(20):
            await asyncio.sleep(1)
            
            status_response = await client.get(status_url)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            task_status = status_data.get('status')
            screenshots = status_data.get('live_screenshots', [])
            
            print(f"   Poll #{i+1}: Status={task_status}, Live screenshots={len(screenshots)}")
            
            if screenshots and len(screenshots) > last_screenshot_count:
                new_count = len(screenshots) - last_screenshot_count
                print(f"   ✓ NEW SCREENSHOTS DETECTED: {new_count} new (total: {len(screenshots)})")
                last_screenshot_count = len(screenshots)
            
            if task_status in ['completed', 'failed']:
                break
        
        print(f"\n3. Final: {last_screenshot_count} live screenshots detected during execution")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_polling())

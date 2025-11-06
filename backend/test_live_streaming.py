"""
Test script to verify live screenshot streaming works end-to-end
"""
import asyncio
import httpx
import time

async def test_live_streaming():
    """Test that browser agent captures live screenshots and orchestrator can poll them"""
    
    # 1. Submit browser task
    print("1. Submitting browser task...")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:8070/browse/async",
            json={"task": "Go to google.com and wait 10 seconds"}
        )
        response.raise_for_status()
        result = response.json()
        task_id = result["task_id"]
        print(f"   Task ID: {task_id}")
        
        # 2. Poll for status and check for live screenshots
        print("\n2. Polling for live screenshots...")
        for i in range(15):  # Poll for 15 seconds
            await asyncio.sleep(1)
            
            status_response = await client.get(f"http://localhost:8070/browse/status/{task_id}")
            status_response.raise_for_status()
            status_data = status_response.json()
            
            live_screenshots = status_data.get('live_screenshots', [])
            print(f"   Poll #{i+1}: Status={status_data['status']}, Live screenshots={len(live_screenshots)}")
            
            if live_screenshots:
                print(f"   ✓ Found {len(live_screenshots)} live screenshots!")
                for idx, screenshot in enumerate(live_screenshots):
                    print(f"     - Screenshot {idx+1}: {screenshot.get('file_name')}")
            
            if status_data['status'] in ['completed', 'failed']:
                break
        
        print("\n3. Final status:")
        final_status = await client.get(f"http://localhost:8070/browse/status/{task_id}")
        final_data = final_status.json()
        print(f"   Status: {final_data['status']}")
        print(f"   Live screenshots: {len(final_data.get('live_screenshots', []))}")
        print(f"   Result screenshots: {len(final_data.get('result', {}).get('screenshot_files', []))}")

if __name__ == "__main__":
    asyncio.run(test_live_streaming())

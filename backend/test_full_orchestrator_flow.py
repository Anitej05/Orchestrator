"""
Comprehensive test of orchestrator -> browser agent flow with live screenshot streaming
"""
import asyncio
import httpx
import json

async def test_full_flow():
    """Test the complete orchestration flow with browser agent"""
    
    print("=" * 80)
    print("TESTING FULL ORCHESTRATOR -> BROWSER AGENT FLOW")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Submit task to orchestrator via WebSocket endpoint (simulating frontend)
        print("\n1. Submitting task to orchestrator...")
        
        # Use the REST endpoint for testing (WebSocket would be more complex)
        # Let's directly call the orchestration service
        from main import execute_orchestration
        from shared_state import conversation_store, store_lock
        import uuid
        
        thread_id = str(uuid.uuid4())
        prompt = "Go to google.com and take a screenshot"
        
        print(f"   Thread ID: {thread_id}")
        print(f"   Prompt: {prompt}")
        
        # Track live canvas updates
        canvas_updates_detected = []
        
        async def check_conversation_store():
            """Background task to monitor conversation_store for live canvas updates"""
            while True:
                await asyncio.sleep(0.5)
                with store_lock:
                    if thread_id in conversation_store:
                        state = conversation_store[thread_id]
                        live_update = state.get('live_canvas_update')
                        if live_update:
                            timestamp = live_update.get('timestamp', 0)
                            if timestamp not in [u['timestamp'] for u in canvas_updates_detected]:
                                canvas_updates_detected.append(live_update)
                                print(f"   🎨 LIVE CANVAS UPDATE DETECTED! Count: {len(canvas_updates_detected)}, Screenshots: {live_update.get('screenshot_count', 0)}")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(check_conversation_store())
        
        try:
            print("\n2. Executing orchestration...")
            final_state = await execute_orchestration(
                prompt=prompt,
                thread_id=thread_id,
                user_response=None,
                files=None,
                stream_callback=None,
                planning_mode=False
            )
            
            print("\n3. Orchestration completed!")
            print(f"   Final response: {final_state.get('final_response', 'N/A')[:200]}...")
            print(f"   Has canvas: {final_state.get('has_canvas', False)}")
            print(f"   Canvas type: {final_state.get('canvas_type', 'N/A')}")
            
            print(f"\n4. Live canvas updates detected during execution: {len(canvas_updates_detected)}")
            for i, update in enumerate(canvas_updates_detected):
                print(f"   Update {i+1}: {update.get('screenshot_count', 0)} screenshots at {update.get('timestamp', 0)}")
            
            if len(canvas_updates_detected) > 0:
                print("\n[SUCCESS] Live canvas streaming is working!")
            else:
                print("\n[FAILURE] No live canvas updates detected")
                print("   This means orchestrator is not updating conversation_store during browser execution")
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(test_full_flow())

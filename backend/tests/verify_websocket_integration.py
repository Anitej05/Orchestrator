import asyncio
import websockets
import json
import uuid

async def test_websocket():
    uri = "ws://127.0.0.1:8000/ws/chat"
    thread_id = str(uuid.uuid4())
    
    print(f"Connecting to {uri} with thread_id {thread_id}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send a simple prompt that requires planning and execution
            prompt = "Create a file named verification_test.txt with the content 'Hello World'."
            
            payload = {
                "prompt": prompt,
                "thread_id": thread_id,
                "owner": {"user_id": "test_user"}
            }
            
            await websocket.send(json.dumps(payload))
            print(f"Sent prompt: {prompt}")
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                node = data.get("node")
                
                print(f"Received message from node: {node}")
                
                if node == "manage_todo_list":
                    tasks = data.get("data", {}).get("task_names", [])
                    print(f"✅ VERIFIED: Brain node received. Tasks identified: {len(tasks)}")
                    print(f"   Tasks: {tasks}")
                    
                elif node == "execute_next_action":
                    print(f"DEBUG: Full execution data: {data}")
                    task_desc = data.get("data", {}).get("description")
                    executed_id = data.get("data", {}).get("executed_task_id")
                    print(f"✅ VERIFIED: Execution node received. Executed task: {task_desc} (ID: {executed_id})")
                    
                elif node == "__end__":
                    print("Workflow finished.")
                    break
                    
                elif node == "__error__":
                    print(f"❌ ERROR: {data.get('error')}")
                    break

    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure the backend is running on localhost:8000")

if __name__ == "__main__":
    asyncio.run(test_websocket())

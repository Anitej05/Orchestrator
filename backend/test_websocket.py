"""
Simple WebSocket test script to verify the connection works.
Run this after starting the backend server.
"""
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/chat"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ WebSocket connected successfully!")
            
            # Send a test message
            test_message = {
                "prompt": "Hello, this is a test message",
                "planning_mode": False
            }
            print(f"Sending test message: {test_message}")
            await websocket.send(json.dumps(test_message))
            print("✓ Message sent successfully!")
            
            # Wait for response
            print("Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"✓ Received response: {response[:200]}...")
            
            return True
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"✗ Connection failed with status code: {e.status_code}")
        print(f"  Headers: {e.headers}")
        return False
    except asyncio.TimeoutError:
        print("✗ Timeout waiting for response")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Connection Test")
    print("=" * 60)
    print()
    
    result = asyncio.run(test_websocket())
    
    print()
    print("=" * 60)
    if result:
        print("✓ WebSocket test PASSED")
    else:
        print("✗ WebSocket test FAILED")
    print("=" * 60)

import httpx
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_chat(prompt):
    print(f"\n--- Testing prompt: {prompt} ---")
    try:
        response = httpx.post(
            f"{BASE_URL}/api/chat",
            json={"prompt": prompt},
            timeout=60.0
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('final_response', data.get('message'))[:500]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    # Test 1: Simple Info
    test_chat("Who are you?")
    
    # Test 2: Tool Call (Finance)
    test_chat("What is the current price of NVDA stock?")
    
    # Test 3: Complex Multi-Agent (Hypothetical search)
    test_chat("Search for news about artificial intelligence and give me a summary.")

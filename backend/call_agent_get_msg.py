
import httpx
import json

URL = "http://localhost:8040/get_message"
MSG_ID = "19bb851e8b34a3a6"

def call_agent():
    try:
        payload = {
            "operation": "get_message",
            "parameters": {
                "message_id": MSG_ID
            }
        }
        response = httpx.post(URL, json=payload)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    call_agent()

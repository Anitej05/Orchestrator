
import os
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

def trigger_error():
    api_key = os.getenv("COMPOSIO_API_KEY")
    client = Composio(api_key=api_key)
    
    # Try to execute with a known invalid param to get the list of valid ones
    print("Executing GMAIL_SEND_EMAIL with invalid params...")
    try:
        res = client.actions.execute(
            action=Action.GMAIL_SEND_EMAIL,
            params={"DUMMY_PARAM_FOR_SCHEMA": "test"},
            connected_account=os.getenv("GMAIL_CONNECTION_ID")
        )
        print(f"Result: {res}")
    except Exception as e:
        print(f"Caught Exception: {e}")

if __name__ == "__main__":
    trigger_error()


import asyncio
import os
import json
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

async def test_send_singular():
    api_key = os.getenv("COMPOSIO_API_KEY")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    client = Composio(api_key=api_key)
    
    # Correct filename
    local_path = os.path.join("storage", "gmail_attachments", "796d81e3-a2e4-46dc-a6b1-7811aa5423b2.pdf")
    
    if not os.path.exists(local_path):
        # Try full absolute path
        abs_path = os.path.abspath(local_path)
        if os.path.exists(abs_path):
            local_path = abs_path
        else:
            print(f"File not found: {local_path} or {abs_path}")
            return

    print(f"Testing send with 'attachment' parameter and path: {local_path}")
    
    res = client.actions.execute(
        action=Action.GMAIL_SEND_EMAIL,
        params={
            "recipient_email": "anitej473@gmail.com",
            "subject": "Test Singular Attachment V2",
            "body": "This is a test with singular 'attachment' key and correct path.",
            "attachment": local_path
        },
        connected_account=connection_id
    )
    
    print(f"Result: {res}")
    if res.get("successful") or res.get("successfull"):
        msg_id = res.get("data", {}).get("id")
        print(f"Sent Message ID: {msg_id}")
    else:
        print("Send failed.")

if __name__ == "__main__":
    asyncio.run(test_send_singular())


import asyncio
import os
import json
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

async def test_send_object():
    api_key = os.getenv("COMPOSIO_API_KEY")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    client = Composio(api_key=api_key)
    
    # 1. Fetch message to get attachment info
    msg_id = "19bb851e8b34a3a6"
    print(f"Fetching message {msg_id}...")
    fetch_res = client.actions.execute(
        action=Action.GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID,
        params={"message_id": msg_id, "format": "full"},
        connected_account=connection_id
    )
    
    data = fetch_res.get("data", {})
    att_list = data.get("attachmentList", [])
    if not att_list:
        print("No attachments to test with.")
        return
        
    att = att_list[0]
    att_id = att["attachmentId"]
    filename = att["filename"]
    
    # 2. Get attachment metadata (to get s3url/key)
    print(f"Getting attachment metadata for {filename}...")
    dl_res = client.actions.execute(
        action=Action.GMAIL_GET_ATTACHMENT,
        params={
            "message_id": msg_id,
            "attachment_id": att_id,
            "file_name": filename
        },
        connected_account=connection_id
    )
    
    if not dl_res.get("successful") and not dl_res.get("successfull"):
        print(f"Get attachment failed: {dl_res}")
        return
        
    file_info = dl_res.get("data", {}).get("file")
    print(f"File info received: {file_info}")
    
    if not file_info:
        print("No file info in response.")
        return

    # 3. Try to send with this object
    print("Testing send with 'attachment' parameter as OBJECT...")
    
    res = client.actions.execute(
        action=Action.GMAIL_SEND_EMAIL,
        params={
            "recipient_email": "anitej473@gmail.com",
            "subject": "Test Object Attachment",
            "body": "This is a test with 'attachment' as a dictionary.",
            "attachment": file_info
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
    asyncio.run(test_send_object())

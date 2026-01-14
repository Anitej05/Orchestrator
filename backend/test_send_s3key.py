
import asyncio
import os
import json
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

async def test_send_s3key():
    api_key = os.getenv("COMPOSIO_API_KEY")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    client = Composio(api_key=api_key)
    
    msg_id = "19bb851e8b34a3a6"
    print(f"Fetching message {msg_id}...")
    fetch_res = client.actions.execute(
        action=Action.GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID,
        params={"message_id": msg_id, "format": "full"},
        connected_account=connection_id
    )
    
    data = fetch_res.get("data", {})
    att = data.get("attachmentList", [])[0]
    
    print(f"Getting attachment metadata...")
    dl_res = client.actions.execute(
        action=Action.GMAIL_GET_ATTACHMENT,
        params={
            "message_id": msg_id,
            "attachment_id": att["attachmentId"],
            "file_name": att["filename"]
        },
        connected_account=connection_id
    )
    
    file_info = dl_res.get("data", {}).get("file")
    if not file_info:
        print("No file info.")
        return

    # Try mapping s3url -> s3key
    attachment_obj = {
        "s3key": file_info["s3url"],
        "mimetype": file_info["mimetype"],
        "name": file_info["name"]
    }
    
    print(f"Testing send with s3key=s3url: {attachment_obj['s3key'][:50]}...")
    
    res = client.actions.execute(
        action=Action.GMAIL_SEND_EMAIL,
        params={
            "recipient_email": "anitej473@gmail.com",
            "subject": "Test S3Key Mapping",
            "body": "This is a test mapping s3url to s3key.",
            "attachment": attachment_obj
        },
        connected_account=connection_id
    )
    
    print(f"Result: {res}")
    if res.get("successful") or res.get("successfull"):
        print("Success!")
    else:
        print("Failed.")

if __name__ == "__main__":
    asyncio.run(test_send_s3key())

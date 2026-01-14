
import asyncio
import os
import json
from composio import Composio, Action
from dotenv import load_dotenv

load_dotenv()

async def test_download():
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
    att_id = att["attachmentId"]
    filename = att["filename"]
    
    print(f"Executing GMAIL_GET_ATTACHMENT for {filename}...")
    dl_res = client.actions.execute(
        action=Action.GMAIL_GET_ATTACHMENT,
        params={
            "message_id": msg_id,
            "attachment_id": att_id,
            "file_name": filename
        },
        connected_account=connection_id
    )
    
    if dl_res.get("successful") or dl_res.get("successfull"):
        data = dl_res.get("data", {})
        print(f"Top-level keys: {list(data.keys())}")
        file_info = data.get("file")
        print(f"Type of 'file': {type(file_info)}")
        if file_info:
            if isinstance(file_info, dict):
                print("Iterating over dict keys:")
                for k, v in file_info.items():
                    print(f"  Key: '{k}', Type: {type(v)}, Value: {v}")
            else:
                print(f"Value of 'file' is not a dict: {file_info}")
                # Maybe try to see if it has attributes?
                try: print(f"Attributes: {dir(file_info)}")
                except: pass
        else:
            print("No 'file' key found in data.")
    else:
        print(f"Action failed: {dl_res}")

if __name__ == "__main__":
    asyncio.run(test_download())

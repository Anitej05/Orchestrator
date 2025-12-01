#!/usr/bin/env python3
"""Try to read unread emails right now"""

import os
from dotenv import load_dotenv
from composio import Composio, Action

load_dotenv()

api_key = os.getenv("COMPOSIO_API_KEY")
connection_id = os.getenv("GMAIL_CONNECTION_ID")

print(f"API Key: {api_key[:15]}...")
print(f"Connection ID: {connection_id}")
print()

client = Composio(api_key=api_key)

print("Attempting to fetch unread emails...")
print()

try:
    result = client.actions.execute(
        action=Action.GMAIL_FETCH_EMAILS,
        params={
            "query": "is:unread",
            "max_results": 5
        },
        connected_account=connection_id
    )
    
    print("Result:")
    print(result)
    print()
    
    if result.get("successful") or result.get("successfull"):
        data = result.get("data", {})
        if "messages" in data:
            messages = data["messages"]
            print(f"Found {len(messages)} unread emails:")
            for i, msg in enumerate(messages, 1):
                print(f"\n{i}. Email:")
                print(f"   ID: {msg.get('id')}")
                print(f"   Snippet: {msg.get('snippet', 'N/A')[:100]}")
        else:
            print("No messages in response")
            print(f"Data: {data}")
    else:
        print(f"Error: {result.get('error')}")
        print(f"Full response: {result}")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()

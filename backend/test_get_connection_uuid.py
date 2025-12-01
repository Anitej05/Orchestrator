#!/usr/bin/env python3
"""Get the proper UUID for Gmail connection"""

import os
from dotenv import load_dotenv
from composio import Composio

load_dotenv()

api_key = os.getenv("COMPOSIO_API_KEY")
connection_id = os.getenv("GMAIL_CONNECTION_ID")

print(f"API Key: {api_key[:15]}...")
print(f"Connection ID from env: {connection_id}")
print()

client = Composio(api_key=api_key)

print("Listing connected accounts...")
try:
    # Get connected accounts
    accounts = client.connected_accounts.get()
    
    print(f"Found {len(accounts)} connected account(s):")
    print()
    
    for account in accounts:
        print(f"Account:")
        print(f"  ID: {account.id}")
        print(f"  App: {account.appName}")
        print(f"  Status: {account.status}")
        print(f"  Integration ID: {account.integrationId}")
        print()
        
        # Check if this is the Gmail account
        if 'gmail' in account.appName.lower():
            print(f"✓ This is the Gmail account!")
            print(f"  Use this ID: {account.id}")
            print()
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

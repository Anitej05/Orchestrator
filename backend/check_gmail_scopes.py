#!/usr/bin/env python3
"""Check current Gmail connection scopes"""

import os
from dotenv import load_dotenv
from composio import Composio

load_dotenv()

api_key = os.getenv("COMPOSIO_API_KEY")
connection_id = os.getenv("GMAIL_CONNECTION_ID")

print(f"Checking Gmail connection: {connection_id}")
print()

client = Composio(api_key=api_key)

try:
    accounts = client.connected_accounts.get()
    
    for account in accounts:
        if account.id == connection_id:
            print(f"Gmail Account Details:")
            print(f"  ID: {account.id}")
            print(f"  App: {account.appName}")
            print(f"  Status: {account.status}")
            print(f"  Created: {account.createdAt}")
            print()
            
            # Check if there's scope information
            if hasattr(account, 'scopes'):
                print(f"  Scopes: {account.scopes}")
            
            if hasattr(account, 'authConfig'):
                print(f"  Auth Config: {account.authConfig}")
            
            # Print all attributes
            print("\n  All attributes:")
            for attr in dir(account):
                if not attr.startswith('_'):
                    try:
                        value = getattr(account, attr)
                        if not callable(value):
                            print(f"    {attr}: {value}")
                    except:
                        pass
            
            break
    
    print()
    print("To fix this, you need to re-authorize Gmail with proper scopes.")
    print("Run: python setup_composio_gmail.py")
    print("When asked for entity ID, enter: default")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

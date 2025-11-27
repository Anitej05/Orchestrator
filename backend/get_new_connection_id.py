#!/usr/bin/env python3
"""Get the UUID for the new Gmail connection"""

import os
from dotenv import load_dotenv
from composio import Composio

load_dotenv()

api_key = os.getenv("COMPOSIO_API_KEY")
client = Composio(api_key=api_key)

print("Fetching all Gmail connections...")
print()

accounts = client.connected_accounts.get()

for account in accounts:
    if 'gmail' in account.appName.lower():
        print(f"Gmail Connection:")
        print(f"  ID (UUID): {account.id}")
        print(f"  Client ID: {account.clientUniqueUserId}")
        print(f"  Status: {account.status}")
        print(f"  Created: {account.createdAt}")
        print(f"  Updated: {account.updatedAt}")
        
        # Check scopes
        if hasattr(account, 'connectionParams'):
            params = account.connectionParams
            scope = params.scope if hasattr(params, 'scope') else None
            print(f"  Scopes: {scope if scope else 'None'}")
        
        print()

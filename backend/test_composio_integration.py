#!/usr/bin/env python3
"""
Test Composio Integration - Verify MCP Server Setup
Based on Composio's latest documentation
"""

import os
from dotenv import load_dotenv
from composio import Composio, Action

load_dotenv()

api_key = os.getenv("COMPOSIO_API_KEY")
connection_id = os.getenv("GMAIL_CONNECTION_ID")

print("=" * 70)
print("Composio Integration Test")
print("=" * 70)
print()

# Step 1: Verify API Key
print("Step 1: Verify API Key")
print("-" * 70)
print(f"API Key: {api_key[:15]}...")

client = Composio(api_key=api_key)

try:
    # Validate API key by trying to get accounts
    _ = client.connected_accounts.get()
    print("✅ API Key is valid")
except Exception as e:
    print(f"❌ API Key validation failed: {e}")
    exit(1)

print()

# Step 2: Check Connected Accounts
print("Step 2: Check Connected Accounts")
print("-" * 70)

try:
    accounts = client.connected_accounts.get()
    print(f"Found {len(accounts)} connected account(s)")
    
    gmail_account = None
    for account in accounts:
        if account.id == connection_id or 'gmail' in account.appName.lower():
            gmail_account = account
            print(f"\n✅ Gmail Account:")
            print(f"   ID: {account.id}")
            print(f"   App: {account.appName}")
            print(f"   Status: {account.status}")
            print(f"   Entity: {account.entityId}")
            
            # Check connection params
            if hasattr(account, 'connectionParams'):
                params = account.connectionParams
                print(f"   Scope: {params.scope if hasattr(params, 'scope') else 'Not available'}")
            break
    
    if not gmail_account:
        print("❌ No Gmail account found")
        print("\nYou need to connect Gmail first:")
        print("1. Go to https://app.composio.dev")
        print("2. Navigate to 'Connected Accounts'")
        print("3. Click 'Add Account' → Gmail")
        print("4. Complete OAuth flow with ALL permissions")
        exit(1)
        
except Exception as e:
    print(f"❌ Error checking accounts: {e}")
    exit(1)

print()

# Step 3: Test Gmail Action
print("Step 3: Test Gmail Action (GMAIL_FETCH_EMAILS)")
print("-" * 70)

try:
    result = client.actions.execute(
        action=Action.GMAIL_FETCH_EMAILS,
        params={
            "query": "is:unread",
            "max_results": 3
        },
        connected_account=gmail_account.id
    )
    
    print(f"Response: {result}")
    print()
    
    if result.get("successful") or result.get("successfull"):
        print("✅ SUCCESS! Gmail integration is working!")
        
        data = result.get("data", {})
        if isinstance(data, dict) and "messages" in data:
            messages = data["messages"]
            print(f"\nFound {len(messages)} unread email(s):")
            for i, msg in enumerate(messages, 1):
                print(f"\n{i}. {msg.get('snippet', 'No snippet')[:80]}...")
        else:
            print(f"\nData: {data}")
    else:
        error = result.get("error", "Unknown error")
        print(f"❌ FAILED: {error}")
        
        if "insufficient authentication scopes" in error.lower():
            print("\n" + "=" * 70)
            print("SOLUTION: Re-authorize Gmail with proper scopes")
            print("=" * 70)
            print()
            print("The Gmail connection doesn't have permission to read emails.")
            print()
            print("Fix this by:")
            print("1. Go to: https://app.composio.dev/apps/gmail")
            print("2. Click on your connected Gmail account")
            print("3. Click 'Reconnect' or 'Edit Scopes'")
            print("4. Make sure these scopes are enabled:")
            print("   - gmail.readonly (read emails)")
            print("   - gmail.send (send emails)")
            print("   - gmail.modify (manage labels)")
            print("5. Complete the OAuth flow")
            print()
            print("OR use the Composio CLI:")
            print("   composio add gmail --force")
            print()
        
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("Test Complete")
print("=" * 70)

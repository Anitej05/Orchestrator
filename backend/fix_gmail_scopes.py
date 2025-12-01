#!/usr/bin/env python3
"""
Fix Gmail scopes by requesting a new connection with proper permissions
"""

import os
from dotenv import load_dotenv, set_key
from pathlib import Path

load_dotenv()

print("=" * 70)
print("Fix Gmail Scopes")
print("=" * 70)
print()

api_key = os.getenv("COMPOSIO_API_KEY")
entity_id = "default"

if not api_key:
    print("❌ COMPOSIO_API_KEY not found in .env")
    exit(1)

print(f"API Key: {api_key[:15]}...")
print(f"Entity: {entity_id}")
print()

try:
    from composio import Composio
    
    client = Composio(api_key=api_key)
    entity = client.get_entity(id=entity_id)
    
    print("Step 1: Initiating new Gmail connection with full scopes...")
    print()
    
    # Request new connection
    connection_request = entity.initiate_connection(
        app_name="gmail",
        redirect_url="http://localhost:3000/auth/callback",
        force_new_integration=False  # Reuse existing integration
    )
    
    print("=" * 70)
    print("ACTION REQUIRED: Authorize Gmail")
    print("=" * 70)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {connection_request.redirectUrl}")
    print()
    print("2. Sign in to your Gmail account")
    print("3. IMPORTANT: Grant ALL requested permissions")
    print("   (including read and send email)")
    print("4. Complete the authorization")
    print()
    print("=" * 70)
    print()
    input("Press Enter AFTER you've completed the authorization...")
    print()
    
    # Check for new connection
    print("Step 2: Verifying new connection...")
    connections = entity.get_connections(app_name="gmail")
    
    if connections:
        # Get the most recent connection
        connection = connections[0]
        print(f"✅ Gmail connection found!")
        print(f"   ID: {connection.id}")
        print(f"   Status: {connection.status}")
        print()
        
        # Update .env with new connection ID
        env_path = Path(".env")
        set_key(env_path, "GMAIL_CONNECTION_ID", connection.id)
        
        print("✅ Updated .env with new connection ID")
        print()
        
        # Test the connection
        print("Step 3: Testing email access...")
        from composio import Action
        
        try:
            result = client.actions.execute(
                action=Action.GMAIL_FETCH_EMAILS,
                params={"query": "is:unread", "max_results": 1},
                connected_account=connection.id
            )
            
            if result.get("successful") or result.get("successfull"):
                print("✅ SUCCESS! Gmail agent can now read emails!")
                print()
                print(f"Test result: {result.get('data', {})}")
            else:
                print(f"⚠️  Still having issues: {result.get('error')}")
                print()
                print("You may need to:")
                print("1. Delete the old connection in Composio dashboard")
                print("2. Run this script again")
        except Exception as e:
            print(f"⚠️  Test failed: {e}")
            print("The connection is created but may need a moment to activate")
        
        print()
        print("=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print()
        print("Your Gmail agent is now configured.")
        print("Try: python test_read_emails_now.py")
        
    else:
        print("❌ No connection found")
        print("Please make sure you completed the OAuth flow")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

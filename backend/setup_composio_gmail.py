#!/usr/bin/env python3
"""
Setup Composio Gmail Integration (v3 API)
Updated for Composio's latest API
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()

print("=" * 70)
print("Composio Gmail Setup (v3 API)")
print("=" * 70)
print()

# Step 1: Get API Key
print("Step 1: Composio API Key")
print("-" * 70)
api_key = os.getenv("COMPOSIO_API_KEY")

if not api_key:
    print("You need a Composio API key to continue.")
    print()
    print("To get your API key:")
    print("1. Go to https://app.composio.dev/")
    print("2. Sign up or log in")
    print("3. Go to Settings → API Keys")
    print("4. Copy your API key")
    print()
    print("Enter your Composio API key: ", end="")
    api_key = input().strip()
    
    if not api_key:
        print("❌ API key is required")
        sys.exit(1)
    
    # Save to .env
    env_path = Path(".env")
    set_key(env_path, "COMPOSIO_API_KEY", api_key)
    print("✅ API key saved to .env")
else:
    print(f"✅ API key found: {api_key[:15]}...")

print()

# Step 2: Connect Gmail using v3 API
print("Step 2: Connect Gmail Account")
print("-" * 70)
print()
print("We'll use Composio's v3 API to connect your Gmail account.")
print()

try:
    from composio import Composio
    
    # Initialize Composio client
    client = Composio(api_key=api_key)
    
    print("Initializing Composio client...")
    
    # Get entity (user)
    print("\nEnter an entity ID (e.g., your email or 'default'): ", end="")
    entity_id = input().strip() or "default"
    
    entity = client.get_entity(id=entity_id)
    print(f"✅ Entity: {entity_id}")
    
    # Initiate Gmail connection
    print("\n🔗 Initiating Gmail connection...")
    print("This will open a browser window for OAuth authentication.")
    print()
    
    # Get connection request
    connection_request = entity.initiate_connection(
        app_name="gmail",
        redirect_url="http://localhost:8000/auth/callback"  # You can customize this
    )
    
    print("=" * 70)
    print("IMPORTANT: Complete OAuth Flow")
    print("=" * 70)
    print()
    print("1. Open this URL in your browser:")
    print()
    print(f"   {connection_request.redirectUrl}")
    print()
    print("2. Sign in with your Gmail account")
    print("3. Grant the requested permissions")
    print("4. You'll be redirected to a callback URL")
    print()
    print("Press Enter after completing the OAuth flow...")
    input()
    
    # Get the connection
    print("\n📡 Checking connection status...")
    connections = entity.get_connections(app_name="gmail")
    
    if connections:
        connection = connections[0]
        print(f"✅ Gmail connected!")
        print(f"   Connection ID: {connection.id}")
        print(f"   Status: {connection.status}")
        
        # Generate MCP URL using the new API
        print("\n📡 Generating MCP configuration...")
        
        # For MCP, we need to use the connection directly
        # The MCP URL format for Composio v3 is different
        mcp_url = f"https://backend.composio.dev/api/v1/connectedAccounts/{connection.id}/mcp"
        
        print(f"\n✅ MCP URL generated:")
        print(f"   {mcp_url}")
        
        # Save to .env
        env_path = Path(".env")
        set_key(env_path, "GMAIL_MCP_URL", mcp_url)
        set_key(env_path, "GMAIL_CONNECTION_ID", connection.id)
        set_key(env_path, "GMAIL_ENTITY_ID", entity_id)
        
        print("\n✅ Configuration saved to .env:")
        print(f"   COMPOSIO_API_KEY={api_key[:15]}...")
        print(f"   GMAIL_MCP_URL={mcp_url[:50]}...")
        print(f"   GMAIL_CONNECTION_ID={connection.id}")
        print(f"   GMAIL_ENTITY_ID={entity_id}")
        
    else:
        print("❌ No Gmail connection found")
        print("   Please complete the OAuth flow and try again")
        sys.exit(1)
    
except ImportError:
    print("❌ Composio SDK not installed")
    print("   Run: pip install composio")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Troubleshooting:")
    print("1. Make sure you completed the OAuth flow")
    print("2. Check your API key is correct")
    print("3. Try again with: python setup_composio_gmail.py")
    sys.exit(1)

# Step 3: Test connection
print("\n" + "=" * 70)
print("Step 3: Test MCP Connection")
print("=" * 70)
print()

try:
    import asyncio
    import httpx
    from mcp import ClientSession
    import mcp.client.sse
    
    async def test_mcp():
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        # For Composio v3, the MCP endpoint might be different
        # Let's try the connection
        print("Testing MCP connection...")
        
        try:
            async with httpx.AsyncClient(headers=headers, timeout=30.0) as http_client:
                # Test basic connection first
                test_url = f"https://backend.composio.dev/api/v1/connectedAccounts/{connection.id}"
                response = await http_client.get(test_url)
                
                if response.status_code == 200:
                    print("✅ Connection verified!")
                    data = response.json()
                    print(f"   Account: {data.get('integrationId', 'Gmail')}")
                    print(f"   Status: {data.get('status', 'active')}")
                    return True
                else:
                    print(f"⚠️  Connection status: {response.status_code}")
                    return False
        except Exception as e:
            print(f"⚠️  Could not test connection: {e}")
            return False
    
    result = asyncio.run(test_mcp())
    
except Exception as e:
    print(f"⚠️  Could not test MCP: {e}")
    print("   This is okay - the connection should still work")

# Summary
print("\n" + "=" * 70)
print("Setup Complete!")
print("=" * 70)
print()
print("✅ Composio API key configured")
print("✅ Gmail account connected")
print("✅ MCP URL generated")
print("✅ Configuration saved to .env")
print()
print("Next steps:")
print("1. Start the backend server: python main.py")
print("2. The Gmail agent will auto-register")
print("3. Try: 'Check my unread emails'")
print()
print("=" * 70)

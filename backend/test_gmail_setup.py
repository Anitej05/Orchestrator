#!/usr/bin/env python3
"""
Quick test script for Gmail MCP Agent setup
Guides you through Composio configuration and tests the connection.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("Gmail MCP Agent - Setup Test")
print("=" * 70)
print()

# Step 1: Check Composio installation
print("Step 1: Checking Composio installation...")
try:
    import composio
    print(f"✅ Composio installed (version {composio.__version__})")
except ImportError:
    print("❌ Composio not installed")
    print("   Run: pip install composio-core")
    sys.exit(1)

# Step 2: Check MCP installation
print("\nStep 2: Checking MCP installation...")
try:
    import mcp
    print(f"✅ MCP installed")
except ImportError:
    print("❌ MCP not installed")
    print("   Run: pip install mcp httpx-sse")
    sys.exit(1)

# Step 3: Check environment variables
print("\nStep 3: Checking environment variables...")
api_key = os.getenv("COMPOSIO_API_KEY")
mcp_url = os.getenv("GMAIL_MCP_URL")

if not api_key:
    print("❌ COMPOSIO_API_KEY not set")
    print()
    print("To set up Composio:")
    print("1. Sign up at https://composio.dev/dashboard")
    print("2. Get your API key from the dashboard")
    print("3. Add to backend/.env:")
    print("   COMPOSIO_API_KEY=cl_your_key_here")
    print()
    print("Would you like to set it up now? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        print("\nPlease enter your Composio API key:")
        api_key = input().strip()
        # Save to .env
        env_path = Path(".env")
        with open(env_path, 'a') as f:
            f.write(f"\nCOMPOSIO_API_KEY={api_key}\n")
        print("✅ API key saved to .env")
        os.environ["COMPOSIO_API_KEY"] = api_key
    else:
        sys.exit(1)
else:
    print(f"✅ COMPOSIO_API_KEY found: {api_key[:10]}...")

if not mcp_url:
    print("⚠️  GMAIL_MCP_URL not set")
    print()
    print("To generate MCP URL:")
    print("1. First, connect your Gmail account")
    print("2. Then generate the MCP URL")
    print()
    print("Would you like to do this now? (y/n): ", end="")
    response = input().strip().lower()
    if response == 'y':
        print("\n" + "=" * 70)
        print("Gmail Connection Setup")
        print("=" * 70)
        print()
        print("This will:")
        print("1. Open Google OAuth consent screen")
        print("2. Ask for Gmail permissions")
        print("3. Return a connected_account_id")
        print()
        print("Press Enter to continue...")
        input()
        
        # Try to connect Gmail
        try:
            from composio import Composio
            composio_client = Composio(api_key=api_key)
            
            print("\n🔗 Connecting to Gmail...")
            print("A browser window will open for OAuth authentication.")
            print("Please grant the requested permissions.")
            print()
            
            # Note: This requires the composio CLI which might not be available
            # We'll provide manual instructions instead
            print("Please run this command in your terminal:")
            print()
            print("  composio add gmail")
            print()
            print("After connecting, you'll receive a connected_account_id (e.g., ca_gmail_456)")
            print("Save this ID and press Enter to continue...")
            input()
            
            print("\nPlease enter your connected_account_id:")
            connected_account_id = input().strip()
            
            print("\nPlease enter your Gmail address:")
            user_email = input().strip()
            
            # Generate MCP URL
            print("\n📡 Generating MCP URL...")
            try:
                mcp_config = composio_client.mcp.create(
                    toolkit="gmail",
                    auth_config_id=connected_account_id,
                    user_id=user_email
                )
                mcp_url = composio_client.mcp.generate(
                    config_id=mcp_config.id,
                    transport="sse"
                )
                
                print(f"\n✅ MCP URL generated:")
                print(f"   {mcp_url}")
                
                # Save to .env
                env_path = Path(".env")
                with open(env_path, 'a') as f:
                    f.write(f"\nGMAIL_MCP_URL={mcp_url}\n")
                print("\n✅ MCP URL saved to .env")
                os.environ["GMAIL_MCP_URL"] = mcp_url
                
            except Exception as e:
                print(f"\n❌ Failed to generate MCP URL: {e}")
                print("\nManual setup required:")
                print("Run this Python code:")
                print(f"""
from composio import Composio
composio = Composio(api_key="{api_key}")
mcp_config = composio.mcp.create(
    toolkit="gmail",
    auth_config_id="{connected_account_id}",
    user_id="{user_email}"
)
mcp_url = composio.mcp.generate(config_id=mcp_config.id, transport="sse")
print("MCP URL:", mcp_url)
""")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    else:
        print("\nPlease set up Composio manually and add GMAIL_MCP_URL to .env")
        sys.exit(1)
else:
    print(f"✅ GMAIL_MCP_URL found: {mcp_url[:50]}...")

# Step 4: Test MCP connection
print("\nStep 4: Testing MCP connection...")
try:
    import asyncio
    import httpx
    from mcp import ClientSession
    import mcp.client.sse
    
    async def test_connection():
        headers = {"x-api-key": api_key}
        sse_url = f"{mcp_url}/sse" if not mcp_url.endswith("/sse") else mcp_url
        
        print(f"   Connecting to: {sse_url[:60]}...")
        
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as http_client:
            async with mcp.client.sse.sse_client(sse_url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize
                    init_result = await session.initialize()
                    print(f"   ✅ Connected to: {init_result.serverInfo.name}")
                    
                    # List tools
                    tools_response = await session.list_tools()
                    tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                    
                    print(f"   ✅ Discovered {len(tools)} Gmail tools")
                    if tools:
                        print("\n   Available tools:")
                        for tool in tools[:5]:
                            print(f"      - {tool.name}")
                        if len(tools) > 5:
                            print(f"      ... and {len(tools) - 5} more")
                    
                    return True
    
    result = asyncio.run(test_connection())
    if result:
        print("\n✅ MCP connection test passed!")
    
except Exception as e:
    print(f"\n❌ MCP connection test failed: {e}")
    print("\nTroubleshooting:")
    print("1. Verify COMPOSIO_API_KEY is correct")
    print("2. Verify GMAIL_MCP_URL includes user_id and connected_account_id")
    print("3. Check if Gmail account is connected: composio apps")
    print("4. Try reconnecting: composio add gmail --force")
    sys.exit(1)

# Step 5: Check agent definition
print("\nStep 5: Checking agent definition...")
agent_json = Path("Agent_entries/gmail_mcp_agent.json")
if agent_json.exists():
    print(f"✅ Agent definition found: {agent_json}")
else:
    print(f"❌ Agent definition not found: {agent_json}")
    sys.exit(1)

# Step 6: Check database (optional)
print("\nStep 6: Checking database registration...")
try:
    from database import SessionLocal
    from models import Agent
    
    db = SessionLocal()
    try:
        agent = db.query(Agent).filter_by(id="gmail_mcp_agent").first()
        if agent:
            print(f"✅ Agent registered in database")
            print(f"   Name: {agent.name}")
            print(f"   Type: {agent.agent_type}")
            print(f"   Endpoints: {len(agent.endpoints)}")
        else:
            print("⚠️  Agent not yet registered in database")
            print("   It will be automatically registered on server startup")
            print("   Or run: python manage.py sync")
    finally:
        db.close()
except Exception as e:
    print(f"⚠️  Could not check database: {e}")
    print("   This is okay - agent will register on server startup")

# Summary
print("\n" + "=" * 70)
print("Setup Summary")
print("=" * 70)
print("✅ Composio installed")
print("✅ MCP installed")
print("✅ COMPOSIO_API_KEY configured")
print("✅ GMAIL_MCP_URL configured")
print("✅ MCP connection working")
print("✅ Agent definition exists")
print()
print("🎉 Gmail MCP Agent is ready to use!")
print()
print("Next steps:")
print("1. Start the backend server: python main.py")
print("2. The agent will auto-register on startup")
print("3. Try a query: 'Check my unread emails'")
print()
print("=" * 70)

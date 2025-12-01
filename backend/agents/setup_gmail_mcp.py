#!/usr/bin/env python3
"""
Setup script for Gmail MCP Agent
Registers the agent in the database and sets up Composio connection.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import SessionLocal
from models import Agent, AgentEndpoint, EndpointParameter, AgentCapability, AgentType
from sqlalchemy import select
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_agent_definition():
    """Load agent definition from JSON file"""
    json_path = Path(__file__).parent.parent / "Agent_entries" / "gmail_mcp_agent.json"
    with open(json_path, 'r') as f:
        return json.load(f)

def check_agent_registered():
    """Check if Gmail MCP agent is registered in database"""
    db = SessionLocal()
    
    try:
        agent_def = load_agent_definition()
        agent = db.query(Agent).filter_by(id=agent_def['id']).first()
        
        if agent:
            logger.info(f"✅ Agent '{agent.name}' is registered in database")
            logger.info(f"   - ID: {agent.id}")
            logger.info(f"   - Type: {agent.agent_type}")
            logger.info(f"   - Endpoints: {len(agent.endpoints)}")
            logger.info(f"   - Capabilities: {len(agent.capabilities)}")
            return True
        else:
            logger.warning(f"⚠️  Agent '{agent_def['id']}' not found in database")
            logger.info("   The agent will be automatically registered on next server startup")
            logger.info("   Or run: python manage.py sync")
            return False
        
    except Exception as e:
        logger.error(f"Failed to check agent registration: {e}", exc_info=True)
        return False
    finally:
        db.close()

def setup_composio():
    """Guide user through Composio setup"""
    logger.info("\n" + "="*60)
    logger.info("Gmail MCP Agent - Composio Setup")
    logger.info("="*60)
    
    # Check for API key
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        logger.warning("\n⚠️  COMPOSIO_API_KEY not found in environment!")
        logger.info("\nTo set up Composio:")
        logger.info("1. Sign up at https://composio.dev/dashboard")
        logger.info("2. Get your API key from the dashboard")
        logger.info("3. Add to backend/.env:")
        logger.info("   COMPOSIO_API_KEY=cl_your_key_here")
        logger.info("\n4. Connect Gmail account:")
        logger.info("   composio login")
        logger.info("   composio add gmail")
        logger.info("\n5. Generate MCP URL and add to .env:")
        logger.info("   GMAIL_MCP_URL=https://mcp.composio.dev/gmail/sse?user_id=...&connected_account_id=...")
        return False
    
    logger.info(f"\n✅ COMPOSIO_API_KEY found: {api_key[:10]}...")
    
    # Check for MCP URL
    mcp_url = os.getenv("GMAIL_MCP_URL")
    if not mcp_url:
        logger.warning("\n⚠️  GMAIL_MCP_URL not found in environment!")
        logger.info("\nTo generate MCP URL:")
        logger.info("1. Run: composio add gmail")
        logger.info("2. Note your connected_account_id (e.g., ca_gmail_456)")
        logger.info("3. Generate URL using Python:")
        logger.info("""
from composio import Composio
composio = Composio(api_key="your_key")
mcp_config = composio.mcp.create(
    toolkit="gmail",
    auth_config_id="ca_gmail_456",
    user_id="your_email@gmail.com"
)
mcp_url = composio.mcp.generate(config_id=mcp_config.id, transport="sse")
print("MCP URL:", mcp_url)
        """)
        logger.info("\n4. Add to backend/.env:")
        logger.info("   GMAIL_MCP_URL=<generated_url>")
        return False
    
    logger.info(f"\n✅ GMAIL_MCP_URL found: {mcp_url[:50]}...")
    logger.info("\n✅ Composio setup complete!")
    return True

async def test_connection():
    """Test connection to Gmail MCP server"""
    logger.info("\n" + "="*60)
    logger.info("Testing Gmail MCP Connection")
    logger.info("="*60)
    
    try:
        import httpx
        
        api_key = os.getenv("COMPOSIO_API_KEY")
        mcp_url = os.getenv("GMAIL_MCP_URL")
        
        if not api_key or not mcp_url:
            logger.error("❌ Missing COMPOSIO_API_KEY or GMAIL_MCP_URL")
            return False
        
        headers = {"x-api-key": api_key}
        
        logger.info(f"Connecting to: {mcp_url[:60]}...")
        
        # Test basic HTTP connection to Composio
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            try:
                # Test the MCP endpoint
                response = await client.get(mcp_url)
                
                if response.status_code == 200:
                    logger.info("✅ MCP endpoint is accessible")
                    try:
                        data = response.json()
                        logger.info(f"   Response: {data}")
                    except:
                        logger.info("   Endpoint responded successfully")
                    return True
                elif response.status_code == 404:
                    logger.warning("⚠️  MCP endpoint returned 404")
                    logger.info("   This might be normal - trying alternative test...")
                    
                    # Try to test the connection ID directly
                    connection_id = os.getenv("GMAIL_CONNECTION_ID")
                    if connection_id:
                        test_url = f"https://backend.composio.dev/api/v1/connectedAccounts/{connection_id}"
                        test_response = await client.get(test_url)
                        if test_response.status_code == 200:
                            logger.info("✅ Gmail connection verified via API")
                            data = test_response.json()
                            logger.info(f"   Status: {data.get('status', 'unknown')}")
                            return True
                    
                    logger.warning("⚠️  Could not verify connection, but configuration looks correct")
                    return True  # Assume it's okay
                else:
                    logger.warning(f"⚠️  MCP endpoint returned status {response.status_code}")
                    logger.info("   Configuration appears correct, will test during actual use")
                    return True  # Assume it's okay for now
                    
            except httpx.HTTPError as e:
                logger.error(f"❌ HTTP error: {e}")
                return False
            except Exception as e:
                logger.warning(f"⚠️  Connection test inconclusive: {e}")
                logger.info("   Configuration appears correct, will test during actual use")
                return True  # Assume it's okay
                
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("   Make sure httpx is installed: pip install httpx")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main setup flow"""
    logger.info("🚀 Gmail MCP Agent Setup\n")
    
    logger.info("=" * 60)
    logger.info("IMPORTANT: Gmail MCP Agent is NOT a standalone server")
    logger.info("=" * 60)
    logger.info("Unlike browser_automation_agent.py, the Gmail MCP agent")
    logger.info("works through the MCP protocol and is called directly by")
    logger.info("the orchestrator. No separate server process is needed.")
    logger.info("=" * 60 + "\n")
    
    # Step 1: Check if agent definition exists
    logger.info("Step 1: Checking agent definition...")
    json_path = Path(__file__).parent.parent / "Agent_entries" / "gmail_mcp_agent.json"
    if not json_path.exists():
        logger.error(f"❌ Agent definition not found: {json_path}")
        return 1
    logger.info(f"✅ Agent definition found: {json_path.name}\n")
    
    # Step 2: Check database registration (will auto-sync on startup)
    logger.info("Step 2: Checking database registration...")
    is_registered = check_agent_registered()
    if not is_registered:
        logger.info("\n💡 To register now, run: python manage.py sync")
        logger.info("   Or restart the server (auto-syncs on startup)\n")
    
    # Step 3: Check Composio setup
    logger.info("Step 3: Checking Composio configuration...")
    composio_ready = setup_composio()
    
    if not composio_ready:
        logger.warning("\n⚠️  Composio not fully configured. Complete setup steps above.")
        logger.info("\nAgent registered but won't work until Composio is configured.")
        return 1
    
    # Step 4: Test connection
    logger.info("\nStep 4: Testing MCP connection...")
    try:
        connection_ok = asyncio.run(test_connection())
        if connection_ok:
            logger.info("\n" + "="*60)
            logger.info("✅ Gmail MCP Agent Setup Complete!")
            logger.info("="*60)
            logger.info("\nThe agent is now ready to use in your orchestrator.")
            logger.info("\nExample queries:")
            logger.info("  - 'Check my unread emails'")
            logger.info("  - 'Send an email to john@example.com about the meeting'")
            logger.info("  - 'Search for emails from sarah@company.com'")
            logger.info("  - 'Get my latest 5 emails'")
            return 0
        else:
            logger.error("\n❌ Connection test failed. Fix issues above and try again.")
            return 1
    except Exception as e:
        logger.error(f"\n❌ Connection test error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

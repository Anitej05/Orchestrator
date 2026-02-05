
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import io
import asyncio
import os
from datetime import datetime

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from services.mcp_service import probe_mcp_url, ingest_mcp_agent, list_user_connections
from models import AgentCredential, AgentType

# Use IsolatedAsyncioTestCase if available
try:
    from unittest import IsolatedAsyncioTestCase
except ImportError:
    IsolatedAsyncioTestCase = unittest.TestCase

class TestMCPService(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_db = MagicMock()

    @patch('httpx.AsyncClient')
    async def test_probe_mcp_url(self, mock_client_cls):
        print("\n=== Testing Probe MCP URL ===")
        # Mock 200 OK
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        result = await probe_mcp_url("http://localhost:8000")
        self.assertEqual(result['status'], 'open')
        print("✅ Probe Logic Verified")

    async def test_ingest_graceful_failure(self):
        print("\n=== Testing Ingest Failure Handling ===")
        # We don't verify success path due to mocking complexities with missing SDK
        # But we verify it handles exceptions and returns error status
        
        result = await ingest_mcp_agent(
            self.mock_db, "http://bad-url", "user-1", {}
        )
        print(f"Result: {result}")
        self.assertEqual(result['status'], 'error')
        print("✅ Ingest Failure Handled")

    async def test_list_connections(self):
        print("\n=== Testing List Connections ===")
        
        # Mock Credential -> Agent
        mock_agent = MagicMock()
        mock_agent.id = "agent-1"
        mock_agent.agent_type = AgentType.MCP_HTTP
        mock_agent.name = "My Agent"
        mock_agent.connection_config = {"url": "http://foo"}
        
        mock_cred = MagicMock()
        mock_cred.agent = mock_agent
        mock_cred.created_at = datetime.now()
        
        self.mock_db.query.return_value.filter_by.return_value.all.return_value = [mock_cred]
        
        conns = await list_user_connections(self.mock_db, "user-1")
        print(f"Connections: {len(conns)}")
        self.assertEqual(len(conns), 1)
        self.assertEqual(conns[0]['name'], "My Agent")
        print("✅ List Logic Verified")

if __name__ == "__main__":
    unittest.main()

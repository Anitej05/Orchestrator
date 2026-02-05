
import unittest
from unittest.mock import MagicMock, patch, mock_open
import sys
import io
import os
import json

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from services.agent_registry_service import AgentRegistryService

class TestAgentRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistryService()
        self.mock_db = MagicMock()
        
    def test_list_active_agents(self):
        print("\n=== Testing List Active Agents ===")
        
        # Mock Agent Objects
        mock_agent = MagicMock()
        mock_agent.id = "agent-1"
        mock_agent.name = "Test Agent"
        mock_agent.endpoints = []
        
        # Mock Query Chain: db.query().options().filter().all()
        # Mock query return value
        mock_query = self.mock_db.query.return_value
        mock_options = mock_query.options.return_value
        mock_filter = mock_options.filter.return_value
        mock_filter.all.return_value = [mock_agent]
        
        result = self.registry.list_active_agents(db=self.mock_db)
        
        print(f"Result count: {len(result)}")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], "agent-1")
        print("✅ List Logic Verified")

    def test_get_request_format_fallback(self):
        print("\n=== Testing Request Format Fallback ===")
        
        # Mock Path to exist and return json
        mock_json_content = json.dumps({
            "endpoints": [
                {
                    "endpoint": "/v1/chat",
                    "request_format": "json"
                }
            ]
        })
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=mock_json_content):
            
            fmt = self.registry.get_request_format("agent-1", "/v1/chat")
            print(f"Format for /v1/chat: {fmt}")
            self.assertEqual(fmt, "json")
            
            fmt_missing = self.registry.get_request_format("agent-1", "/other")
            print(f"Format for /other: {fmt_missing}")
            self.assertIsNone(fmt_missing)
            
        print("✅ Fallback Logic Verified")

if __name__ == "__main__":
    unittest.main()

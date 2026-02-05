
import unittest
from unittest.mock import MagicMock, patch
import sys
import io

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# NOTE: This script expects PYTHONPATH to include the 'backend' directory
# e.g. set PYTHONPATH=path/to/backend
from models import AgentCredential
from services.credential_service import (
    save_agent_credentials,
    get_agent_credentials,
    get_credentials_for_headers
)

class TestCredentialService(unittest.TestCase):
    def setUp(self):
        # Mock DB Session
        self.mock_db = MagicMock()
        self.agent_id = "test-agent-1"
        self.user_id = "user-123"
        
    def test_encrypt_and_save(self):
        print("\n=== Testing Encryption & Save ===")
        creds = {"api_key": "secret_abc", "app_id": "12345"}
        
        # Mock query returning None (no existing creds)
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = None
        
        success = save_agent_credentials(self.mock_db, self.agent_id, self.user_id, creds)
        
        self.assertTrue(success)
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        
        # Verify call arguments
        # args[0] is the new credential object
        new_cred = self.mock_db.add.call_args[0][0]
        self.assertEqual(new_cred.agent_id, self.agent_id)
        # Verify it's NOT plain text
        self.assertNotEqual(new_cred.encrypted_credentials['api_key'], "secret_abc")
        print("✅ Save logic verified (Encryption applied)")

    def test_decrypt_and_retrieve(self):
        print("\n=== Testing Decrypt & Retrieve ===")
        # Create a real credential object with encrypted data
        from utils.encryption import encrypt
        encrypted_data = {
            "api_key": encrypt("secret_abc"),
            "region": encrypt("us-east-1")
        }
        
        mock_cred = AgentCredential(
            agent_id=self.agent_id,
            user_id=self.user_id,
            encrypted_credentials=encrypted_data,
            is_active=True
        )
        
        self.mock_db.query.return_value.filter_by.return_value.first.return_value = mock_cred
        
        result = get_agent_credentials(self.mock_db, self.agent_id, self.user_id)
        
        self.assertEqual(result['api_key'], "secret_abc")
        self.assertEqual(result['region'], "us-east-1")
        print(f"✅ Decrytion Verified: {result}")

    def test_headers_formatting(self):
        print("\n=== Testing Header Formatting ===")
        
        # Patch get_agent_credentials to avoid needing complex db mocks just for this test
        # We assume get_agent_credentials works based on previous test
        with patch('services.credential_service.get_agent_credentials') as mock_get:
            mock_get.return_value = {"api_key": "token_123", "custom-header": "custom_val"}
            
            # Test REST format
            headers_rest = get_credentials_for_headers(self.mock_db, self.agent_id, self.user_id, "http_rest")
            self.assertEqual(headers_rest.get('Authorization'), "Bearer token_123")
            self.assertEqual(headers_rest.get('custom-header'), "custom_val")
            print("✅ REST Headers OK")
            
            # Test MCP format
            headers_mcp = get_credentials_for_headers(self.mock_db, self.agent_id, self.user_id, "mcp_http")
            # header logic: if 'api_key' present, REST uses Authorization, MCP uses x-api-key?
            # Let's check logic:
            # if agent_type == "mcp_http":
            #   if 'api_key' in credentials: headers['x-api-key'] = credentials['api_key']
            self.assertEqual(headers_mcp.get('x-api-key'), "token_123")
            print("✅ MCP Headers OK")

if __name__ == "__main__":
    unittest.main()

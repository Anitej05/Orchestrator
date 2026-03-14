"""
Mail Agent Integration Test Fixtures
"""
import pytest
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

@pytest.fixture(scope="session")
def test_user_id() -> str:
    """Test user ID for Mail agent tests"""
    return os.getenv("TEST_USER_ID", "test_user_mail_ci")

@pytest.fixture(scope="session")
def composio_api_key() -> str:
    """Composio API key from environment"""
    api_key = os.getenv("COMPOSIO_API_KEY")
    if not api_key:
        pytest.skip("COMPOSIO_API_KEY not set")
    return api_key

@pytest.fixture
def test_email_address() -> str:
    """Test email address for sending tests"""
    return os.getenv("TEST_EMAIL_ADDRESS", "test@example.com")

@pytest.fixture
def mail_service(test_user_id: str):
    """Create Mail service instance for testing"""
    from agents.mail_agent.service import GmailService

    try:
        service = GmailService(test_user_id)
        return service
    except ValueError as e:
        pytest.skip(f"Gmail not connected for test user: {e}")

@pytest.fixture
def sample_search_query() -> str:
    """Sample search query for testing"""
    return "label:inbox"

@pytest.fixture
def sample_draft_data() -> dict:
    """Sample draft data for testing"""
    return {
        "to": "test@example.com",
        "subject": "Test Draft",
        "body": "This is a test draft created by integration tests."
    }

"""
General Agent Integration Tests

Tests the General Fallback Agent with real Composio connections.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from agents.general_agent.tool_cache import ToolCache


class TestGeneralAgentService:
    """Test GeneralAgentService functionality"""
    
    @pytest.fixture
    def service(self):
        """Create service with mocked dependencies"""
        tool_cache = ToolCache(ttl_seconds=60)
        
        # Mock Composio imports before importing service
        with patch('services.integrations.composio_tools.ComposioToolManager') as mock_composio, \
             patch('services.integrations.composio_auth.ComposioAuthManager') as mock_auth:
            
            # Import here to avoid import errors
            from agents.general_agent.service import GeneralAgentService
            
            service = GeneralAgentService(user_id="test_user_123", tool_cache=tool_cache)
            
            # Replace with mocks
            service.composio = MagicMock()
            service.auth_service = MagicMock()
            
            return service
    
    def test_extract_app_from_prompt_slack(self, service):
        """Test app extraction for Slack"""
        app = service._extract_app_from_prompt("Send a Slack message to #general")
        assert app == "Slack"
    
    def test_extract_app_from_prompt_notion(self, service):
        """Test app extraction for Notion"""
        app = service._extract_app_from_prompt("Create a Notion page titled 'Notes'")
        assert app == "Notion"
    
    def test_extract_app_from_prompt_github(self, service):
        """Test app extraction for GitHub"""
        app = service._extract_app_from_prompt("Open a GitHub PR for feature branch")
        assert app == "Github"
    
    def test_extract_app_from_prompt_unknown(self, service):
        """Test app extraction for unknown app"""
        app = service._extract_app_from_prompt("Do something generic")
        assert app is None
    
    @pytest.mark.asyncio
    async def test_execute_no_connection(self, service):
        """Test execution when user has no connection"""
        with patch.object(service, '_check_connection', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {"connected": False}
            
            result = await service.execute(
                prompt="Send a Slack message",
                payload={"user_id": "test_user_123"}
            )
            
            assert result["success"] is False
            assert result["status"] == "needs_input"
            assert "connect" in result["question"].lower()
            assert "Slack" in result["error"]
    
    @pytest.mark.asyncio
    async def test_execute_with_connection(self, service):
        """Test execution when user has connection"""
        mock_tools = [Mock(name="SLACK_SEND_MESSAGE")]
        
        with patch.object(service, '_check_connection', new_callable=AsyncMock) as mock_check, \
             patch.object(service, '_get_tools_for_app', new_callable=AsyncMock) as mock_get_tools, \
             patch.object(service, '_execute_with_tools', new_callable=AsyncMock) as mock_execute:
            
            mock_check.return_value = {"connected": True}
            mock_get_tools.return_value = mock_tools
            mock_execute.return_value = {
                "success": True,
                "result": {"message": "sent"},
                "status": "completed"
            }
            
            result = await service.execute(
                prompt="Send a Slack message to #general",
                payload={"user_id": "test_user_123"}
            )
            
            assert result["success"] is True
            assert result["status"] == "completed"
            assert result["result"]["message"] == "sent"
    
    @pytest.mark.asyncio
    async def test_execute_unknown_app(self, service):
        """Test execution with undetectable app"""
        result = await service.execute(
            prompt="Do something vague",
            payload={"user_id": "test_user_123"}
        )
        
        assert result["success"] is False
        assert result["status"] == "error"
        assert "could not determine" in result["error"].lower()
    
    def test_select_tool_heuristic_slack(self, service):
        """Test tool selection heuristic for Slack"""
        tools = [Mock(name="SLACK_SEND_MESSAGE")]
        tool = service._select_tool_heuristic(
            prompt="Send a message to #general",
            tools=tools,
            app_name="Slack"
        )
        assert tool == "SLACK_SEND_MESSAGE"
    
    def test_select_tool_heuristic_notion(self, service):
        """Test tool selection heuristic for Notion"""
        tools = [Mock(name="NOTION_CREATE_PAGE")]
        tool = service._select_tool_heuristic(
            prompt="Create a page in Notion",
            tools=tools,
            app_name="Notion"
        )
        assert tool == "NOTION_CREATE_PAGE"
    
    def test_extract_parameters_channel(self, service):
        """Test parameter extraction for channel"""
        params = service._extract_parameters(
            prompt="Send to #general saying hello",
            payload={}
        )
        assert "channel" in params
        assert params["channel"] == "general"
    
    def test_extract_parameters_message(self, service):
        """Test parameter extraction for message"""
        params = service._extract_parameters(
            prompt='Send message "Hello, world!"',
            payload={}
        )
        assert "text" in params
        assert params["text"] == "Hello, world!"


class TestToolCache:
    """Test ToolCache functionality"""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance"""
        return ToolCache(ttl_seconds=2)  # Short TTL for testing
    
    def test_cache_miss(self, cache):
        """Test cache miss"""
        result = cache.get("user_123", "Slack")
        assert result is None
    
    def test_cache_hit(self, cache):
        """Test cache hit"""
        tools = [Mock(), Mock()]
        cache.set("user_123", "Slack", tools)
        
        result = cache.get("user_123", "Slack")
        assert result == tools
    
    def test_cache_expiry(self, cache):
        """Test cache expiration"""
        tools = [Mock()]
        cache.set("user_123", "Slack", tools)
        
        # Wait for expiry
        import time
        time.sleep(3)
        
        result = cache.get("user_123", "Slack")
        assert result is None
    
    def test_invalidate_user(self, cache):
        """Test user cache invalidation"""
        cache.set("user_123", "Slack", [Mock()])
        cache.set("user_123", "Notion", [Mock()])
        
        cache.invalidate_user("user_123")
        
        assert cache.get("user_123", "Slack") is None
        assert cache.get("user_123", "Notion") is None
    
    def test_invalidate_app(self, cache):
        """Test app cache invalidation"""
        cache.set("user_123", "Slack", [Mock()])
        cache.set("user_123", "Notion", [Mock()])
        
        cache.invalidate_app("user_123", "Slack")
        
        assert cache.get("user_123", "Slack") is None
        assert cache.get("user_123", "Notion") is not None
    
    def test_cache_stats(self, cache):
        """Test cache statistics"""
        cache.set("user_123", "Slack", [Mock()])
        cache.get("user_123", "Slack")  # Hit
        cache.get("user_123", "Notion")  # Miss
        
        stats = cache.get_stats()
        assert stats["total_users"] == 1
        assert stats["total_entries"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_percent"] == 50.0
    
    def test_cleanup_expired(self, cache):
        """Test expired entry cleanup"""
        cache.set("user_123", "Slack", [Mock()])
        
        import time
        time.sleep(3)  # Wait for expiry
        
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.get_stats()["total_entries"] == 0


class TestIntegration:
    """Integration tests with FastAPI"""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client"""
        from fastapi.testclient import TestClient
        from agents.general_agent.agent import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test / endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "General Fallback Agent" in response.json()["agent"]
    
    def test_execute_missing_user_id(self, client):
        """Test /execute without user_id"""
        response = client.post("/execute", json={
            "prompt": "Send a Slack message"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "user_id" in data["error"]
    
    def test_cache_stats_endpoint(self, client):
        """Test /cache/stats endpoint"""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        assert "total_users" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

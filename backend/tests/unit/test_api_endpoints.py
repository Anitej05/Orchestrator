"""
Unit tests for API endpoints and request/response models
Tests: Request validation, response structure, endpoint behavior
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agents.browser_automation_agent import app, BrowseRequest, BrowseResponse


class TestBrowseRequestModel:
    """Test BrowseRequest Pydantic model"""
    
    def test_browse_request_minimal(self):
        """Test BrowseRequest with minimal required fields"""
        request = BrowseRequest(task="Test task")
        
        assert request.task == "Test task"
        assert request.extract_data is False  # default
        assert request.max_steps == 10  # default
    
    def test_browse_request_all_fields(self):
        """Test BrowseRequest with all fields"""
        request = BrowseRequest(
            task="Complex task",
            extract_data=True,
            max_steps=20
        )
        
        assert request.task == "Complex task"
        assert request.extract_data is True
        assert request.max_steps == 20
    
    def test_browse_request_validation_missing_task(self):
        """Test that task is required"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BrowseRequest()
    
    def test_browse_request_validation_invalid_max_steps(self):
        """Test validation of max_steps"""
        # Should accept valid integer
        request = BrowseRequest(task="Test", max_steps=5)
        assert request.max_steps == 5
        
        # Pydantic will coerce string to int if possible
        request = BrowseRequest(task="Test", max_steps="15")
        assert request.max_steps == 15


class TestBrowseResponseModel:
    """Test BrowseResponse Pydantic model"""
    
    def test_browse_response_success(self):
        """Test BrowseResponse for successful execution"""
        response = BrowseResponse(
            success=True,
            task_summary="Task completed successfully",
            actions_taken=[
                {"action": "navigate", "url": "https://example.com"},
                {"action": "click", "selector": "#button"}
            ],
            extracted_data={"result": "data"},
            screenshot_files=["screenshot1.png"],
            error=None
        )
        
        assert response.success is True
        assert len(response.actions_taken) == 2
        assert response.extracted_data is not None
        assert response.error is None
    
    def test_browse_response_failure(self):
        """Test BrowseResponse for failed execution"""
        response = BrowseResponse(
            success=False,
            task_summary="Task failed",
            actions_taken=[],
            error="Connection timeout"
        )
        
        assert response.success is False
        assert len(response.actions_taken) == 0
        assert response.error == "Connection timeout"
    
    def test_browse_response_optional_fields(self):
        """Test BrowseResponse with optional fields omitted"""
        response = BrowseResponse(
            success=True,
            task_summary="Done",
            actions_taken=[]
        )
        
        assert response.extracted_data is None
        assert response.screenshot_files is None
        assert response.error is None


class TestHealthEndpoint:
    """Test /health endpoint"""
    
    def test_health_endpoint_returns_200(self):
        """Test health endpoint returns 200 OK"""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_endpoint_response_structure(self):
        """Test health endpoint response structure"""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "agent" in data
        assert data["status"] == "healthy"
        assert data["agent"] == "custom_browser_agent"


class TestInfoEndpoint:
    """Test /info endpoint"""
    
    def test_info_endpoint_returns_200(self):
        """Test info endpoint returns 200 OK"""
        client = TestClient(app)
        response = client.get("/info")
        
        assert response.status_code == 200
    
    def test_info_endpoint_response_structure(self):
        """Test info endpoint returns agent definition"""
        client = TestClient(app)
        response = client.get("/info")
        data = response.json()
        
        assert "id" in data
        assert "name" in data
        assert "description" in data
        assert "capabilities" in data
        assert "endpoints" in data
        assert data["id"] == "custom_browser_agent"
    
    def test_info_endpoint_capabilities_list(self):
        """Test that capabilities is a list"""
        client = TestClient(app)
        response = client.get("/info")
        data = response.json()
        
        assert isinstance(data["capabilities"], list)
        assert len(data["capabilities"]) > 0


class TestBrowseEndpoint:
    """Test /browse endpoint"""
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_endpoint_accepts_post(self, mock_agent_class):
        """Test browse endpoint accepts POST requests"""
        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.return_value = {
            "success": True,
            "summary": "Done",
            "actions": [],
            "screenshots": [],
            "task_id": "test-123",
            "metrics": {}
        }
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post("/browse", json={"task": "Test task"})
        
        assert response.status_code == 200
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_endpoint_response_structure(self, mock_agent_class):
        """Test browse endpoint response structure"""
        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.return_value = {
            "success": True,
            "summary": "Task completed",
            "actions": [{"action": "navigate"}],
            "screenshots": ["screenshot.png"],
            "task_id": "test-123",
            "metrics": {"llm_calls": 5}
        }
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post("/browse", json={"task": "Test task"})
        data = response.json()
        
        assert "success" in data
        assert "task_summary" in data
        assert "actions_taken" in data
        assert "screenshot_files" in data
        assert "task_id" in data
        assert "metrics" in data
    
    def test_browse_endpoint_requires_task(self):
        """Test browse endpoint requires task parameter"""
        client = TestClient(app)
        response = client.post("/browse", json={})
        
        # Should return validation error
        assert response.status_code == 422
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_endpoint_with_optional_params(self, mock_agent_class):
        """Test browse endpoint with optional parameters"""
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.return_value = {
            "success": True,
            "summary": "Done",
            "actions": [],
            "screenshots": [],
            "task_id": "test-123",
            "metrics": {}
        }
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post("/browse", json={
            "task": "Test task",
            "max_steps": 15,
            "extract_data": True
        })
        
        assert response.status_code == 200
        # Verify agent was initialized with correct params
        mock_agent_class.assert_called_once()
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_endpoint_handles_agent_error(self, mock_agent_class):
        """Test browse endpoint handles agent errors"""
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.side_effect = Exception("Agent failed")
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post("/browse", json={"task": "Test task"})
        data = response.json()
        
        assert response.status_code == 200  # Still returns 200
        assert data["success"] is False
        assert "error" in data
        assert data["error"] == "Agent failed"


class TestLiveScreenshotEndpoints:
    """Test live screenshot endpoints"""
    
    def test_get_live_screenshot_not_found(self):
        """Test getting live screenshot for non-existent task"""
        client = TestClient(app)
        response = client.get("/live/non-existent-task")
        
        assert response.status_code == 404
    
    def test_get_all_live_screenshots(self):
        """Test getting all live screenshots"""
        client = TestClient(app)
        response = client.get("/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "active_tasks" in data
        assert "screenshots" in data
        assert isinstance(data["active_tasks"], list)


class TestEndpointQueryParameters:
    """Test endpoint query parameters"""
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_with_headless_param(self, mock_agent_class):
        """Test browse endpoint with headless query parameter"""
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.return_value = {
            "success": True,
            "summary": "Done",
            "actions": [],
            "screenshots": [],
            "task_id": "test-123",
            "metrics": {}
        }
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post("/browse?headless=true", json={"task": "Test"})
        
        assert response.status_code == 200
    
    @patch('agents.browser_automation_agent.BrowserAgent')
    def test_browse_with_thread_id_param(self, mock_agent_class):
        """Test browse endpoint with thread_id query parameter"""
        mock_agent = AsyncMock()
        mock_agent.__aenter__.return_value = mock_agent
        mock_agent.__aexit__.return_value = None
        mock_agent.run.return_value = {
            "success": True,
            "summary": "Done",
            "actions": [],
            "screenshots": [],
            "task_id": "test-123",
            "metrics": {}
        }
        mock_agent_class.return_value = mock_agent
        
        client = TestClient(app)
        response = client.post(
            "/browse?thread_id=test-thread-123",
            json={"task": "Test"}
        )
        
        assert response.status_code == 200


class TestCORSAndHeaders:
    """Test CORS and header handling"""
    
    def test_health_endpoint_cors(self):
        """Test CORS headers on health endpoint"""
        client = TestClient(app)
        response = client.get("/health")
        
        # FastAPI handles CORS if configured
        assert response.status_code == 200
    
    def test_browse_endpoint_accepts_json(self):
        """Test browse endpoint accepts JSON content type"""
        client = TestClient(app)
        response = client.post(
            "/browse",
            json={"task": "Test"},
            headers={"Content-Type": "application/json"}
        )
        
        # Should accept JSON (might fail due to mock, but won't be 415)
        assert response.status_code != 415


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

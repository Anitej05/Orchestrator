"""
Pytest configuration and shared fixtures for unit tests
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def mock_env_cerebras():
    """Mock environment with Cerebras API key"""
    with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_cerebras_key'}):
        yield


@pytest.fixture
def mock_env_groq():
    """Mock environment with Groq API key"""
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test_groq_key'}):
        yield


@pytest.fixture
def mock_env_nvidia():
    """Mock environment with NVIDIA API key"""
    with patch.dict(os.environ, {'NVIDIA_API_KEY': 'test_nvidia_key'}):
        yield


@pytest.fixture
def mock_env_ollama():
    """Mock environment with Ollama API key"""
    with patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_ollama_key'}):
        yield


@pytest.fixture
def mock_env_all_providers():
    """Mock environment with all provider API keys"""
    with patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras_key',
        'GROQ_API_KEY': 'test_groq_key',
        'NVIDIA_API_KEY': 'test_nvidia_key',
        'OLLAMA_API_KEY': 'test_ollama_key'
    }):
        yield


@pytest.fixture
def mock_env_no_providers():
    """Mock environment with no provider API keys"""
    with patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    client = Mock()
    response = Mock()
    response.choices = [Mock(message=Mock(content="Test response"))]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_vision_client():
    """Mock vision client"""
    client = Mock()
    response = Mock()
    response.choices = [Mock(message=Mock(content='{"action": "click", "selector": "#button"}'))]
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_playwright_page():
    """Mock Playwright page object"""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title.return_value = "Example Page"
    page.screenshot.return_value = b'fake_screenshot_data'
    page.evaluate.return_value = {
        "interactive_elements": [],
        "forms": [],
        "images": []
    }
    return page


@pytest.fixture
def mock_playwright_browser():
    """Mock Playwright browser object"""
    browser = AsyncMock()
    context = AsyncMock()
    page = AsyncMock()
    
    browser.new_context.return_value = context
    context.new_page.return_value = page
    
    page.url = "https://example.com"
    page.title.return_value = "Example Page"
    page.screenshot.return_value = b'fake_screenshot_data'
    
    return browser, context, page


@pytest.fixture
def sample_page_content():
    """Sample page content for testing"""
    return {
        "url": "https://example.com",
        "title": "Example Page",
        "interactive_elements": [
            {"type": "button", "selector": "#btn1", "text": "Click me"},
            {"type": "link", "selector": "a.nav", "text": "Home", "href": "/home"},
            {"type": "input", "selector": "#email", "input_type": "email"}
        ],
        "forms": [
            {
                "selector": "#login-form",
                "action": "/login",
                "method": "POST",
                "fields": [
                    {"name": "username", "type": "text"},
                    {"name": "password", "type": "password"}
                ]
            }
        ],
        "images": [
            {"src": "image1.jpg", "alt": "Image 1"},
            {"src": "image2.jpg", "alt": "Image 2"}
        ]
    }


@pytest.fixture
def sample_action_plan():
    """Sample action plan for testing"""
    return {
        "action": "click",
        "selector": "#button",
        "reasoning": "Need to click the button to proceed",
        "confidence": 0.9
    }


@pytest.fixture
def sample_task_plan():
    """Sample task plan for testing"""
    return [
        {"subtask": "Navigate to website", "status": "pending"},
        {"subtask": "Find search box", "status": "pending"},
        {"subtask": "Enter search query", "status": "pending"},
        {"subtask": "Click search button", "status": "pending"}
    ]


@pytest.fixture
def sample_extracted_data():
    """Sample extracted data for testing"""
    return {
        "entries": [
            {"name": "Product 1", "price": "$10.99", "rating": "4.5"},
            {"name": "Product 2", "price": "$20.99", "rating": "4.8"},
            {"name": "Product 3", "price": "$15.99", "rating": "4.2"}
        ],
        "total_entries": 3,
        "extraction_method": "vision"
    }


@pytest.fixture
def sample_metrics():
    """Sample metrics for testing"""
    return {
        "start_time": 1234567890.0,
        "llm_calls": 5,
        "vision_calls": 2,
        "actions_planned": 10,
        "actions_succeeded": 8,
        "actions_failed": 2,
        "screenshots_captured": 10
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add asyncio marker to async tests
        if 'asyncio' in item.keywords:
            item.add_marker(pytest.mark.asyncio)


# Cleanup after tests
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any cleanup code here if needed

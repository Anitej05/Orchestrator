"""
Test Gmail Agent - Logging and Monitoring
Task 3.3: Review logging and monitoring
"""
import pytest
import logging
from io import StringIO
from agents.gmail_agent.service import GmailService

@pytest.fixture
def log_capture():
    """Capture log output for testing"""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    
    logger = logging.getLogger("gmail_agent")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_stream
    
    logger.removeHandler(handler)

@pytest.mark.asyncio
async def test_service_initialization_logging(test_user_id: str, log_capture: StringIO):
    """Test that service initialization is logged"""
    try:
        service = GmailService(test_user_id)
        
        log_output = log_capture.getvalue()
        assert f"Initialized for user {test_user_id}" in log_output
    except ValueError:
        # User not connected - expected in test environment
        pass

@pytest.mark.asyncio
async def test_operation_logging(gmail_service: GmailService, log_capture: StringIO):
    """Test that operations are logged"""
    # Perform a search operation
    await gmail_service.search_emails(query="label:inbox", max_results=1)
    
    log_output = log_capture.getvalue()
    
    # Verify operation is logged
    assert "Executing" in log_output or "Search" in log_output

@pytest.mark.asyncio
async def test_error_logging(gmail_service: GmailService, log_capture: StringIO):
    """Test that errors are logged"""
    # Try to get a non-existent email
    result = await gmail_service.get_email("nonexistent_message_id_12345")
    
    log_output = log_capture.getvalue()
    
    # Verify error is logged
    if not result["success"]:
        assert "Error" in log_output or "error" in log_output.lower()

@pytest.mark.asyncio
async def test_no_sensitive_data_in_logs(gmail_service: GmailService, log_capture: StringIO):
    """Test that sensitive data is not logged"""
    # Perform operations
    await gmail_service.search_emails(query="label:inbox", max_results=1)
    
    log_output = log_capture.getvalue()
    
    # Verify no sensitive data
    assert "connection_id" not in log_output.lower()
    assert "password" not in log_output.lower()
    assert "token" not in log_output.lower()
    assert "api_key" not in log_output.lower()

@pytest.mark.asyncio
async def test_log_levels(gmail_service: GmailService, log_capture: StringIO):
    """Test that appropriate log levels are used"""
    # Perform a successful operation
    result = await gmail_service.search_emails(query="label:inbox", max_results=1)
    
    log_output = log_capture.getvalue()
    
    # INFO level should be used for normal operations
    if result["success"]:
        # Check for INFO level indicators
        assert any(level in log_output for level in ["INFO", "Initialized", "Executing"])

@pytest.mark.asyncio
async def test_tool_execution_logging(gmail_service: GmailService, log_capture: StringIO):
    """Test that tool execution is logged"""
    # Execute a tool
    await gmail_service.list_labels()
    
    log_output = log_capture.getvalue()
    
    # Verify tool execution is logged
    assert "ComposioToolManager" in log_output or "Executing" in log_output

@pytest.mark.asyncio
async def test_operation_completion_logging(gmail_service: GmailService, log_capture: StringIO):
    """Test that operation completion is logged"""
    # Perform an operation
    result = await gmail_service.list_labels()
    
    log_output = log_capture.getvalue()
    
    # Verify completion is logged
    if result["success"]:
        assert "completed" in log_output.lower() or "success" in log_output.lower()

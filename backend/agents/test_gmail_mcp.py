"""
Unit tests for Gmail MCP Agent
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock
from gmail_mcp_agent import GmailMCPClient, AGENT_DEFINITION

# Test agent definition
def test_agent_definition():
    """Test that agent definition is properly structured"""
    assert AGENT_DEFINITION['id'] == 'gmail_mcp_agent'
    assert AGENT_DEFINITION['agent_type'] == 'mcp_http'
    assert len(AGENT_DEFINITION['endpoints']) == 8
    assert 'read emails' in AGENT_DEFINITION['capabilities']
    assert 'send emails' in AGENT_DEFINITION['capabilities']

def test_agent_endpoints():
    """Test that all required endpoints are defined"""
    endpoint_names = [ep['endpoint'] for ep in AGENT_DEFINITION['endpoints']]
    
    required_endpoints = [
        'GMAIL_FETCH_EMAILS',
        'GMAIL_SEND_EMAIL',
        'GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID',
        'GMAIL_LIST_THREADS',
        'GMAIL_CREATE_DRAFT',
        'GMAIL_ADD_LABEL',
        'GMAIL_DELETE_MESSAGE',
        'GMAIL_GET_ATTACHMENT'
    ]
    
    for endpoint in required_endpoints:
        assert endpoint in endpoint_names, f"Missing endpoint: {endpoint}"

def test_endpoint_parameters():
    """Test that endpoints have required parameters"""
    # Test GMAIL_FETCH_EMAILS
    fetch_endpoint = next(
        ep for ep in AGENT_DEFINITION['endpoints'] 
        if ep['endpoint'] == 'GMAIL_FETCH_EMAILS'
    )
    param_names = [p['name'] for p in fetch_endpoint['parameters']]
    assert 'query' in param_names
    assert 'max_results' in param_names
    
    # Test GMAIL_SEND_EMAIL
    send_endpoint = next(
        ep for ep in AGENT_DEFINITION['endpoints'] 
        if ep['endpoint'] == 'GMAIL_SEND_EMAIL'
    )
    param_names = [p['name'] for p in send_endpoint['parameters']]
    assert 'to' in param_names
    assert 'subject' in param_names
    assert 'body' in param_names

# Test MCP client
@pytest.mark.asyncio
async def test_gmail_client_initialization():
    """Test that Gmail MCP client initializes correctly"""
    client = GmailMCPClient()
    assert client.mcp_url is not None
    assert client.api_key is not None or os.getenv("COMPOSIO_API_KEY") is None

@pytest.mark.asyncio
@patch('gmail_mcp_agent.ClientSession')
@patch('gmail_mcp_agent.mcp.client.sse.sse_client')
async def test_call_tool_success(mock_sse_client, mock_session):
    """Test successful tool call"""
    # Mock MCP response
    mock_result = Mock()
    mock_result.content = [Mock(text="Test email content")]
    
    mock_session_instance = AsyncMock()
    mock_session_instance.initialize = AsyncMock()
    mock_session_instance.call_tool = AsyncMock(return_value=mock_result)
    mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session.return_value.__aexit__ = AsyncMock()
    
    mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=(Mock(), Mock()))
    mock_sse_client.return_value.__aexit__ = AsyncMock()
    
    # Test
    client = GmailMCPClient()
    result = await client.call_tool(
        "GMAIL_FETCH_EMAILS",
        {"query": "is:unread", "max_results": 5}
    )
    
    assert result['success'] == True
    assert 'data' in result

@pytest.mark.asyncio
@patch('gmail_mcp_agent.ClientSession')
@patch('gmail_mcp_agent.mcp.client.sse.sse_client')
async def test_call_tool_error(mock_sse_client, mock_session):
    """Test tool call error handling"""
    # Mock error
    mock_sse_client.side_effect = Exception("Connection failed")
    
    # Test
    client = GmailMCPClient()
    result = await client.call_tool(
        "GMAIL_FETCH_EMAILS",
        {"query": "is:unread"}
    )
    
    assert result['success'] == False
    assert 'error' in result

# Integration tests (require actual Composio setup)
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("COMPOSIO_API_KEY") or not os.getenv("GMAIL_MCP_URL"),
    reason="Composio not configured"
)
@pytest.mark.asyncio
async def test_real_connection():
    """Test real connection to Composio MCP server"""
    client = GmailMCPClient()
    
    # Try to fetch 1 email
    result = await client.call_tool(
        "GMAIL_FETCH_EMAILS",
        {"query": "is:unread", "max_results": 1}
    )
    
    # Should succeed or fail gracefully
    assert 'success' in result
    if not result['success']:
        assert 'error' in result

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("COMPOSIO_API_KEY") or not os.getenv("GMAIL_MCP_URL"),
    reason="Composio not configured"
)
@pytest.mark.asyncio
async def test_list_threads():
    """Test listing email threads"""
    client = GmailMCPClient()
    
    result = await client.call_tool(
        "GMAIL_LIST_THREADS",
        {"max_results": 5}
    )
    
    assert 'success' in result

# Test parameter validation
def test_fetch_emails_parameters():
    """Test GMAIL_FETCH_EMAILS parameter definitions"""
    endpoint = next(
        ep for ep in AGENT_DEFINITION['endpoints']
        if ep['endpoint'] == 'GMAIL_FETCH_EMAILS'
    )
    
    # Check required parameters
    query_param = next(p for p in endpoint['parameters'] if p['name'] == 'query')
    assert query_param['required'] == True
    assert query_param['param_type'] == 'string'
    
    # Check optional parameters
    max_results_param = next(p for p in endpoint['parameters'] if p['name'] == 'max_results')
    assert max_results_param['required'] == False
    assert max_results_param['param_type'] == 'integer'

def test_send_email_parameters():
    """Test GMAIL_SEND_EMAIL parameter definitions"""
    endpoint = next(
        ep for ep in AGENT_DEFINITION['endpoints']
        if ep['endpoint'] == 'GMAIL_SEND_EMAIL'
    )
    
    # Check required parameters
    required_params = ['to', 'subject', 'body']
    for param_name in required_params:
        param = next(p for p in endpoint['parameters'] if p['name'] == param_name)
        assert param['required'] == True
    
    # Check optional parameters
    optional_params = ['cc', 'bcc', 'user_id', 'thread_id']
    for param_name in optional_params:
        param = next(p for p in endpoint['parameters'] if p['name'] == param_name)
        assert param['required'] == False

# Test capabilities
def test_capabilities_coverage():
    """Test that capabilities cover common Gmail operations"""
    capabilities = AGENT_DEFINITION['capabilities']
    
    # Check for key capabilities
    assert any('read' in cap.lower() for cap in capabilities)
    assert any('send' in cap.lower() for cap in capabilities)
    assert any('search' in cap.lower() for cap in capabilities)
    assert any('label' in cap.lower() for cap in capabilities)
    assert any('draft' in cap.lower() for cap in capabilities)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

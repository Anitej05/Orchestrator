"""
Test Gmail Agent - Search Emails Functionality
Task 3.2.1: Test search emails endpoint
"""
import pytest
from agents.gmail_agent.service import GmailService

@pytest.mark.asyncio
async def test_search_emails_basic(gmail_service: GmailService, sample_search_query: str):
    """Test basic email search functionality"""
    result = await gmail_service.search_emails(
        query=sample_search_query,
        max_results=5,
        include_payload=False
    )
    
    assert result["success"] is True
    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert "total_count" in result
    assert result["total_count"] >= 0

@pytest.mark.asyncio
async def test_search_emails_with_payload(gmail_service: GmailService):
    """Test email search with full payload"""
    result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=1,
        include_payload=True
    )
    
    assert result["success"] is True
    if result["total_count"] > 0:
        message = result["messages"][0]
        assert "id" in message

@pytest.mark.asyncio
async def test_search_emails_max_results(gmail_service: GmailService):
    """Test max_results parameter"""
    max_results = 3
    result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=max_results,
        include_payload=False
    )
    
    assert result["success"] is True
    assert len(result["messages"]) <= max_results

@pytest.mark.asyncio
async def test_search_emails_no_results(gmail_service: GmailService):
    """Test search with query that returns no results"""
    result = await gmail_service.search_emails(
        query="from:nonexistent@example.com subject:impossible",
        max_results=10,
        include_payload=False
    )
    
    assert result["success"] is True
    assert result["total_count"] == 0
    assert result["messages"] == []

@pytest.mark.asyncio
async def test_search_emails_llm_optimization(gmail_service: GmailService):
    """Test search with LLM query optimization"""
    result = await gmail_service.search_emails(
        query="emails from last week about meetings",
        max_results=5,
        use_llm_optimization=True
    )
    
    # Should succeed even if LLM optimization fails
    assert result["success"] is True
    assert "query_used" in result

@pytest.mark.asyncio
async def test_get_email_by_id(gmail_service: GmailService):
    """Test fetching a single email by ID"""
    # First search for an email
    search_result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=1
    )
    
    if search_result["total_count"] > 0:
        message_id = search_result["messages"][0]["id"]
        
        # Get the email
        result = await gmail_service.get_email(message_id)
        
        assert result["success"] is True
        assert "message" in result
        assert result["message"]["id"] == message_id

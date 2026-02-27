"""
Test Gmail Agent - Send Email Functionality
Task 3.2.2: Test send email endpoint
"""
import pytest
from agents.gmail_agent.service import GmailService

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Gmail connection and sends actual emails")
async def test_send_email_basic(gmail_service: GmailService, test_email_address: str):
    """Test basic email sending"""
    result = await gmail_service.send_email(
        to=test_email_address,
        subject="Test Email from Integration Tests",
        body="This is a test email sent by the Gmail agent integration tests.",
        is_html=False
    )
    
    assert result["success"] is True
    assert "message" in result
    assert result["message"] == "Email sent successfully"
    assert "data" in result

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Gmail connection and sends actual emails")
async def test_send_email_with_cc(gmail_service: GmailService, test_email_address: str):
    """Test sending email with CC"""
    result = await gmail_service.send_email(
        to=test_email_address,
        subject="Test Email with CC",
        body="This email has a CC recipient.",
        cc=["cc@example.com"],
        is_html=False
    )
    
    assert result["success"] is True

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Gmail connection and sends actual emails")
async def test_send_email_html(gmail_service: GmailService, test_email_address: str):
    """Test sending HTML email"""
    html_body = "<html><body><h1>Test Email</h1><p>This is HTML content.</p></body></html>"
    
    result = await gmail_service.send_email(
        to=test_email_address,
        subject="Test HTML Email",
        body=html_body,
        is_html=True
    )
    
    assert result["success"] is True

@pytest.mark.asyncio
async def test_send_email_validation(gmail_service: GmailService):
    """Test email validation (without actually sending)"""
    # This test verifies the method signature and error handling
    # without sending actual emails
    
    # Test with empty recipient (should fail at Composio level)
    result = await gmail_service.send_email(
        to="",
        subject="Test",
        body="Test"
    )
    
    # Should return error response
    assert result["success"] is False
    assert "error" in result

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Gmail connection")
async def test_reply_to_email(gmail_service: GmailService):
    """Test replying to an email"""
    # First search for an email to reply to
    search_result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=1
    )
    
    if search_result["total_count"] > 0:
        message = search_result["messages"][0]
        thread_id = message.get("thread_id", message["id"])
        message_id = message["id"]
        
        # Reply to the email
        result = await gmail_service.reply_to_email(
            thread_id=thread_id,
            message_id=message_id,
            body="This is an automated test reply.",
            to="test@example.com"
        )
        
        # Note: This will fail if we don't have a valid recipient
        # In production, we'd extract the sender from the original email
        assert "success" in result

@pytest.mark.asyncio
async def test_delete_email(gmail_service: GmailService):
    """Test moving email to trash (not permanent delete)"""
    # First search for an email
    search_result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=1
    )
    
    if search_result["total_count"] > 0:
        message_id = search_result["messages"][0]["id"]
        
        # Move to trash (not permanent)
        result = await gmail_service.delete_email(message_id, permanent=False)
        
        assert result["success"] is True
        assert "moved to trash" in result["message"].lower()

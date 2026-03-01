"""
Test Gmail Agent - Attachment Handling
Task 3.2.4: Test attachment handling up to 25MB
"""
import pytest
from pathlib import Path
from agents.gmail_agent.service import GmailService

@pytest.mark.asyncio
async def test_search_emails_with_attachments(gmail_service: GmailService):
    """Test searching for emails with attachments"""
    result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=5,
        include_payload=True
    )
    
    assert result["success"] is True
    assert "messages" in result

@pytest.mark.asyncio
async def test_download_attachment(gmail_service: GmailService):
    """Test downloading a single attachment"""
    # First find an email with attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=1,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        message = search_result["messages"][0]
        message_id = message["id"]
        
        # Get full email details
        email_result = await gmail_service.get_email(message_id)
        
        if email_result["success"]:
            email_data = email_result["message"]
            attachments = email_data.get("attachments", [])
            
            if attachments:
                attachment = attachments[0]
                attachment_id = attachment["id"]
                file_name = attachment.get("filename", "test_attachment")
                
                # Download attachment
                result = await gmail_service.download_attachment(
                    message_id=message_id,
                    attachment_id=attachment_id,
                    file_name=file_name
                )
                
                assert result["success"] is True
                assert "file_path" in result
                assert "file_name" in result
                
                # Verify file exists
                file_path = Path(result["file_path"])
                assert file_path.exists()
                
                # Cleanup
                if file_path.exists():
                    file_path.unlink()

@pytest.mark.asyncio
async def test_download_all_attachments(gmail_service: GmailService):
    """Test downloading all attachments from an email"""
    # Find an email with attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=1,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        message_id = search_result["messages"][0]["id"]
        
        # Download all attachments
        result = await gmail_service.download_all_attachments(message_id)
        
        assert result["success"] is True
        assert "files" in result
        assert isinstance(result["files"], list)
        
        # Cleanup downloaded files
        for file_info in result["files"]:
            if file_info.get("success") and "file_path" in file_info:
                file_path = Path(file_info["file_path"])
                if file_path.exists():
                    file_path.unlink()

@pytest.mark.asyncio
async def test_download_attachment_no_attachments(gmail_service: GmailService):
    """Test downloading attachments from email with no attachments"""
    # Find an email without attachments
    search_result = await gmail_service.search_emails(
        query="-has:attachment label:inbox",
        max_results=1
    )
    
    if search_result["total_count"] > 0:
        message_id = search_result["messages"][0]["id"]
        
        # Try to download attachments
        result = await gmail_service.download_all_attachments(message_id)
        
        assert result["success"] is True
        assert result["message"] == "No attachments found"
        assert result["files"] == []

@pytest.mark.asyncio
async def test_attachment_size_handling(gmail_service: GmailService):
    """Test handling of various attachment sizes"""
    # Search for emails with attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=5,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        for message in search_result["messages"]:
            message_id = message["id"]
            
            # Get email details
            email_result = await gmail_service.get_email(message_id)
            
            if email_result["success"]:
                email_data = email_result["message"]
                attachments = email_data.get("attachments", [])
                
                for attachment in attachments:
                    size = attachment.get("size", 0)
                    
                    # Log attachment size for analysis
                    print(f"Attachment: {attachment.get('filename')} - Size: {size} bytes ({size / 1024 / 1024:.2f} MB)")
                    
                    # Verify size is within Gmail's 25MB limit
                    assert size <= 25 * 1024 * 1024, f"Attachment exceeds 25MB limit: {size} bytes"

@pytest.mark.asyncio
async def test_attachment_storage_directory(gmail_service: GmailService):
    """Test that attachments are stored in user-specific directories"""
    from agents.gmail_agent.config import ATTACHMENT_DIR
    
    # Find an email with attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=1,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        message = search_result["messages"][0]
        message_id = message["id"]
        
        # Get email details
        email_result = await gmail_service.get_email(message_id)
        
        if email_result["success"]:
            email_data = email_result["message"]
            attachments = email_data.get("attachments", [])
            
            if attachments:
                attachment = attachments[0]
                attachment_id = attachment["id"]
                file_name = attachment.get("filename", "test_file")
                
                # Download attachment
                result = await gmail_service.download_attachment(
                    message_id=message_id,
                    attachment_id=attachment_id,
                    file_name=file_name
                )
                
                if result["success"]:
                    file_path = Path(result["file_path"])
                    
                    # Verify file is in user-specific directory
                    expected_dir = ATTACHMENT_DIR / gmail_service.user_id
                    assert file_path.parent == expected_dir
                    
                    # Cleanup
                    if file_path.exists():
                        file_path.unlink()

@pytest.mark.asyncio
async def test_attachment_base64_decoding(gmail_service: GmailService):
    """Test that attachments are properly base64 decoded"""
    # Find an email with attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=1,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        message = search_result["messages"][0]
        message_id = message["id"]
        
        # Get email details
        email_result = await gmail_service.get_email(message_id)
        
        if email_result["success"]:
            email_data = email_result["message"]
            attachments = email_data.get("attachments", [])
            
            if attachments:
                attachment = attachments[0]
                attachment_id = attachment["id"]
                file_name = attachment.get("filename", "test_file")
                
                # Download attachment
                result = await gmail_service.download_attachment(
                    message_id=message_id,
                    attachment_id=attachment_id,
                    file_name=file_name
                )
                
                if result["success"]:
                    file_path = Path(result["file_path"])
                    
                    # Verify file exists and has content
                    assert file_path.exists()
                    assert file_path.stat().st_size > 0
                    
                    # Cleanup
                    if file_path.exists():
                        file_path.unlink()

@pytest.mark.asyncio
async def test_concurrent_attachment_downloads(gmail_service: GmailService):
    """Test downloading multiple attachments concurrently"""
    # Find an email with multiple attachments
    search_result = await gmail_service.search_emails(
        query="has:attachment",
        max_results=1,
        include_payload=True
    )
    
    if search_result["total_count"] > 0:
        message_id = search_result["messages"][0]["id"]
        
        # Download all attachments (uses asyncio.gather internally)
        result = await gmail_service.download_all_attachments(message_id)
        
        assert result["success"] is True
        
        # Verify all downloads completed
        if result["files"]:
            for file_info in result["files"]:
                if file_info.get("success"):
                    assert "file_path" in file_info
                    
                    # Cleanup
                    file_path = Path(file_info["file_path"])
                    if file_path.exists():
                        file_path.unlink()

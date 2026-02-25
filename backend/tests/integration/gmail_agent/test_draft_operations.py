"""
Test Gmail Agent - Draft Operations
Task 3.2.3: Test draft operations (create, list, send, delete)
"""
import pytest
from agents.gmail_agent.service import GmailService

@pytest.mark.asyncio
async def test_create_draft(gmail_service: GmailService, sample_draft_data: dict):
    """Test creating an email draft"""
    result = await gmail_service.create_draft(
        to=sample_draft_data["to"],
        subject=sample_draft_data["subject"],
        body=sample_draft_data["body"]
    )
    
    assert result["success"] is True
    assert "draft" in result
    assert "id" in result["draft"]
    assert result["message"] == "Draft created"
    
    # Return draft ID for cleanup
    return result["draft"]["id"]

@pytest.mark.asyncio
async def test_list_drafts(gmail_service: GmailService):
    """Test listing email drafts"""
    result = await gmail_service.list_drafts(max_results=10)
    
    assert result["success"] is True
    assert "drafts" in result
    assert isinstance(result["drafts"], list)

@pytest.mark.asyncio
async def test_create_and_delete_draft(gmail_service: GmailService, sample_draft_data: dict):
    """Test creating and then deleting a draft"""
    # Create draft
    create_result = await gmail_service.create_draft(
        to=sample_draft_data["to"],
        subject=sample_draft_data["subject"],
        body=sample_draft_data["body"]
    )
    
    assert create_result["success"] is True
    draft_id = create_result["draft"]["id"]
    
    # Delete draft
    delete_result = await gmail_service.delete_draft(draft_id)
    
    assert delete_result["success"] is True
    assert delete_result["message"] == "Draft deleted"

@pytest.mark.asyncio
async def test_create_draft_with_cc(gmail_service: GmailService):
    """Test creating draft with CC recipients"""
    result = await gmail_service.create_draft(
        to="test@example.com",
        subject="Test Draft with CC",
        body="This draft has CC recipients.",
        cc=["cc1@example.com", "cc2@example.com"]
    )
    
    assert result["success"] is True
    assert "draft" in result
    
    # Cleanup
    if result["success"]:
        await gmail_service.delete_draft(result["draft"]["id"])

@pytest.mark.asyncio
async def test_create_draft_in_thread(gmail_service: GmailService):
    """Test creating draft as reply in a thread"""
    # First search for an email to get a thread ID
    search_result = await gmail_service.search_emails(
        query="label:inbox",
        max_results=1
    )
    
    if search_result["total_count"] > 0:
        thread_id = search_result["messages"][0].get("thread_id")
        
        if thread_id:
            result = await gmail_service.create_draft(
                to="test@example.com",
                subject="Re: Test",
                body="This is a draft reply in a thread.",
                thread_id=thread_id
            )
            
            assert result["success"] is True
            
            # Cleanup
            if result["success"]:
                await gmail_service.delete_draft(result["draft"]["id"])

@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires real Gmail connection and sends actual emails")
async def test_send_draft(gmail_service: GmailService, sample_draft_data: dict):
    """Test sending an existing draft"""
    # Create draft
    create_result = await gmail_service.create_draft(
        to=sample_draft_data["to"],
        subject=sample_draft_data["subject"],
        body=sample_draft_data["body"]
    )
    
    assert create_result["success"] is True
    draft_id = create_result["draft"]["id"]
    
    # Send draft
    send_result = await gmail_service.send_draft(draft_id)
    
    assert send_result["success"] is True
    assert "Draft sent successfully" in send_result["message"]

@pytest.mark.asyncio
async def test_draft_lifecycle(gmail_service: GmailService):
    """Test complete draft lifecycle: create -> list -> delete"""
    # Create draft
    create_result = await gmail_service.create_draft(
        to="lifecycle@example.com",
        subject="Draft Lifecycle Test",
        body="Testing the complete draft lifecycle."
    )
    
    assert create_result["success"] is True
    draft_id = create_result["draft"]["id"]
    
    # List drafts and verify it exists
    list_result = await gmail_service.list_drafts(max_results=50)
    assert list_result["success"] is True
    
    draft_ids = [d.get("id") for d in list_result["drafts"]]
    assert draft_id in draft_ids, "Created draft should appear in list"
    
    # Delete draft
    delete_result = await gmail_service.delete_draft(draft_id)
    assert delete_result["success"] is True
    
    # Verify it's deleted (list again)
    list_after_delete = await gmail_service.list_drafts(max_results=50)
    assert list_after_delete["success"] is True
    
    draft_ids_after = [d.get("id") for d in list_after_delete["drafts"]]
    assert draft_id not in draft_ids_after, "Deleted draft should not appear in list"

@pytest.mark.asyncio
async def test_delete_nonexistent_draft(gmail_service: GmailService):
    """Test deleting a draft that doesn't exist"""
    fake_draft_id = "nonexistent_draft_12345"
    
    result = await gmail_service.delete_draft(fake_draft_id)
    
    # Should return error
    assert result["success"] is False
    assert "error" in result

# agents/mail_agent/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class GmailRequest(BaseModel):
    """Generic Gmail operation request"""
    operation: str = Field(..., description="Gmail operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")

# FetchEmailsRequest removed as part of Smart Agent transition

class SendEmailRequest(BaseModel):
    """Request model for sending emails"""
    to: list = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    cc: list = Field(default=[], description="CC email addresses")
    bcc: list = Field(default=[], description="BCC email addresses")
    user_id: str = Field(default="me", description="Gmail user ID")
    is_html: bool = Field(default=False, description="Whether body is HTML formatted")
    attachment_file_ids: list = Field(default=[], description="List of file IDs from file manager to attach")
    attachment_paths: list = Field(default=[], description="List of local file paths to attach")
    show_preview: bool = Field(default=False, description="Show email preview in canvas before sending (optional confirmation)")

class GmailResponse(BaseModel):
    """Gmail operation response"""
    success: bool
    result: Any
    error: Optional[str] = None
    canvas_display: Optional[Dict[str, Any]] = Field(
        None,
        description="Canvas display data for visual preview"
    )

# Removed FetchSentEmailsRequest, GetSentEmailRequest


class DownloadAttachmentsRequest(BaseModel):
    """Request model for downloading attachments from an email"""
    message_id: str = Field(..., description="Gmail message ID")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID for orchestrator integration")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

class SemanticSearchRequest(BaseModel):
    """Request model for semantic search (now /search)"""
    query: str = Field(..., description="Vague or specific search query")
    max_results: int = Field(default=10, description="Maximum results per keyword search")
    user_id: str = Field(default="me", description="Gmail user ID")

class SummarizeRequest(BaseModel):
    """Request model for smart email summarization"""
    message_ids: Optional[List[str]] = Field(None, description="List of message IDs to summarize. If None, summarizes last search results.")
    user_id: str = Field(default="me", description="Gmail user ID")
    use_history: bool = Field(default=True, description="Whether to use agent memory finding emails to summarize")


class DraftReplyRequest(BaseModel):
    """Request to draft a context-aware reply"""
    message_id: str = Field(..., description="ID of the email to reply to")
    intent: str = Field(..., description="User's intent (e.g., 'accept invite', 'ask for more time')")
    user_id: str = Field(default="me", description="Gmail user ID")

class ExtractActionItemsRequest(BaseModel):
    """Request to extract task list from emails"""
    message_ids: Optional[List[str]] = Field(None, description="Specific emails to scan. If None, uses last search.")
    user_id: str = Field(default="me", description="Gmail user ID")
    use_history: bool = Field(default=True, description="Use recent search results if no IDs provided")

# Removed ComposeHtmlEmailRequest, FormatEmailRequest

class EmailAction(str, Enum):
    MARK_READ = "mark_read"
    MARK_UNREAD = "mark_unread"
    ARCHIVE = "archive"
    MOVE_TO_INBOX = "move_to_inbox"
    DELETE = "delete"
    STAR = "star"
    UNSTAR = "unstar"
    ADD_LABELS = "add_labels"
    REMOVE_LABELS = "remove_labels"

class ManageEmailsRequest(BaseModel):
    """Request model for managing email state (labels, read status, etc)"""
    message_ids: Optional[List[str]] = Field(None, description="List of message IDs to modify. If empty, uses last search results.")
    action: EmailAction = Field(..., description="Action to perform")
    labels: Optional[List[str]] = Field(None, description="Labels to add/remove (required for label actions)")
    use_history: bool = Field(default=True, description="Apply action to last search results if message_ids is empty")
    user_id: str = Field(default="me", description="Gmail user ID")

# agents/gmail_agent/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# === Request Models ===

class SearchRequest(BaseModel):
    user_id: str
    query: str
    max_results: int = 10
    include_payload: bool = False
    label_ids: Optional[List[str]] = None

class SendEmailRequest(BaseModel):
    user_id: str
    to: str
    subject: str
    body: str
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    is_html: bool = False
    attachments: Optional[List[Dict[str, Any]]] = None

class ReplyRequest(BaseModel):
    user_id: str
    thread_id: str
    message_id: str
    body: str
    to: str
    cc: Optional[List[str]] = None
    attachments: Optional[List[Dict[str, Any]]] = None

class CreateDraftRequest(BaseModel):
    user_id: str
    to: str
    subject: str
    body: str
    cc: Optional[List[str]] = None
    thread_id: Optional[str] = None

class SummarizeRequest(BaseModel):
    user_id: str
    message_ids: List[str]
    summary_type: str = "concise"

class DraftReplyRequest(BaseModel):
    user_id: str
    message_id: str
    user_instructions: Optional[str] = None
    tone: str = "professional"

class ExtractActionsRequest(BaseModel):
    user_id: str
    message_ids: List[str]

class AddLabelsRequest(BaseModel):
    user_id: str
    message_id: str
    label_ids: List[str]

class DownloadAttachmentsRequest(BaseModel):
    user_id: str
    message_id: str
    attachment_ids: Optional[List[str]] = None

class ExecuteRequest(BaseModel):
    user_id: str
    prompt: str
    context: Optional[Dict[str, Any]] = None

# === Response Models ===

class GmailResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None

class EmailMessage(BaseModel):
    id: str
    thread_id: str
    subject: Optional[str] = None
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[List[str]] = None
    date: Optional[str] = None
    body: Optional[str] = None
    snippet: Optional[str] = None
    labels: Optional[List[str]] = None
    attachments: Optional[List[Dict[str, Any]]] = None

    class Config:
        populate_by_name = True

class SearchResponse(BaseModel):
    success: bool
    messages: List[EmailMessage]
    total_count: int
    next_page_token: Optional[str] = None

class SummaryResponse(BaseModel):
    success: bool
    summaries: List[Dict[str, Any]]
    overall_summary: Optional[str] = None

class ActionItemsResponse(BaseModel):
    success: bool
    action_items: List[Dict[str, Any]]
    by_email: Dict[str, List[str]]

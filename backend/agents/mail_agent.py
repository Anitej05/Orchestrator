# agents/mail_agent.py
"""
Mail Agent - Gmail integration via Composio
Provides email reading, sending, and management capabilities.
"""

import os
import asyncio
import logging
import base64
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
from pathlib import Path

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
MCP_URL = os.getenv("GMAIL_MCP_URL")  # e.g., https://mcp.composio.dev/gmail/sse?user_id=...
CONNECTION_ID = os.getenv("GMAIL_CONNECTION_ID")  # Gmail connection ID

if not COMPOSIO_API_KEY:
    logger.warning("COMPOSIO_API_KEY not set. Gmail agent will not function.")
if not MCP_URL:
    logger.warning("GMAIL_MCP_URL not set. Gmail agent will not function.")
if not CONNECTION_ID:
    logger.warning("GMAIL_CONNECTION_ID not set. Using URL without connection ID.")

# Agent Definition
AGENT_DEFINITION = {
    "id": "mail_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Mail Agent",
    "description": "Gmail integration via Composio. Read, send, search, and manage emails.",
    "capabilities": [
        "read emails",
        "send emails",
        "search emails",
        "fetch emails",
        "compose emails",
        "reply to emails",
        "manage email labels",
        "get email threads",
        "fetch unread emails",
        "send email with attachments",
        "search inbox",
        "check email",
        "email automation",
        "gmail integration",
        "email management",
        "fetch sent emails",
        "get sent email content",
        "view sent mail",
        "check sent folder",
        "get last sent email",
        "verify sent email"
    ],
    "price_per_call_usd": 0.005,
    "status": "active",
    "agent_type": "http_rest",
    "connection_config": {
        "base_url": "http://localhost:8040"
    },
    "endpoints": [
        {
            "endpoint": "/fetch_emails",
            "http_method": "POST",
            "description": "Fetch/search emails from Gmail inbox",
            "parameters": [
                {
                    "name": "query",
                    "param_type": "string",
                    "required": True,
                    "description": "Search query (e.g., 'is:unread', 'from:example@gmail.com', 'subject:invoice')"
                },
                {
                    "name": "max_results",
                    "param_type": "integer",
                    "required": False,
                    "description": "Maximum number of emails to fetch (default: 10)"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "/send_email",
            "http_method": "POST",
            "description": "Send a new email via Gmail",
            "parameters": [
                {
                    "name": "to",
                    "param_type": "array",
                    "required": True,
                    "description": "List of recipient email addresses"
                },
                {
                    "name": "subject",
                    "param_type": "string",
                    "required": True,
                    "description": "Email subject line"
                },
                {
                    "name": "body",
                    "param_type": "string",
                    "required": True,
                    "description": "Email body content"
                },
                {
                    "name": "cc",
                    "param_type": "array",
                    "required": False,
                    "description": "List of CC email addresses"
                },
                {
                    "name": "bcc",
                    "param_type": "array",
                    "required": False,
                    "description": "List of BCC email addresses"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
            "http_method": "MCP",
            "description": "Fetch full email details by message ID",
            "parameters": [
                {
                    "name": "message_id",
                    "param_type": "string",
                    "required": True,
                    "description": "Gmail message ID"
                },
                {
                    "name": "format",
                    "param_type": "string",
                    "required": False,
                    "description": "Message format: 'full' or 'minimal' (default: 'full')"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "GMAIL_LIST_THREADS",
            "http_method": "MCP",
            "description": "List email threads (for polling new emails)",
            "parameters": [
                {
                    "name": "query",
                    "param_type": "string",
                    "required": False,
                    "description": "Search query (e.g., 'after:2025-11-24')"
                },
                {
                    "name": "max_results",
                    "param_type": "integer",
                    "required": False,
                    "description": "Maximum number of threads (default: 10)"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "GMAIL_CREATE_DRAFT",
            "http_method": "MCP",
            "description": "Create an email draft",
            "parameters": [
                {
                    "name": "to",
                    "param_type": "array",
                    "required": True,
                    "description": "List of recipient email addresses"
                },
                {
                    "name": "subject",
                    "param_type": "string",
                    "required": True,
                    "description": "Email subject line"
                },
                {
                    "name": "body",
                    "param_type": "string",
                    "required": True,
                    "description": "Email body content"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "/add_label",
            "http_method": "POST",
            "description": "Add labels to emails",
            "parameters": [
                {
                    "name": "message_ids",
                    "param_type": "array",
                    "required": True,
                    "description": "List of message IDs to label"
                },
                {
                    "name": "label_ids",
                    "param_type": "array",
                    "required": True,
                    "description": "List of label IDs to add (e.g., ['IMPORTANT', 'STARRED'])"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "/delete_message",
            "http_method": "POST",
            "description": "Delete emails by message ID",
            "parameters": [
                {
                    "name": "message_ids",
                    "param_type": "array",
                    "required": True,
                    "description": "List of message IDs to delete"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        },
        {
            "endpoint": "/get_attachment",
            "http_method": "POST",
            "description": "Download email attachment",
            "parameters": [
                {
                    "name": "message_id",
                    "param_type": "string",
                    "required": True,
                    "description": "Gmail message ID"
                },
                {
                    "name": "attachment_id",
                    "param_type": "string",
                    "required": True,
                    "description": "Attachment ID from the message"
                },
                {
                    "name": "user_id",
                    "param_type": "string",
                    "required": False,
                    "description": "Gmail user ID (default: 'me')"
                }
            ]
        }
    ]
}

app = FastAPI(title="Mail Agent")

# Request/Response Models
class GmailRequest(BaseModel):
    """Generic Gmail operation request"""
    operation: str = Field(..., description="Gmail operation to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")

class FetchEmailsRequest(BaseModel):
    """Request model for fetching emails"""
    query: str = Field(default="is:unread", description="Gmail search query")
    max_results: int = Field(default=10, description="Maximum number of emails to fetch")
    user_id: str = Field(default="me", description="Gmail user ID")
    label_ids: list = Field(default=["INBOX"], description="Label IDs to filter by")

class SendEmailRequest(BaseModel):
    """Request model for sending emails"""
    to: list = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    cc: list = Field(default=[], description="CC email addresses")
    bcc: list = Field(default=[], description="BCC email addresses")
    user_id: str = Field(default="me", description="Gmail user ID")

class GmailResponse(BaseModel):
    """Gmail operation response"""
    success: bool
    result: Any
    error: Optional[str] = None

# MCP Client Manager
class GmailClient:
    """Manages connection to Composio Gmail API"""
    
    def __init__(self):
        self.mcp_url = MCP_URL
        self.api_key = COMPOSIO_API_KEY
        self.connection_id = CONNECTION_ID
        self.attachments_dir = Path("storage/gmail_attachments")
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text by stripping tags"""
        try:
            # Remove style and script tags with content
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            # Replace <br> and </p> with newlines
            html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
            html = re.sub(r'</p>', '\n', html, flags=re.IGNORECASE)
            html = re.sub(r'</div>', '\n', html, flags=re.IGNORECASE)
            html = re.sub(r'</li>', '\n', html, flags=re.IGNORECASE)
            # Remove all other HTML tags
            html = re.sub(r'<[^>]+>', '', html)
            # Decode HTML entities
            html = html.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
            # Clean up whitespace
            html = re.sub(r'\n\s*\n', '\n\n', html)
            return html.strip()
        except Exception as e:
            logger.warning(f"Failed to convert HTML to text: {e}")
            return html

    def _extract_clean_text(self, payload: Dict) -> str:
        """Extract clean text from email payload, with HTML fallback"""
        try:
            # Try to find text/plain part first
            parts = payload.get("parts", [])
            html_content = None
            
            if not parts:
                # Single part message
                mime_type = payload.get("mimeType", "")
                data = payload.get("body", {}).get("data", "")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    if mime_type == "text/plain":
                        return decoded
                    elif mime_type == "text/html":
                        return self._html_to_text(decoded)
                return ""
            
            # Multi-part message - find text/plain first, then HTML as fallback
            def find_content(parts_list):
                nonlocal html_content
                plain_text = None
                
                for part in parts_list:
                    mime_type = part.get("mimeType", "")
                    
                    if mime_type == "text/plain":
                        data = part.get("body", {}).get("data", "")
                        if data:
                            plain_text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                            return plain_text
                    
                    elif mime_type == "text/html" and html_content is None:
                        data = part.get("body", {}).get("data", "")
                        if data:
                            html_content = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    
                    # Check nested parts (multipart/alternative, etc.)
                    if "parts" in part:
                        nested_result = find_content(part["parts"])
                        if nested_result:
                            return nested_result
                
                return plain_text
            
            plain_result = find_content(parts)
            
            # Return plain text if found, otherwise convert HTML
            if plain_result:
                return plain_result
            elif html_content:
                return self._html_to_text(html_content)
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to extract clean text: {e}")
            return ""
    
    def _find_attachments(self, payload: Dict) -> List[Dict]:
        """Find all attachments in email payload"""
        attachments = []
        
        def traverse_parts(parts):
            for part in parts:
                filename = part.get("filename", "")
                if filename and part.get("body", {}).get("attachmentId"):
                    attachments.append({
                        "filename": filename,
                        "attachment_id": part["body"]["attachmentId"],
                        "mime_type": part.get("mimeType", ""),
                        "size": part.get("body", {}).get("size", 0)
                    })
                
                # Check nested parts
                if "parts" in part:
                    traverse_parts(part["parts"])
        
        parts = payload.get("parts", [])
        if parts:
            traverse_parts(parts)
        
        return attachments
    
    async def _download_attachment(self, message_id: str, attachment_id: str, filename: str) -> Optional[str]:
        """Download attachment and return file path"""
        try:
            from composio import Composio, Action
            
            client = Composio(api_key=self.api_key)
            
            result = client.actions.execute(
                action=Action.GMAIL_GET_ATTACHMENT,
                params={
                    "message_id": message_id,
                    "attachment_id": attachment_id,
                    "file_name": filename
                },
                connected_account=self.connection_id
            )
            
            if result.get("successful") or result.get("successfull"):
                data = result.get("data", {})
                attachment_data = data.get("attachment_data", "")
                
                if attachment_data:
                    # Decode and save
                    decoded = base64.urlsafe_b64decode(attachment_data)
                    
                    # Create safe filename
                    safe_filename = re.sub(r'[^\w\s.-]', '_', filename)
                    filepath = self.attachments_dir / f"{message_id}_{safe_filename}"
                    
                    with open(filepath, "wb") as f:
                        f.write(decoded)
                    
                    logger.info(f"Saved attachment: {filepath}")
                    return str(filepath)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to download attachment: {e}")
            return None
    
    def _process_email_response(self, data: Dict, tool_name: str) -> Dict:
        """Process email response to extract clean content and handle attachments"""
        try:
            if tool_name == "GMAIL_FETCH_EMAILS":
                # Process list of emails
                messages = data.get("messages", [])
                processed_messages = []
                
                for msg in messages[:10]:  # Limit to prevent context overflow
                    processed = {
                        "id": msg.get("messageId") or msg.get("id"),
                        "subject": msg.get("subject", ""),
                        "from": msg.get("sender", ""),
                        "to": msg.get("to", ""),
                        "date": msg.get("messageTimestamp", ""),
                        "snippet": msg.get("preview", {}).get("body", "")[:200] if isinstance(msg.get("preview"), dict) else "",
                        "has_attachments": len(msg.get("attachmentList", [])) > 0
                    }
                    
                    # Extract clean text if messageText is available
                    if "messageText" in msg:
                        clean_text = msg["messageText"][:500]  # Limit text
                        processed["preview"] = clean_text
                    
                    processed_messages.append(processed)
                
                return {
                    "messages": processed_messages,
                    "count": len(processed_messages),
                    "total": data.get("resultSizeEstimate", len(messages))
                }
            
            elif tool_name == "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID":
                # Process single email with full details
                payload = data.get("payload", {})
                
                # Extract clean text
                clean_text = self._extract_clean_text(payload)
                
                # Find attachments
                attachments = self._find_attachments(payload)
                
                # Get headers
                headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
                
                return {
                    "id": data.get("id"),
                    "subject": headers.get("Subject", ""),
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "date": headers.get("Date", ""),
                    "body": clean_text[:2000] if clean_text else "(No plain text body)",  # Limit to 2000 chars
                    "attachments": [
                        {
                            "filename": att["filename"],
                            "size": att["size"],
                            "type": att["mime_type"]
                        }
                        for att in attachments
                    ],
                    "attachment_ids": {att["filename"]: att["attachment_id"] for att in attachments}
                }
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to process email response: {e}")
            return data
        
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a Gmail tool via Composio Python SDK.
        
        Args:
            tool_name: Name of the Gmail tool (e.g., "GMAIL_FETCH_EMAILS")
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Use Composio Python SDK
            from composio import Composio, Action
            
            if not self.api_key:
                return {
                    "success": False,
                    "error": "Missing COMPOSIO_API_KEY"
                }
            
            logger.info(f"Calling Gmail tool via Composio SDK: {tool_name}")
            logger.info(f"Parameters: {parameters}")
            
            # Initialize Composio client
            client = Composio(api_key=self.api_key)
            
            # Convert tool name to Action enum
            try:
                action_enum = getattr(Action, tool_name)
            except AttributeError:
                return {
                    "success": False,
                    "error": f"Unknown action: {tool_name}"
                }
            
            # Execute the action using the actions.execute() method
            result = client.actions.execute(
                action=action_enum,
                params=parameters,
                connected_account=self.connection_id if self.connection_id else None
            )
            
            logger.info(f"SDK Response: {str(result)[:200]}...")
            
            # Parse the result
            if isinstance(result, dict):
                # Check for success indicators
                if result.get("successful") or result.get("successfull"):
                    raw_data = result.get("data", result)
                    
                    # Process email content to extract clean text and handle attachments
                    processed_data = self._process_email_response(raw_data, tool_name)
                    
                    return {"success": True, "data": processed_data}
                elif "error" in result:
                    return {"success": False, "error": result["error"]}
                else:
                    # Assume success if no error
                    return {"success": True, "data": result}
            else:
                return {"success": True, "data": result}
                    
        except Exception as e:
            logger.error(f"Gmail tool call failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _detect_file_type(self, filename: str, mime_type: str) -> str:
        """Detect if file is image or document"""
        # Image types
        if mime_type.startswith("image/"):
            return "image"
        
        # Document types
        doc_extensions = [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"]
        doc_mimes = ["application/pdf", "application/msword", 
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                     "text/plain", "application/rtf"]
        
        ext = os.path.splitext(filename)[1].lower()
        if ext in doc_extensions or mime_type in doc_mimes:
            return "document"
        
        return "other"
    
    async def download_email_attachments(self, message_id: str) -> Dict[str, Any]:
        """Download all attachments from an email and return in orchestrator-compatible format"""
        try:
            # First fetch the message to get attachment info
            result = await self.call_tool(
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
                {"message_id": message_id, "format": "full"}
            )
            
            if not result.get("success"):
                return result
            
            data = result["data"]
            attachments = data.get("attachments", [])
            attachment_ids = data.get("attachment_ids", {})
            
            if not attachment_ids:
                return {
                    "success": True,
                    "message": "No attachments found",
                    "files": []
                }
            
            # Download each attachment
            downloaded_files = []
            for att_info in attachments:
                filename = att_info["filename"]
                att_id = attachment_ids.get(filename)
                
                if not att_id:
                    continue
                
                filepath = await self._download_attachment(message_id, att_id, filename)
                if filepath:
                    # Detect file type for orchestrator
                    file_type = self._detect_file_type(filename, att_info["type"])
                    
                    # Return in orchestrator-compatible format (FileObject schema)
                    downloaded_files.append({
                        "file_name": filename,
                        "file_path": filepath,
                        "file_type": file_type,
                        "mime_type": att_info["type"],
                        "size": att_info["size"],
                        "source": "gmail_attachment"
                    })
            
            return {
                "success": True,
                "message": f"Downloaded {len(downloaded_files)} attachment(s). Files are ready for analysis by image/document agents.",
                "files": downloaded_files,
                "instructions": "These files can now be analyzed. For images, use the image analysis agent with 'image_path'. For documents, they will be preprocessed into vector stores automatically."
            }
            
        except Exception as e:
            logger.error(f"Failed to download attachments: {e}")
            return {"success": False, "error": str(e)}

# Global client instance
gmail_client = GmailClient()

# API Endpoints
@app.post("/fetch_emails", response_model=GmailResponse)
async def fetch_emails(request: FetchEmailsRequest):
    """Fetch/search emails from Gmail"""
    try:
        params = {
            "query": request.query,
            "max_results": request.max_results,
            "user_id": request.user_id,
            "label_ids": request.label_ids
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_EMAILS", params)
        
        if result.get("success"):
            return GmailResponse(success=True, result=result.get("data"))
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Fetch emails failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_email", response_model=GmailResponse)
async def send_email(request: SendEmailRequest):
    """Send an email via Gmail - returns full details of what was sent"""
    try:
        # Map our parameter names to Composio's expected parameter names
        params = {
            "recipient_email": request.to[0] if request.to else "",  # Composio expects single email string
            "subject": request.subject,
            "body": request.body,
            "user_id": request.user_id
        }
        
        # Add cc and bcc only if provided (Composio expects them as strings, comma-separated)
        if request.cc:
            params["cc"] = ",".join(request.cc)
        if request.bcc:
            params["bcc"] = ",".join(request.bcc)
        
        result = await gmail_client.call_tool("GMAIL_SEND_EMAIL", params)
        
        if result.get("success"):
            # ENHANCED: Include the sent content in the response
            api_data = result.get("data", {})
            enhanced_result = {
                "status": "sent",
                "message_id": api_data.get("id") or api_data.get("messageId"),
                "thread_id": api_data.get("threadId"),
                "labels": api_data.get("labelIds", []),
                # Include what was actually sent
                "sent_content": {
                    "to": request.to,
                    "cc": request.cc if request.cc else [],
                    "bcc": request.bcc if request.bcc else [],
                    "subject": request.subject,
                    "body": request.body
                }
            }
            return GmailResponse(success=True, result=enhanced_result)
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Send email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_message", response_model=GmailResponse)
async def get_message(request: GmailRequest):
    """Get full message details by ID"""
    try:
        params = {
            "message_id": request.parameters.get("message_id"),
            "format": request.parameters.get("format", "full"),
            "user_id": request.parameters.get("user_id", "me")
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", params)
        
        if result.get("success"):
            return GmailResponse(success=True, result=result.get("data"))
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Get message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== NEW ENHANCED ENDPOINTS ==============

class FetchSentEmailsRequest(BaseModel):
    """Request model for fetching sent emails"""
    max_results: int = Field(default=10, description="Maximum number of sent emails to fetch")
    user_id: str = Field(default="me", description="Gmail user ID")
    include_body: bool = Field(default=True, description="Whether to include full email body")

class GetSentEmailRequest(BaseModel):
    """Request model for getting a specific sent email"""
    message_id: str = Field(..., description="Gmail message ID")
    user_id: str = Field(default="me", description="Gmail user ID")

@app.post("/fetch_sent_emails", response_model=GmailResponse)
async def fetch_sent_emails(request: FetchSentEmailsRequest):
    """
    Fetch sent emails with full content (subject, body, recipients).
    Use this to see what emails you have sent.
    """
    try:
        # First, get the list of sent emails
        params = {
            "query": "in:sent",
            "max_results": request.max_results,
            "user_id": request.user_id,
            "label_ids": ["SENT"]
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_EMAILS", params)
        
        if not result.get("success"):
            return GmailResponse(success=False, result=None, error=result.get("error"))
        
        messages = result.get("data", {}).get("messages", [])
        
        if not request.include_body:
            # Return just the list without fetching full content
            return GmailResponse(success=True, result={
                "sent_emails": messages,
                "count": len(messages)
            })
        
        # Fetch full content for each message
        detailed_emails = []
        for msg in messages[:request.max_results]:
            msg_id = msg.get("id") or msg.get("messageId")
            if not msg_id:
                continue
            
            # Get full message details
            detail_result = await gmail_client.call_tool(
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
                {"message_id": msg_id, "format": "full", "user_id": request.user_id}
            )
            
            if detail_result.get("success"):
                detailed_emails.append(detail_result.get("data"))
            else:
                # Include partial info if full fetch fails
                detailed_emails.append({
                    "id": msg_id,
                    "error": "Could not fetch full details",
                    "partial_info": msg
                })
        
        return GmailResponse(success=True, result={
            "sent_emails": detailed_emails,
            "count": len(detailed_emails),
            "note": "Full email content including subject, body, and recipients"
        })
        
    except Exception as e:
        logger.error(f"Fetch sent emails failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_sent_email", response_model=GmailResponse)
async def get_sent_email(request: GetSentEmailRequest):
    """
    Get a specific sent email by message ID with full content.
    Use this to see the exact subject and body of an email you sent.
    """
    try:
        result = await gmail_client.call_tool(
            "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
            {"message_id": request.message_id, "format": "full", "user_id": request.user_id}
        )
        
        if result.get("success"):
            data = result.get("data", {})
            return GmailResponse(success=True, result={
                "message_id": request.message_id,
                "subject": data.get("subject", ""),
                "from": data.get("from", ""),
                "to": data.get("to", ""),
                "date": data.get("date", ""),
                "body": data.get("body", ""),
                "attachments": data.get("attachments", [])
            })
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Get sent email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_last_sent_email", response_model=GmailResponse)
async def get_last_sent_email():
    """
    Get the most recently sent email with full content.
    Convenient endpoint to quickly check what was just sent.
    """
    try:
        # Fetch the most recent sent email
        params = {
            "query": "in:sent",
            "max_results": 1,
            "user_id": "me",
            "label_ids": ["SENT"]
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_EMAILS", params)
        
        if not result.get("success"):
            return GmailResponse(success=False, result=None, error=result.get("error"))
        
        messages = result.get("data", {}).get("messages", [])
        
        if not messages:
            return GmailResponse(success=True, result={
                "message": "No sent emails found",
                "sent_emails": []
            })
        
        # Get the first (most recent) message
        msg = messages[0]
        msg_id = msg.get("id") or msg.get("messageId")
        
        if not msg_id:
            return GmailResponse(success=False, result=None, error="Could not get message ID")
        
        # Fetch full details
        detail_result = await gmail_client.call_tool(
            "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
            {"message_id": msg_id, "format": "full", "user_id": "me"}
        )
        
        if detail_result.get("success"):
            data = detail_result.get("data", {})
            return GmailResponse(success=True, result={
                "message_id": msg_id,
                "subject": data.get("subject", ""),
                "from": data.get("from", ""),
                "to": data.get("to", ""),
                "date": data.get("date", ""),
                "body": data.get("body", ""),
                "attachments": data.get("attachments", []),
                "note": "This is the most recently sent email"
            })
        else:
            return GmailResponse(success=False, result=None, error=detail_result.get("error"))
            
    except Exception as e:
        logger.error(f"Get last sent email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DownloadAttachmentsRequest(BaseModel):
    """Request model for downloading attachments from an email"""
    message_id: str = Field(..., description="Gmail message ID")

@app.post("/download_attachments", response_model=GmailResponse)
async def download_attachments(request: DownloadAttachmentsRequest):
    """
    Download all attachments from an email.
    Returns file paths that can be used by image/document agents for analysis.
    Files are saved to storage/gmail_attachments/
    """
    try:
        result = await gmail_client.download_email_attachments(request.message_id)
        
        if result.get("success"):
            return GmailResponse(success=True, result=result)
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Download attachments failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_emails_with_attachments", response_model=GmailResponse)
async def search_emails_with_attachments(max_results: int = 10):
    """
    Search for emails that have attachments.
    Returns list of emails with attachment info.
    """
    try:
        params = {
            "query": "has:attachment",
            "max_results": max_results,
            "user_id": "me",
            "label_ids": ["INBOX"]
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_EMAILS", params)
        
        if result.get("success"):
            return GmailResponse(success=True, result=result.get("data"))
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Search emails with attachments failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "mail_agent",
        "composio_configured": bool(COMPOSIO_API_KEY and CONNECTION_ID)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MAIL_AGENT_PORT", 8040))
    logger.info(f"Starting Mail Agent on port {port}...")
    uvicorn.run("mail_agent:app", host="0.0.0.0", port=port, reload=False)

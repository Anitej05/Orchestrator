# agents/mail_agent.py
"""
Mail Agent - Gmail integration via Composio
Provides email reading, sending, and management capabilities.
"""

import os
import sys
import asyncio
import logging
import base64
import re
import time
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add parent directory to path for imports when running as standalone
CURRENT_DIR = Path(__file__).parent
BACKEND_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import standardized file manager
try:
    from agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    try:
        from utils.agent_file_manager import AgentFileManager, FileType, FileStatus
    except ImportError:
        logger.error("Failed to import agent_file_manager from any location")
        raise

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

# MCP Client Manager
class GmailClient:
    """Manages connection to Composio Gmail API"""
    
    def __init__(self):
        self.mcp_url = MCP_URL
        self.api_key = COMPOSIO_API_KEY
        self.connection_id = CONNECTION_ID
        self.attachments_dir = Path("storage/gmail_attachments")
        self.attachments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize standardized file manager for attachments
        self.file_manager = AgentFileManager(
            agent_id="mail_agent",
            storage_dir=str(self.attachments_dir),
            default_ttl_hours=72,  # Attachments expire after 3 days by default
            auto_cleanup=True,
            cleanup_interval_hours=6
        )
        
        # Initialize metrics
        self._metrics_start_time = time.time()
        self.metrics = {
            "emails": {
                "fetched": 0,
                "sent": 0,
                "read": 0,
                "deleted": 0,
                "total_operations": 0
            },
            "attachments": {
                "downloaded": 0,
                "uploaded": 0,
                "total_size_mb": 0
            },
            "api_calls": {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "by_operation": {}
            },
            "performance": {
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "operations_completed": 0
            },
            "errors": {
                "total": 0,
                "api_errors": 0,
                "processing_errors": 0
            },
            "resource": {
                "peak_memory_mb": 0,
                "current_memory_mb": 0
            }
        }
    
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
    
    async def _download_attachment(
        self, 
        message_id: str, 
        attachment_id: str, 
        filename: str,
        mime_type: str = "",
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download attachment using standardized file manager.
        
        Returns:
            Dict with file metadata in orchestrator-compatible format, or None on failure
        """
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
                    # Decode attachment
                    decoded = base64.urlsafe_b64decode(attachment_data)
                    
                    # Create safe filename
                    safe_filename = re.sub(r'[^\w\s.-]', '_', filename)
                    
                    # Register with standardized file manager
                    metadata = await self.file_manager.register_file(
                        content=decoded,
                        filename=safe_filename,
                        mime_type=mime_type or None,
                        file_type=FileType.ATTACHMENT,
                        thread_id=thread_id,
                        user_id=user_id,
                        custom_metadata={
                            "message_id": message_id,
                            "attachment_id": attachment_id,
                            "original_filename": filename,
                            "source": "gmail"
                        },
                        tags=["attachment", "gmail", f"msg:{message_id}"]
                    )
                    
                    logger.info(f"Saved attachment via file manager: {metadata.file_id} ({filename})")
                    
                    # Track attachment download
                    self.metrics["attachments"]["downloaded"] += 1
                    file_size_mb = len(decoded) / (1024 * 1024)
                    self.metrics["attachments"]["total_size_mb"] += file_size_mb
                    
                    # Update resource metrics
                    process = psutil.Process()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    self.metrics["resource"]["current_memory_mb"] = current_memory
                    self.metrics["resource"]["peak_memory_mb"] = max(
                        self.metrics["resource"]["peak_memory_mb"],
                        current_memory
                    )
                    
                    # Return orchestrator-compatible format
                    return metadata.to_orchestrator_format()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to download attachment: {e}")
            return None
    
    def _process_email_response(self, data: Dict, tool_name: str) -> Dict:
        """Process email response to extract clean content and handle attachments"""
        try:
            if tool_name == "GMAIL_FETCH_EMAILS":
                # Track fetched emails
                self.metrics["emails"]["fetched"] += 1
                self.metrics["emails"]["total_operations"] += 1
                
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
        
    async def send_email_with_attachments(
        self,
        to: list,
        subject: str,
        body: str,
        cc: list = None,
        bcc: list = None,
        is_html: bool = False,
        attachment_file_ids: list = None,
        attachment_paths: list = None,
        user_id: str = "me"
    ) -> Dict[str, Any]:
        """
        Send email using Composio's native parameters (no raw MIME).
        HTML emails work via is_html parameter.
        Note: Attachments require S3 upload (not implemented yet).
        """
        try:
            from composio import Composio, Action
            
            if not self.api_key:
                return {"success": False, "error": "Missing COMPOSIO_API_KEY"}
            
            # Prepare parameters for Composio
            params = {
                "recipient_email": to[0] if to else "",
                "subject": subject,
                "body": body,
                "is_html": is_html,
                "user_id": user_id
            }
            
            # Handle multiple recipients
            if len(to) > 1:
                params["extra_recipients"] = to[1:]
            
            # Add CC and BCC
            if cc:
                params["cc"] = cc
            if bcc:
                params["bcc"] = bcc
            
            # Log attachment warning if provided
            attachment_count = 0
            if attachment_file_ids or attachment_paths:
                logger.warning("Attachments requested but not yet implemented (requires S3 upload)")
                attachment_count = len(attachment_file_ids or []) + len(attachment_paths or [])
            
            # Send via Composio using native parameters
            client = Composio(api_key=self.api_key)
            
            logger.info(f"Sending email with params: recipient={params['recipient_email']}, subject={params['subject']}, is_html={is_html}")
            
            result = client.actions.execute(
                action=Action.GMAIL_SEND_EMAIL,
                params=params,
                connected_account=self.connection_id
            )
            
            logger.info(f"Email send result: {result}")
            
            if result.get("successful") or result.get("successfull"):
                self.metrics["emails"]["sent"] += 1
                self.metrics["emails"]["total_operations"] += 1
                return {
                    "success": True,
                    "data": result.get("data", {}),
                    "attachments_sent": 0,  # Not implemented yet
                    "attachments_requested": attachment_count
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a Gmail tool via Composio Python SDK.
        
        Args:
            tool_name: Name of the Gmail tool (e.g., "GMAIL_FETCH_EMAILS")
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        call_start = time.time()
        operation_type = tool_name.replace("GMAIL_", "").lower()
        
        try:
            # Use Composio Python SDK
            from composio import Composio, Action
            
            if not self.api_key:
                self.metrics["errors"]["total"] += 1
                self.metrics["errors"]["api_errors"] += 1
                return {
                    "success": False,
                    "error": "Missing COMPOSIO_API_KEY"
                }
            
            logger.info(f"Calling Gmail tool via Composio SDK: {tool_name}")
            logger.info(f"Parameters: {parameters}")
            
            # Track API call
            self.metrics["api_calls"]["total"] += 1
            if operation_type not in self.metrics["api_calls"]["by_operation"]:
                self.metrics["api_calls"]["by_operation"][operation_type] = 0
            self.metrics["api_calls"]["by_operation"][operation_type] += 1
            
            # Initialize Composio client
            client = Composio(api_key=self.api_key)
            
            # Convert tool name to Action enum
            try:
                action_enum = getattr(Action, tool_name)
            except AttributeError:
                self.metrics["errors"]["total"] += 1
                self.metrics["errors"]["api_errors"] += 1
                self.metrics["api_calls"]["failed"] += 1
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
            
            # Track latency
            latency_ms = (time.time() - call_start) * 1000
            self.metrics["performance"]["total_latency_ms"] += latency_ms
            self.metrics["performance"]["operations_completed"] += 1
            self.metrics["performance"]["avg_latency_ms"] = (
                self.metrics["performance"]["total_latency_ms"] / 
                self.metrics["performance"]["operations_completed"]
            )
            
            # Parse the result
            if isinstance(result, dict):
                # Check for success indicators
                if result.get("successful") or result.get("successfull"):
                    self.metrics["api_calls"]["successful"] += 1
                    raw_data = result.get("data", result)
                    
                    # Process email content to extract clean text and handle attachments
                    processed_data = self._process_email_response(raw_data, tool_name)
                    
                    return {"success": True, "data": processed_data}
                elif "error" in result:
                    self.metrics["api_calls"]["failed"] += 1
                    self.metrics["errors"]["total"] += 1
                    self.metrics["errors"]["api_errors"] += 1
                    return {"success": False, "error": result["error"]}
                else:
                    # Assume success if no error
                    self.metrics["api_calls"]["successful"] += 1
                    return {"success": True, "data": result}
            else:
                self.metrics["api_calls"]["successful"] += 1
                return {"success": True, "data": result}
                    
        except Exception as e:
            latency_ms = (time.time() - call_start) * 1000
            self.metrics["performance"]["total_latency_ms"] += latency_ms
            self.metrics["performance"]["operations_completed"] += 1
            self.metrics["api_calls"]["failed"] += 1
            self.metrics["errors"]["total"] += 1
            self.metrics["errors"]["api_errors"] += 1
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
    
    async def download_email_attachments(
        self, 
        message_id: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download all attachments from an email using standardized file manager.
        Returns in orchestrator-compatible format.
        """
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
            
            # Download each attachment using standardized file manager
            downloaded_files = []
            for att_info in attachments:
                filename = att_info["filename"]
                att_id = attachment_ids.get(filename)
                
                if not att_id:
                    continue
                
                # Use updated _download_attachment that returns orchestrator format
                file_metadata = await self._download_attachment(
                    message_id=message_id,
                    attachment_id=att_id,
                    filename=filename,
                    mime_type=att_info.get("type", ""),
                    thread_id=thread_id,
                    user_id=user_id
                )
                
                if file_metadata:
                    # Add additional info for orchestrator
                    file_metadata["size"] = att_info.get("size", 0)
                    file_metadata["source"] = "gmail_attachment"
                    downloaded_files.append(file_metadata)
            
            return {
                "success": True,
                "message": f"Downloaded {len(downloaded_files)} attachment(s). Files are ready for analysis by image/document agents.",
                "files": downloaded_files,
                "instructions": "These files can now be analyzed. For images, use the image analysis agent with 'image_path'. For documents, they will be preprocessed into vector stores automatically."
            }
            
        except Exception as e:
            logger.error(f"Failed to download attachments: {e}")
            return {"success": False, "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive mail agent metrics."""
        uptime_seconds = time.time() - self._metrics_start_time
        
        total_api_calls = self.metrics["api_calls"]["total"]
        success_rate = (
            (self.metrics["api_calls"]["successful"] / total_api_calls * 100) 
            if total_api_calls > 0 else 0
        )
        
        # Update resource metrics
        process = psutil.Process()
        self.metrics["resource"]["current_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        return {
            "uptime_seconds": uptime_seconds,
            "emails": self.metrics["emails"].copy(),
            "attachments": self.metrics["attachments"].copy(),
            "api_calls": self.metrics["api_calls"].copy(),
            "success_rate": success_rate,
            "performance": self.metrics["performance"].copy(),
            "errors": self.metrics["errors"].copy(),
            "resource": self.metrics["resource"].copy()
        }

    def _log_execution_metrics(self, operation: str, success: bool):
        """Log execution metrics with clean formatting."""
        status_emoji = "✅" if success else "❌"
        
        logger.info("")
        logger.info(f"{status_emoji} MAIL AGENT METRICS - {operation}")
        logger.info("")
        
        # Emails
        logger.info("Email Operations:")
        logger.info(f"  Fetched: {self.metrics['emails']['fetched']}")
        logger.info(f"  Sent: {self.metrics['emails']['sent']}")
        logger.info(f"  Read: {self.metrics['emails']['read']}")
        logger.info(f"  Total: {self.metrics['emails']['total_operations']}")
        
        # Attachments
        if self.metrics['attachments']['downloaded'] > 0 or self.metrics['attachments']['uploaded'] > 0:
            logger.info("")
            logger.info("Attachments:")
            logger.info(f"  Downloaded: {self.metrics['attachments']['downloaded']}")
            logger.info(f"  Total Size: {self.metrics['attachments']['total_size_mb']:.2f} MB")
        
        # API Calls
        logger.info("")
        logger.info("API Calls:")
        logger.info(f"  Total: {self.metrics['api_calls']['total']}")
        logger.info(f"  Successful: {self.metrics['api_calls']['successful']}")
        logger.info(f"  Failed: {self.metrics['api_calls']['failed']}")
        success_rate = (self.metrics['api_calls']['successful'] / self.metrics['api_calls']['total'] * 100) if self.metrics['api_calls']['total'] > 0 else 0
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        
        # Performance
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Operations: {self.metrics['performance']['operations_completed']}")
        logger.info(f"  Total Time: {self.metrics['performance']['total_latency_ms']:.0f} ms")
        if self.metrics['performance']['operations_completed'] > 0:
            logger.info(f"  Avg Time: {self.metrics['performance']['avg_latency_ms']:.0f} ms")
        
        # Errors
        if self.metrics['errors']['total'] > 0:
            logger.info("")
            logger.info("Errors:")
            logger.info(f"  Total: {self.metrics['errors']['total']}")
            logger.info(f"  API Errors: {self.metrics['errors']['api_errors']}")
            logger.info(f"  Processing Errors: {self.metrics['errors']['processing_errors']}")
        
        # Resources
        logger.info("")
        logger.info("Resources:")
        logger.info(f"  Current Memory: {self.metrics['resource']['current_memory_mb']:.1f} MB")
        logger.info(f"  Peak Memory: {self.metrics['resource']['peak_memory_mb']:.1f} MB")
        logger.info("")

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
    """
    Send an email via Gmail with optional attachments and HTML formatting.
    Supports plain text, HTML emails, and file attachments.
    
    Two modes:
    1. show_preview=True: Returns preview WITHOUT sending (waits for confirmation)
    2. show_preview=False: Sends immediately (default)
    """
    try:
        # PREVIEW MODE: Show preview and wait for confirmation (don't send yet)
        if request.show_preview:
            canvas_display = {
                "canvas_type": "email_preview",
                "canvas_data": {
                    "to": request.to,
                    "cc": request.cc if request.cc else [],
                    "bcc": request.bcc if request.bcc else [],
                    "subject": request.subject,
                    "body": request.body,
                    "is_html": request.is_html,
                    "attachments": {
                        "file_ids": request.attachment_file_ids if request.attachment_file_ids else [],
                        "paths": request.attachment_paths if request.attachment_paths else [],
                        "count": len(request.attachment_file_ids or []) + len(request.attachment_paths or [])
                    }
                },
                "canvas_title": f"Email Preview: {request.subject}",
                "requires_confirmation": True,
                "confirmation_message": "Review and confirm to send this email"
            }
            
            logger.info(f"📊 Email preview generated (NOT sent yet): {request.subject}")
            
            # Return preview without sending
            return GmailResponse(
                success=True,
                result={
                    "status": "preview_ready",
                    "message": "Email preview ready. Waiting for confirmation to send.",
                    "preview_data": {
                        "to": request.to,
                        "subject": request.subject,
                        "body_preview": request.body[:100] + "..." if len(request.body) > 100 else request.body
                    }
                },
                canvas_display=canvas_display
            )
        
        # SEND MODE: Send email immediately (no preview)
        logger.info(f"📧 Sending email immediately: {request.subject}")
        
        # Check if attachments are requested
        has_attachments = bool(request.attachment_file_ids or request.attachment_paths)
        
        if has_attachments or request.is_html:
            # Use enhanced method for attachments or HTML
            result = await gmail_client.send_email_with_attachments(
                to=request.to,
                subject=request.subject,
                body=request.body,
                cc=request.cc if request.cc else None,
                bcc=request.bcc if request.bcc else None,
                is_html=request.is_html,
                attachment_file_ids=request.attachment_file_ids if request.attachment_file_ids else None,
                attachment_paths=request.attachment_paths if request.attachment_paths else None,
                user_id=request.user_id
            )
        else:
            # Use simple method for plain text without attachments
            params = {
                "recipient_email": request.to[0] if request.to else "",
                "subject": request.subject,
                "body": request.body,
                "user_id": request.user_id
            }
            
            if request.cc:
                params["cc"] = ",".join(request.cc)
            if request.bcc:
                params["bcc"] = ",".join(request.bcc)
            
            result = await gmail_client.call_tool("GMAIL_SEND_EMAIL", params)
        
        if result.get("success"):
            api_data = result.get("data", {})
            enhanced_result = {
                "status": "sent",
                "message_id": api_data.get("id") or api_data.get("messageId"),
                "thread_id": api_data.get("threadId"),
                "labels": api_data.get("labelIds", []),
                "sent_content": {
                    "to": request.to,
                    "cc": request.cc if request.cc else [],
                    "bcc": request.bcc if request.bcc else [],
                    "subject": request.subject,
                    "body": request.body[:500] + "..." if len(request.body) > 500 else request.body,
                    "is_html": request.is_html,
                    "attachments_count": result.get("attachments_sent", 0)
                }
            }
            logger.info(f"✅ Email sent successfully: {request.subject}")
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
    thread_id: Optional[str] = Field(None, description="Conversation thread ID for orchestrator integration")
    user_id: Optional[str] = Field(None, description="User ID for tracking")

@app.post("/download_attachments", response_model=GmailResponse)
async def download_attachments(request: DownloadAttachmentsRequest):
    """
    Download all attachments from an email using standardized file manager.
    Returns file paths that can be used by image/document agents for analysis.
    Files are saved to storage/gmail_attachments/ with persistent tracking.
    """
    try:
        result = await gmail_client.download_email_attachments(
            message_id=request.message_id,
            thread_id=request.thread_id,
            user_id=request.user_id
        )
        
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


# ============== STANDARDIZED FILE MANAGEMENT ENDPOINTS ==============

@app.get("/files", response_model=GmailResponse)
async def list_attachment_files(
    status: Optional[str] = None,
    thread_id: Optional[str] = None
):
    """
    List all attachment files managed by this agent.
    
    Parameters:
        status: Filter by status (active, expired, deleted)
        thread_id: Filter by conversation thread
    """
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = gmail_client.file_manager.list_files(
            status=file_status,
            thread_id=thread_id
        )
        
        return GmailResponse(success=True, result={
            "files": [f.to_orchestrator_format() for f in files],
            "count": len(files)
        })
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return GmailResponse(success=False, result=None, error=str(e))


@app.get("/files/{file_id}", response_model=GmailResponse)
async def get_attachment_file_info(file_id: str):
    """
    Get detailed information about a specific attachment file.
    """
    try:
        metadata = gmail_client.file_manager.get_file(file_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        return GmailResponse(success=True, result=metadata.to_orchestrator_format())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return GmailResponse(success=False, result=None, error=str(e))


@app.delete("/files/{file_id}", response_model=GmailResponse)
async def delete_attachment_file(file_id: str):
    """
    Delete an attachment file from the agent's storage.
    """
    try:
        success = gmail_client.file_manager.delete_file(file_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return GmailResponse(success=True, result={"message": f"File {file_id} deleted"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file: {e}")
        return GmailResponse(success=False, result=None, error=str(e))


@app.post("/cleanup", response_model=GmailResponse)
async def cleanup_attachment_files(max_age_hours: int = 72):
    """
    Clean up old/expired attachment files.
    
    Parameters:
        max_age_hours: Remove files older than this many hours (default: 72)
    """
    try:
        expired_count = gmail_client.file_manager.cleanup_expired()
        old_count = gmail_client.file_manager.cleanup_old(max_age_hours=max_age_hours)
        
        return GmailResponse(success=True, result={
            "expired_removed": expired_count,
            "old_removed": old_count
        })
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return GmailResponse(success=False, result=None, error=str(e))


@app.get("/stats", response_model=GmailResponse)
async def get_attachment_stats():
    """
    Get attachment file management statistics.
    """
    try:
        stats = gmail_client.file_manager.get_stats()
        return GmailResponse(success=True, result=stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return GmailResponse(success=False, result=None, error=str(e))


# ============== EMAIL COMPOSITION HELPERS ==============

class ComposeHtmlEmailRequest(BaseModel):
    """Request for composing HTML emails with templates"""
    to: list = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Main email content (will be formatted)")
    template: str = Field(default="basic", description="Template style: basic, professional, newsletter")
    header_text: Optional[str] = Field(None, description="Optional header text")
    footer_text: Optional[str] = Field(None, description="Optional footer text")
    cc: list = Field(default=[], description="CC email addresses")
    bcc: list = Field(default=[], description="BCC email addresses")
    attachment_file_ids: list = Field(default=[], description="File IDs to attach")
    attachment_paths: list = Field(default=[], description="File paths to attach")

@app.post("/compose_html_email", response_model=GmailResponse)
async def compose_html_email(request: ComposeHtmlEmailRequest):
    """
    Compose and send a professionally formatted HTML email using templates.
    Supports multiple templates and automatic formatting.
    """
    try:
        # Generate HTML based on template
        if request.template == "professional":
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-bottom: 3px solid #007bff; }}
                    .content {{ padding: 30px 20px; background-color: #ffffff; }}
                    .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                    h1 {{ color: #007bff; margin: 0; }}
                    p {{ margin: 15px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    {f'<div class="header"><h1>{request.header_text}</h1></div>' if request.header_text else ''}
                    <div class="content">
                        {request.content.replace(chr(10), '<br>')}
                    </div>
                    {f'<div class="footer">{request.footer_text}</div>' if request.footer_text else ''}
                </div>
            </body>
            </html>
            """
        elif request.template == "newsletter":
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
                    .container {{ max-width: 650px; margin: 20px auto; background-color: #ffffff; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; }}
                    .content {{ padding: 40px 30px; }}
                    .footer {{ background-color: #333; color: #fff; padding: 20px; text-align: center; font-size: 12px; }}
                    h1 {{ margin: 0; font-size: 28px; }}
                    p {{ line-height: 1.8; color: #555; }}
                </style>
            </head>
            <body>
                <div class="container">
                    {f'<div class="header"><h1>{request.header_text}</h1></div>' if request.header_text else ''}
                    <div class="content">
                        {request.content.replace(chr(10), '<br>')}
                    </div>
                    {f'<div class="footer">{request.footer_text}</div>' if request.footer_text else ''}
                </div>
            </body>
            </html>
            """
        else:  # basic template
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; padding: 20px; }}
                    .content {{ max-width: 600px; margin: 0 auto; }}
                    p {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <div class="content">
                    {f'<h2>{request.header_text}</h2>' if request.header_text else ''}
                    {request.content.replace(chr(10), '<br>')}
                    {f'<hr><p style="font-size: 12px; color: #666;">{request.footer_text}</p>' if request.footer_text else ''}
                </div>
            </body>
            </html>
            """
        
        # Send the HTML email
        result = await gmail_client.send_email_with_attachments(
            to=request.to,
            subject=request.subject,
            body=html_body,
            cc=request.cc if request.cc else None,
            bcc=request.bcc if request.bcc else None,
            is_html=True,
            attachment_file_ids=request.attachment_file_ids if request.attachment_file_ids else None,
            attachment_paths=request.attachment_paths if request.attachment_paths else None
        )
        
        if result.get("success"):
            api_data = result.get("data", {})
            return GmailResponse(success=True, result={
                "status": "sent",
                "message_id": api_data.get("id") or api_data.get("messageId"),
                "template_used": request.template,
                "is_html": True,
                "attachments_count": result.get("attachments_sent", 0),
                "sent_to": request.to
            })
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Compose HTML email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class FormatEmailRequest(BaseModel):
    """Request for formatting email content"""
    content: str = Field(..., description="Raw email content")
    format_type: str = Field(default="markdown", description="Format type: markdown, bullet_points, numbered_list")

@app.post("/format_email_content", response_model=GmailResponse)
async def format_email_content(request: FormatEmailRequest):
    """
    Format email content with various styles (markdown to HTML, bullet points, etc.).
    Returns formatted HTML that can be used in send_email with is_html=True.
    """
    try:
        content = request.content
        
        if request.format_type == "markdown":
            # Simple markdown to HTML conversion
            # Bold: **text** or __text__
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'__(.+?)__', r'<strong>\1</strong>', content)
            # Italic: *text* or _text_
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            content = re.sub(r'_(.+?)_', r'<em>\1</em>', content)
            # Links: [text](url)
            content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', content)
            # Headers: # Header
            content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
            content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
            content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
            # Line breaks
            content = content.replace('\n\n', '</p><p>')
            content = f'<p>{content}</p>'
            
        elif request.format_type == "bullet_points":
            # Convert lines to bullet points
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '<ul>' + ''.join([f'<li>{line}</li>' for line in lines]) + '</ul>'
            
        elif request.format_type == "numbered_list":
            # Convert lines to numbered list
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '<ol>' + ''.join([f'<li>{line}</li>' for line in lines]) + '</ol>'
        
        return GmailResponse(success=True, result={
            "formatted_content": content,
            "format_type": request.format_type,
            "usage": "Use this HTML content in send_email with is_html=True"
        })
        
    except Exception as e:
        logger.error(f"Format email content failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/email_templates", response_model=GmailResponse)
async def list_email_templates():
    """
    List available email templates and their descriptions.
    """
    templates = {
        "basic": {
            "name": "Basic",
            "description": "Simple, clean email format with optional header and footer",
            "use_case": "General purpose emails, quick messages"
        },
        "professional": {
            "name": "Professional",
            "description": "Business-style email with blue header and structured layout",
            "use_case": "Business communications, formal emails, client correspondence"
        },
        "newsletter": {
            "name": "Newsletter",
            "description": "Eye-catching gradient header, perfect for announcements",
            "use_case": "Newsletters, announcements, marketing emails"
        }
    }
    
    return GmailResponse(success=True, result={
        "templates": templates,
        "usage": "Use template name in compose_html_email endpoint"
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MAIL_AGENT_PORT", 8040))
    logger.info(f"Starting Mail Agent on port {port}...")
    uvicorn.run("mail_agent:app", host="0.0.0.0", port=port, reload=False)

# agents/mail_agent/client.py
import asyncio
import base64
import re
import time
import logging
import psutil
import httpx
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import COMPOSIO_API_KEY, MCP_URL, CONNECTION_ID, logger
from .llm import llm_client

# Standardized file manager import
try:
    from agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    try:
        from utils.agent_file_manager import AgentFileManager, FileType, FileStatus
    except ImportError:
        logger.error("Failed to import agent_file_manager from any location")
        raise

class GmailClient:
    """Manages connection to Composio Gmail API"""
    
    def __init__(self):
        self.mcp_url = MCP_URL
        self.api_key = COMPOSIO_API_KEY
        # Update to V3 Connection ID (Found via inspection: ca_xZUTNToOnUiQ)
        # The V2 UUID in env var is deprecated for this SDK version.
        self.connection_id = "ca_xZUTNToOnUiQ" 
        self.attachments_dir = Path("storage/mail_agent/gmail_attachments")
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
        Handles both direct base64 data and external S3 URLs.
        """
        try:
            from composio import Composio
            
            client = Composio(api_key=self.api_key)
            
            result = client.tools.execute(
                slug="GMAIL_GET_ATTACHMENT",
                arguments={
                    "message_id": message_id,
                    "attachment_id": attachment_id,
                    "file_name": filename,
                    "user_id": user_id or "me"
                },
                connected_account_id=self.connection_id,
                user_id="default",
                dangerously_skip_version_check=True
            )
            logger.warning(f"DEBUG: Raw SDK Result: {result}")
            if result.get("successful") or result.get("successfull"):
                data = result.get("data", {})
                logger.warning(f"DEBUG: Attachment Response Data Keys: {list(data.keys())}")
                if "file" in data: logger.warning(f"DEBUG: File info present: {data['file']}")
                if "attachment_data" in data: logger.warning(f"DEBUG: Base64 data length: {len(data['attachment_data'])}")
                
                # Check for various response formats
                attachment_data = data.get("attachment_data", "")
                decoded = None

                file_info = data.get("file")
                if file_info:
                    if isinstance(file_info, dict):
                        # Handle S3 URL format
                        url = file_info.get("s3url") or file_info.get("url")
                        if url:
                            logger.info(f"Downloading attachment from URL: {url}")
                            async with httpx.AsyncClient() as hclient:
                                response = await hclient.get(url)
                                if response.status_code == 200:
                                    decoded = response.content
                                else:
                                    logger.error(f"Failed to download from URL: {response.status_code}")
                    elif isinstance(file_info, str):
                        # Handle local file path (Composio sometimes saves to disk)
                        try:
                            logger.info(f"Reading attachment from local path: {file_info}")
                            path = Path(file_info)
                            if path.exists():
                                decoded = path.read_bytes()
                            else:
                                logger.error(f"Local attachment file not found: {file_info}")
                        except Exception as e:
                            logger.error(f"Failed to read local attachment file: {e}")
                
                if not decoded and attachment_data:
                    # Handle direct base64 format
                    decoded = base64.urlsafe_b64decode(attachment_data)
                
                if decoded:
                    # Create safe filename
                    safe_filename = re.sub(r'[^\w\s.-]', '_', filename)
                    
                    # Store s3url in metadata for later forwarding
                    s3_url = None
                    if file_info and isinstance(file_info, dict):
                        s3_url = file_info.get("s3url") or file_info.get("url")

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
                            "source": "gmail",
                            "s3url": s3_url
                        },
                        tags=["attachment", "gmail", f"msg:{message_id}"]
                    )
                    
                    logger.info(f"Saved attachment via file manager: {metadata.file_id} ({filename})")
                    
                    # Track attachment download
                    self.metrics["attachments"]["downloaded"] += 1
                    file_size_mb = len(decoded) / (1024 * 1024)
                    self.metrics["attachments"]["total_size_mb"] += file_size_mb
                    
                    # Return orchestrator-compatible format
                    return metadata.to_orchestrator_format()
            else:
                 logger.error(f"DEBUG: Tool execution failed. Result keys: {list(result.keys())}, Result: {result}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to download attachment: {e}", exc_info=True)
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
                
                for msg in messages:  # Process all messages returned by the API
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
                # Check if it's raw Gmail API format (payload) or Composio V3 flat format
                payload = data.get("payload")
                
                if payload:
                    # Logic for raw Gmail API structure
                    clean_text = self._extract_clean_text(payload)
                    attachments = self._find_attachments(payload)
                    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
                    
                    return {
                        "id": data.get("id"),
                        "subject": headers.get("Subject", ""),
                        "from": headers.get("From", ""),
                        "to": headers.get("To", ""),
                        "date": headers.get("Date", ""),
                        "body": clean_text[:2000] if clean_text else "(No plain text body)",
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
                else:
                    # Logic for Composio V3 flattened structure
                    logger.info("Using Composio V3 flat structure parsing for message details")
                    attachments = []
                    for att in data.get("attachmentList", []):
                        attachments.append({
                            "filename": att.get("filename", "unknown"),
                            "size": att.get("size", 0),
                            "mime_type": att.get("mimeType", "application/octet-stream"),
                            "attachment_id": att.get("attachmentId")
                        })
                    
                    return {
                        "id": data.get("id") or data.get("messageId"),
                        "subject": data.get("subject", ""),
                        "from": data.get("sender", "") or data.get("from", ""),
                        "to": data.get("to", "") or data.get("recipient", ""),
                        "date": data.get("date") or data.get("messageTimestamp", ""),
                        "body": data.get("messageText", "")[:2000],
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
        attachment_path: str = None,
        user_id: str = "me"
    ) -> Dict[str, Any]:
        """
        Send email using Composio's native parameters (no raw MIME).
        HTML emails work via is_html parameter.
        Note: Attachments require S3 upload (not implemented yet).
        """
        try:
            from composio import Composio
            
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
            if cc: params["cc"] = cc
            if bcc: params["bcc"] = bcc
            
            # ATTENTION: Native attachment sending is currently blocked by Composio SDK limitations 
            # (Upload actions are deprecated, R2 URLs rejected as s3key, Base64 content rejected).
            # FALLBACK STRATEGY: Append S3 URLs to email body so recipient can still access files.
            
            attachment_links = []
            if attachment_file_ids:
                for file_id in attachment_file_ids:
                    # Fix: Use get_file (sync)
                    file_info = self.file_manager.get_file(file_id)
                    if file_info:
                        meta = file_info.custom_metadata
                        s3_url = meta.get("s3url")
                        
                        if s3_url:
                            # Append to links
                            attachment_links.append(f"[{file_info.original_name}]({s3_url})")
                            logger.info(f"Added link fallback for attachment: {file_info.original_name}")
                        else:
                             # Try to check if we can upload? No, we validated we can't.
                             logger.warning(f"Attachment {file_info.original_name} has no S3 URL and cannot be uploaded. Skipping.")

            if attachment_links:
                if is_html:
                    params["body"] += "<br><br><b>Attachments:</b><br>" + "<br>".join(attachment_links)
                else:
                    params["body"] += "\n\nAttachments:\n" + "\n".join(attachment_links)
            
            # Handling Native Upload
            # Try using 'attachment' (singular) key for local file to trigger SDK auto-upload
            if attachment_path:
                 params["attachment"] = attachment_path # Try singular
            elif attachment_paths and len(attachment_paths) > 0:
                 # Fallback if list provided: use first one as singular 'attachment'
                 # assuming tool only supports one native upload or key is singular
                 params["attachment"] = attachment_paths[0]
            
            # Send via Composio using native parameters
            client = Composio(api_key=self.api_key)
            
            # DEBUG: Write params to file
            try:
                with open("last_send_params.json", "w") as f:
                    json.dump(params, f, indent=2)
            except:
                pass
                
            logger.info(f"Sending email (attachment count: {len(attachment_file_ids or [])}) params keys: {list(params.keys())}")
            
            # Use client.tools.execute for native file handling + version skip
            result = client.tools.execute(
                slug="GMAIL_SEND_EMAIL",
                arguments=params,
                connected_account_id=self.connection_id,
                user_id="default",
                dangerously_skip_version_check=True
            )
            
            logger.info(f"Email send result: {result}")
            
            if result.get("successful") or result.get("successfull"):
                self.metrics["emails"]["sent"] += 1
                self.metrics["emails"]["total_operations"] += 1
                return {
                    "success": True,
                    "data": result.get("data", {}),
                    "attachments_sent": 1 if attachment_links else 0
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a Gmail tool via Composio Python SDK.
        """
        call_start = time.time()
        operation_type = tool_name.replace("GMAIL_", "").lower()
        
        try:
            # Use Composio Python SDK
            from composio import Composio
            
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
            
            # Execute via tools.execute (New SDK)
            try:
                result = client.tools.execute(
                    slug=tool_name,
                    arguments=parameters,
                    connected_account_id=self.connection_id if self.connection_id else None,
                    user_id="default",
                    dangerously_skip_version_check=True
                )
            except Exception as e:
                self.metrics["errors"]["total"] += 1
                self.metrics["errors"]["api_errors"] += 1
                self.metrics["api_calls"]["failed"] += 1
                return {
                    "success": False, 
                    "error": f"SDK Execution Failed: {str(e)}"
                }
            
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

    async def download_email_attachments(
        self, 
        message_id: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download all attachments from an email using standardized file manager.
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
            
            # Download each attachment
            downloaded_files = []
            for att_info in attachments:
                filename = att_info["filename"]
                att_id = attachment_ids.get(filename)
                
                if not att_id:
                    continue
                
                file_metadata = await self._download_attachment(
                    message_id=message_id,
                    attachment_id=att_id,
                    filename=filename,
                    mime_type=att_info.get("type", ""),
                    thread_id=thread_id,
                    user_id=user_id
                )
                
                if file_metadata:
                    file_metadata["size"] = att_info.get("size", 0)
                    file_metadata["source"] = "gmail_attachment"
                    downloaded_files.append(file_metadata)
            
            return {
                "success": True,
                "message": f"Downloaded {len(downloaded_files)} attachment(s).",
                "files": downloaded_files,
                "instructions": "These files can now be analyzed."
            }
            
        except Exception as e:
            logger.error(f"Failed to download attachments: {e}")
            return {"success": False, "error": str(e)}

    async def semantic_search(
        self,
        vague_query: str,
        max_results: int = 10,
        user_id: str = "me"
    ) -> Dict[str, Any]:
        """
        Perform semantic search by generating a single optimized query with multiple terms.
        """
        try:
            logger.info(f"🔎 Starting optimized semantic search for: {vague_query}")
            
            # 1. Generate optimized combined query using LLM
            optimized_query = await llm_client.generate_optimized_query(vague_query)
            
            # 2. Perform a single search
            params = {
                "query": optimized_query,
                "max_results": max_results,
                "user_id": user_id
            }
            
            result = await self.call_tool("GMAIL_FETCH_EMAILS", params)
            
            if result.get("success"):
                data = result.get("data", {})
                messages = data.get("messages", [])
                
                logger.info(f"✨ Semantic search found {len(messages)} unique messages using query: {optimized_query}")
                
                return {
                    "success": True,
                    "data": {
                        "messages": messages,
                        "count": len(messages),
                        "query_used": optimized_query,
                        "original_query": vague_query
                    }
                }
            else:
                return {"success": False, "error": result.get("error")}
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"success": False, "error": str(e)}

    async def summarize_email(
        self,
        message_id: str,
        user_id: str = "me"
    ) -> Dict[str, Any]:
        """
        Fetch an email and generate a concise summary.
        """
        try:
            logger.info(f"📝 Summarizing email: {message_id}")
            
            # 1. Fetch full message details
            result = await self.call_tool(
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
                {"message_id": message_id, "format": "full", "user_id": user_id}
            )
            
            if not result.get("success"):
                return result
            
            data = result.get("data", {})
            body = data.get("body", "")
            subject = data.get("subject", "")
            
            if not body:
                return {"success": False, "error": "Email has no readable body content to summarize."}
            
            # 2. Generate summary using LLM
            summary = await llm_client.summarize_email_content(body)
            
            return {
                "success": True,
                "data": {
                    "message_id": message_id,
                    "subject": subject,
                    "summary": summary
                }
            }
            
        except Exception as e:
            logger.error(f"Summarization failed for message {message_id}: {e}")
            return {"success": False, "error": str(e)}

    async def batch_fetch_emails(
        self,
        message_ids: List[str],
        user_id: str = "me"
    ) -> List[Dict[str, str]]:
        """
        Fetch full content for a list of message IDs efficiently.
        Returns a list of dicts with {"id": ..., "subject": ..., "body": ...}
        """
        results = []
        logger.info(f"📦 Batch fetching {len(message_ids)} emails...")
        
        # Parallelize fetches with a semaphore to avoid rate limits (e.g. 10 concurrent)
        semaphore = asyncio.Semaphore(10)
        
        async def fetch_one(msg_id):
            async with semaphore:
                try:
                    res = await self.call_tool(
                        "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID",
                        {"message_id": msg_id, "format": "full", "user_id": user_id}
                    )
                    if res.get("success"):
                        data = res.get("data", {})
                        body = data.get("body", "")
                        if not body and data.get("snippet"):
                             body = data.get("snippet")
                        
                        return {
                            "id": msg_id,
                            "subject": data.get("subject", "No Subject"),
                            "body": body,
                            "from": data.get("from", "Unknown")
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch {msg_id}: {e}")
                return None

        # Gather all fetches
        tasks = [fetch_one(mid) for mid in message_ids]
        fetched_data = await asyncio.gather(*tasks)
        
        # Filter successful results
        results = [f for f in fetched_data if f is not None]
        logger.info(f"✅ Successfully fetched {len(results)}/{len(message_ids)} emails")
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive mail agent metrics."""
        uptime_seconds = time.time() - self._metrics_start_time
        process = psutil.Process()
        self.metrics["resource"]["current_memory_mb"] = process.memory_info().rss / 1024 / 1024
        
        return {
            "uptime_seconds": uptime_seconds,
            "emails": self.metrics["emails"].copy(),
            "attachments": self.metrics["attachments"].copy(),
            "api_calls": self.metrics["api_calls"].copy(),
            "performance": self.metrics["performance"].copy(),
            "errors": self.metrics["errors"].copy(),
            "resource": self.metrics["resource"].copy()
        }

# Global client instance
gmail_client = GmailClient()

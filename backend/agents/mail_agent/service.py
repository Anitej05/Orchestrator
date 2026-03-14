# agents/mail_agent/service.py
import asyncio
import base64
import logging
from typing import Dict, Any, Optional, List

from .config import ATTACHMENT_DIR, MAX_CONCURRENT_FETCHES
from .tools import ComposioToolManager
from .llm import llm_client
from .memory import agent_memory

logger = logging.getLogger("mail_agent")

class GmailService:
    """
    Core business logic for Gmail Agent.
    Uses Composio tools directly with per-user authentication.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize Gmail service for a specific user.
        
        Args:
            user_id: User ID to get Gmail connection for
        """
        self.user_id = user_id
        self.tool_mgr = ComposioToolManager(user_id)
        self.llm = llm_client
        self.memory = agent_memory
        
        logger.info(f"[GmailService] Initialized for user {user_id}")

    @property
    def tools(self) -> ComposioToolManager:
        """Alias for tool_mgr — used by base_agent_impl capabilities."""
        return self.tool_mgr

    # === Email Operations ===
    
    async def search_emails(
        self,
        query: str,
        max_results: int = 10,
        include_payload: bool = False,
        use_llm_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Search emails with optional LLM query optimization.
        
        Args:
            query: Search query (natural language or Gmail syntax)
            max_results: Maximum number of results
            include_payload: Include full email body
            use_llm_optimization: Use LLM to optimize vague queries
        """
        try:
            # Optimize query with LLM if enabled
            optimized_query = query
            if use_llm_optimization and len(query.split()) > 2:
                optimized_query = await self.llm.generate_optimized_query(query)
                logger.info(f"[Search] Optimized '{query}' → '{optimized_query}'")
            
            # Execute search
            result = await self.tool_mgr.fetch_emails(
                query=optimized_query,
                max_results=max_results,
                include_payload=include_payload
            )
            
            if result["success"]:
                messages = result["data"].get("messages", [])
                message_ids = [msg.get("id") for msg in messages]
                
                # Save to memory for follow-up actions
                self.memory.save_search_results(self.user_id, message_ids)
                
                return {
                    "success": True,
                    "messages": messages,
                    "total_count": len(messages),
                    "query_used": optimized_query
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[Search] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_email(self, message_id: str) -> Dict[str, Any]:
        """Get single email by ID"""
        try:
            result = await self.tool_mgr.fetch_message_by_id(message_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[GetEmail] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        is_html: bool = False,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Send email"""
        try:
            result = await self.tool_mgr.send_email(
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
                is_html=is_html,
                attachments=attachments
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Email sent successfully",
                    "data": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[SendEmail] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def reply_to_email(
        self,
        thread_id: str,
        message_id: str,
        body: str,
        to: str,
        cc: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Reply to an email"""
        try:
            result = await self.tool_mgr.reply_to_thread(
                thread_id=thread_id,
                body=body,
                to=to,
                cc=cc
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Reply sent successfully",
                    "data": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[ReplyEmail] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_email(self, message_id: str, permanent: bool = False) -> Dict[str, Any]:
        """Delete email (trash or permanent)"""
        try:
            if permanent:
                result = await self.tool_mgr.delete_message(message_id)
            else:
                result = await self.tool_mgr.move_to_trash(message_id)
            
            if result["success"]:
                action = "permanently deleted" if permanent else "moved to trash"
                return {
                    "success": True,
                    "message": f"Email {action}",
                    "data": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[DeleteEmail] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Draft Operations ===
    
    async def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create email draft"""
        try:
            result = await self.tool_mgr.create_draft(
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                thread_id=thread_id
            )
            
            if result["success"]:
                draft_data = result["data"]
                draft_id = draft_data.get("id")
                
                # Save context to memory
                if draft_id:
                    self.memory.save_draft_context(self.user_id, draft_id, {
                        "to": to,
                        "subject": subject,
                        "thread_id": thread_id,
                        "created_at": draft_data.get("created_at")
                    })
                
                return {
                    "success": True,
                    "message": "Draft created",
                    "draft": draft_data
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[CreateDraft] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_drafts(self, max_results: int = 10) -> Dict[str, Any]:
        """List email drafts"""
        try:
            result = await self.tool_mgr.list_drafts(max_results)
            
            if result["success"]:
                return {
                    "success": True,
                    "drafts": result["data"].get("drafts", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[ListDrafts] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_draft(self, draft_id: str) -> Dict[str, Any]:
        """Send an existing draft"""
        try:
            result = await self.tool_mgr.send_draft(draft_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Draft sent successfully",
                    "data": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[SendDraft] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_draft(self, draft_id: str) -> Dict[str, Any]:
        """Delete a draft"""
        try:
            result = await self.tool_mgr.delete_draft(draft_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Draft deleted"
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[DeleteDraft] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Label Operations ===
    
    async def add_labels(
        self,
        message_id: str,
        label_ids: List[str]
    ) -> Dict[str, Any]:
        """Add labels to email"""
        try:
            result = await self.tool_mgr.add_label_to_email(message_id, label_ids)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Labels added",
                    "data": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[AddLabels] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_labels(self) -> Dict[str, Any]:
        """List all Gmail labels"""
        try:
            result = await self.tool_mgr.list_labels()
            
            if result["success"]:
                return {
                    "success": True,
                    "labels": result["data"].get("labels", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[ListLabels] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_label(self, name: str) -> Dict[str, Any]:
        """Create a new label"""
        try:
            result = await self.tool_mgr.create_label(name)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Label created",
                    "label": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[CreateLabel] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Attachment Operations ===
    
    async def download_attachment(
        self,
        message_id: str,
        attachment_id: str,
        file_name: str
    ) -> Dict[str, Any]:
        """Download attachment"""
        try:
            result = await self.tool_mgr.get_attachment(message_id, attachment_id)
            
            if result["success"]:
                # Save to storage
                user_dir = ATTACHMENT_DIR / self.user_id
                user_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = user_dir / file_name
                
                # Decode and save
                attachment_data = result["data"].get("data")
                if attachment_data:
                    file_content = base64.b64decode(attachment_data)
                    file_path.write_bytes(file_content)
                    
                    return {
                        "success": True,
                        "message": "Attachment downloaded",
                        "file_path": str(file_path),
                        "file_name": file_name
                    }
                else:
                    return {"success": False, "error": "No attachment data"}
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[DownloadAttachment] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def download_all_attachments(self, message_id: str) -> Dict[str, Any]:
        """Download all attachments from an email"""
        try:
            # Get email first
            email_result = await self.get_email(message_id)
            if not email_result["success"]:
                return {"success": False, "error": "Failed to fetch email"}
            
            message = email_result["message"]
            attachments = message.get("attachments", [])
            
            if not attachments:
                return {"success": True, "message": "No attachments found", "files": []}
            
            # Download all concurrently
            download_tasks = [
                self.download_attachment(
                    message_id=message_id,
                    attachment_id=att["id"],
                    file_name=att.get("filename", f"attachment_{att['id']}")
                )
                for att in attachments
            ]
            
            results = await asyncio.gather(*download_tasks)
            successful = [r for r in results if r["success"]]
            
            return {
                "success": True,
                "message": f"Downloaded {len(successful)}/{len(attachments)} attachments",
                "files": successful
            }
            
        except Exception as e:
            logger.error(f"[DownloadAllAttachments] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === LLM-Enhanced Operations ===
    
    async def summarize_emails(
        self,
        message_ids: List[str]
    ) -> Dict[str, Any]:
        """Summarize multiple emails using LLM"""
        try:
            # Fetch emails concurrently
            fetch_tasks = [self.get_email(msg_id) for msg_id in message_ids[:MAX_CONCURRENT_FETCHES]]
            results = await asyncio.gather(*fetch_tasks)
            
            # Extract successful emails
            emails = []
            for result in results:
                if result["success"]:
                    message = result["message"]
                    body = message.get("body", message.get("snippet", ""))
                    subject = message.get("subject", "No subject")
                    emails.append(f"Subject: {subject}\n\n{body}")
            
            if not emails:
                return {"success": False, "error": "No emails to summarize"}
            
            # Generate summary
            summary = await self.llm.summarize_text_batch(emails)
            
            return {
                "success": True,
                "summary": summary,
                "emails_summarized": len(emails)
            }
            
        except Exception as e:
            logger.error(f"[SummarizeEmails] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def draft_smart_reply(
        self,
        message_id: str,
        user_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate AI reply draft"""
        try:
            # Get original email
            email_result = await self.get_email(message_id)
            if not email_result["success"]:
                return {"success": False, "error": "Failed to fetch email"}
            
            message = email_result["message"]
            body = message.get("body", message.get("snippet", ""))
            subject = message.get("subject", "")
            sender = message.get("from", "Sender")
            
            # Generate reply with LLM
            intent = user_instructions or "Reply professionally"
            draft_data = await self.llm.draft_email_reply(body, intent, sender)
            
            # Create draft in Gmail
            to_email = sender
            draft_result = await self.create_draft(
                to=to_email,
                subject=draft_data.get("subject", f"Re: {subject}"),
                body=draft_data.get("body", ""),
                thread_id=message.get("thread_id")
            )
            
            if draft_result["success"]:
                return {
                    "success": True,
                    "message": "Smart reply draft created",
                    "draft": draft_result["draft"],
                    "generated_content": draft_data
                }
            else:
                return {"success": False, "error": draft_result["error"]}
                
        except Exception as e:
            logger.error(f"[DraftSmartReply] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_action_items(
        self,
        message_ids: List[str]
    ) -> Dict[str, Any]:
        """Extract action items from emails using LLM"""
        try:
            # Fetch emails concurrently
            fetch_tasks = [self.get_email(msg_id) for msg_id in message_ids[:MAX_CONCURRENT_FETCHES]]
            results = await asyncio.gather(*fetch_tasks)
            
            # Extract email texts
            email_texts = []
            email_subjects = {}
            for result in results:
                if result["success"]:
                    message = result["message"]
                    body = message.get("body", message.get("snippet", ""))
                    subject = message.get("subject", "No subject")
                    msg_id = message.get("id")
                    email_texts.append(body)
                    email_subjects[msg_id] = subject
            
            if not email_texts:
                return {"success": False, "error": "No emails to analyze"}
            
            # Extract actions
            actions = await self.llm.extract_actions(email_texts)
            
            # Group by email
            by_email = {}
            for action in actions:
                source = action.get("source", "Unknown")
                if source not in by_email:
                    by_email[source] = []
                by_email[source].append(action.get("description"))
            
            return {
                "success": True,
                "action_items": actions,
                "by_email": by_email,
                "total_actions": len(actions)
            }
            
        except Exception as e:
            logger.error(f"[ExtractActions] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Contact Operations ===
    
    async def list_contacts(self, max_results: int = 100) -> Dict[str, Any]:
        """List contacts"""
        try:
            result = await self.tool_mgr.get_contacts(max_results)
            
            if result["success"]:
                return {
                    "success": True,
                    "contacts": result["data"].get("contacts", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[ListContacts] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_contacts(self, query: str) -> Dict[str, Any]:
        """Search contacts"""
        try:
            result = await self.tool_mgr.search_people(query)
            
            if result["success"]:
                return {
                    "success": True,
                    "contacts": result["data"].get("people", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[SearchContacts] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Thread Operations ===
    
    async def list_threads(
        self,
        query: str = "",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """List email threads"""
        try:
            result = await self.tool_mgr.list_threads(query, max_results)
            
            if result["success"]:
                return {
                    "success": True,
                    "threads": result["data"].get("threads", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[ListThreads] Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Get all messages in a thread"""
        try:
            result = await self.tool_mgr.fetch_message_by_thread(thread_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "messages": result["data"].get("messages", [])
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[GetThread] Error: {e}")
            return {"success": False, "error": str(e)}
    
    # === Profile Operations ===
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get Gmail profile"""
        try:
            result = await self.tool_mgr.get_profile()
            
            if result["success"]:
                return {
                    "success": True,
                    "profile": result["data"]
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            logger.error(f"[GetProfile] Error: {e}")
            return {"success": False, "error": str(e)}

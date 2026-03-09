# agents/gmail_agent/tools.py
import os
import logging
from typing import Dict, Any, Optional, List
from composio import Composio, Action
import json

logger = logging.getLogger("gmail_agent")

class ComposioToolManager:
    """
    Wrapper for Composio SDK tool execution.
    Handles all 23 Gmail tools with per-user authentication.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize Composio tool manager for a specific user.
        
        Args:
            user_id: User ID to get Gmail connection for
            
        Raises:
            ValueError: If user doesn't have Gmail connected
        """
        from services.integrations.composio_auth import get_auth_manager
        
        # Verify user has Gmail connected
        auth_mgr = get_auth_manager()
        connection = auth_mgr.get_connection_for_agent(user_id, "gmail")
        if not connection:
            raise ValueError(f"User {user_id} not connected to Gmail")
        
        self.user_id = user_id
        # connection is a plain dict returned by get_connection_for_agent, not an object.
        # Use dict key access. connection_id isn't used in execute_action (which uses
        # entity_id instead) but we store it for logging/debugging purposes.
        self.connection_id = connection["connection_id"]
        self.composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
        
        logger.info(f"[ComposioToolManager] Initialized for user {user_id}")
    
    async def execute_tool(
        self,
        tool_slug: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a Composio Gmail tool.
        
        Args:
            tool_slug: Tool name (e.g., "GMAIL_FETCH_EMAILS")
            parameters: Tool parameters
            
        Returns:
            Dict with success, data, error
        """
        try:
            logger.info(f"[ComposioToolManager] Executing {tool_slug} for user {self.user_id}")
            logger.debug(f"[ComposioToolManager] Parameters: {json.dumps(parameters, indent=2)}")
            
            # Execute tool with user connection.
            # Composio SDK ≥0.7 API changes:
            #   - execute_action() removed → use composio.actions.execute()
            #   - Action subscript (Action["slug"]) removed → use getattr(Action, "slug")
            #   - connected_account must be passed explicitly (entity_id alone not enough)
            action_enum = getattr(Action, tool_slug)
            result = self.composio.actions.execute(
                action=action_enum,
                params=parameters,
                entity_id=self.user_id,           # entity_id = Clerk user_id
                connected_account=self.connection_id  # Composio connection UUID
            )
            
            logger.info(f"[ComposioToolManager] {tool_slug} completed successfully")
            
            return {
                "success": True,
                "data": result.get("data", result),
                "error": None
            }
            
        except Exception as e:
            logger.error(f"[ComposioToolManager] Tool execution failed: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }
    
    async def fetch_emails(
        self,
        query: str = "label:inbox",
        max_results: int = 10,
        include_payload: bool = False
    ) -> Dict[str, Any]:
        """Fetch emails using GMAIL_FETCH_EMAILS tool"""
        return await self.execute_tool("GMAIL_FETCH_EMAILS", {
            "query": query,
            "max_results": max_results,
            "include_payload": include_payload
        })
    
    async def fetch_message_by_id(self, message_id: str) -> Dict[str, Any]:
        """Fetch single email by message ID"""
        return await self.execute_tool("GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", {
            "message_id": message_id
        })
    
    async def fetch_message_by_thread(self, thread_id: str) -> Dict[str, Any]:
        """Fetch messages in a thread"""
        return await self.execute_tool("GMAIL_FETCH_MESSAGE_BY_THREAD_ID", {
            "thread_id": thread_id
        })
    
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
        """Send email using GMAIL_SEND_EMAIL tool"""
        params = {
            "to": to,
            "subject": subject,
            "body": body,
            "is_html": is_html
        }
        
        if cc:
            params["cc"] = cc
        if bcc:
            params["bcc"] = bcc
        if attachments:
            params["attachments"] = attachments
        
        return await self.execute_tool("GMAIL_SEND_EMAIL", params)
    
    async def reply_to_thread(
        self,
        thread_id: str,
        body: str,
        to: str,
        cc: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Reply to email thread"""
        params = {
            "thread_id": thread_id,
            "body": body,
            "to": to
        }
        if cc:
            params["cc"] = cc
        
        return await self.execute_tool("GMAIL_REPLY_TO_THREAD", params)
    
    async def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create email draft"""
        params = {
            "to": to,
            "subject": subject,
            "body": body
        }
        if cc:
            params["cc"] = cc
        if thread_id:
            params["thread_id"] = thread_id
        
        return await self.execute_tool("GMAIL_CREATE_EMAIL_DRAFT", params)
    
    async def list_drafts(self, max_results: int = 10) -> Dict[str, Any]:
        """List email drafts"""
        return await self.execute_tool("GMAIL_LIST_DRAFTS", {
            "max_results": max_results
        })
    
    async def send_draft(self, draft_id: str) -> Dict[str, Any]:
        """Send an existing draft"""
        return await self.execute_tool("GMAIL_SEND_DRAFT", {
            "draft_id": draft_id
        })
    
    async def delete_draft(self, draft_id: str) -> Dict[str, Any]:
        """Delete a draft"""
        return await self.execute_tool("GMAIL_DELETE_DRAFT", {
            "draft_id": draft_id
        })
    
    async def add_label_to_email(
        self,
        message_id: str,
        label_ids: List[str]
    ) -> Dict[str, Any]:
        """Add labels to email"""
        return await self.execute_tool("GMAIL_ADD_LABEL_TO_EMAIL", {
            "message_id": message_id,
            "label_ids": label_ids
        })
    
    async def modify_thread_labels(
        self,
        thread_id: str,
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Modify labels on a thread"""
        params = {"thread_id": thread_id}
        if add_label_ids:
            params["add_label_ids"] = add_label_ids
        if remove_label_ids:
            params["remove_label_ids"] = remove_label_ids
        
        return await self.execute_tool("GMAIL_MODIFY_THREAD_LABELS", params)
    
    async def list_labels(self) -> Dict[str, Any]:
        """List all Gmail labels"""
        return await self.execute_tool("GMAIL_LIST_LABELS", {})
    
    async def create_label(self, name: str) -> Dict[str, Any]:
        """Create a new label"""
        return await self.execute_tool("GMAIL_CREATE_LABEL", {
            "name": name
        })
    
    async def delete_message(self, message_id: str) -> Dict[str, Any]:
        """Permanently delete a message"""
        return await self.execute_tool("GMAIL_DELETE_MESSAGE", {
            "message_id": message_id
        })
    
    async def move_to_trash(self, message_id: str) -> Dict[str, Any]:
        """Move message to trash"""
        return await self.execute_tool("GMAIL_MOVE_TO_TRASH", {
            "message_id": message_id
        })
    
    async def get_attachment(
        self,
        message_id: str,
        attachment_id: str
    ) -> Dict[str, Any]:
        """Download attachment"""
        return await self.execute_tool("GMAIL_GET_ATTACHMENT", {
            "message_id": message_id,
            "attachment_id": attachment_id
        })
    
    async def list_threads(
        self,
        query: str = "",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """List email threads"""
        return await self.execute_tool("GMAIL_LIST_THREADS", {
            "query": query,
            "max_results": max_results
        })
    
    async def get_contacts(self, max_results: int = 100) -> Dict[str, Any]:
        """Get contacts list"""
        return await self.execute_tool("GMAIL_GET_CONTACTS", {
            "max_results": max_results
        })
    
    async def search_people(self, query: str) -> Dict[str, Any]:
        """Search contacts"""
        return await self.execute_tool("GMAIL_SEARCH_PEOPLE", {
            "query": query
        })
    
    async def get_profile(self) -> Dict[str, Any]:
        """Get Gmail profile"""
        return await self.execute_tool("GMAIL_GET_PROFILE", {})

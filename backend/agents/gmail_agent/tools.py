# agents/gmail_agent/tools.py
"""
Gmail Agent — Composio Tool Manager (v3 SDK)

Handles all 60 Gmail tools with per-user authentication.
Uses composio v0.10.x (v3 API):
    composio.tools.execute(slug, arguments, connected_account_id=...)

Provides:
  - Explicit wrappers for 30+ high-value Gmail tools
  - execute_any_tool() for dynamic access to ALL 60 Gmail tools
  - get_all_tools() to discover the full tool catalog at runtime
"""

import logging
import json
from typing import Dict, Any, Optional, List

from .config import COMPOSIO_API_KEY

logger = logging.getLogger("gmail_agent")


class ComposioToolManager:
    """
    Wrapper for Composio SDK tool execution (v3).
    Handles all 60 Gmail tools with per-user authentication.
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
        self.connection_id = connection["connection_id"]

        from composio import Composio
        self.composio = Composio(api_key=COMPOSIO_API_KEY)

        logger.info(f"[ComposioToolManager] v3 init for user {user_id}")

    # ------------------------------------------------------------------
    # Core execution (v3 SDK)
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_slug: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a Composio Gmail tool via v3 SDK.

        Args:
            tool_slug: Tool slug (e.g., "GMAIL_SEND_EMAIL")
            parameters: Tool parameters

        Returns:
            Dict with success, data, error
        """
        try:
            logger.info(f"[Tools] Executing {tool_slug} for user {self.user_id}")
            logger.debug(f"[Tools] Parameters: {json.dumps(parameters, indent=2)}")

            result = self.composio.tools.execute(
                slug=tool_slug,
                arguments=parameters or {},
                connected_account_id=self.connection_id,
            )

            # Handle ToolExecutionResponse
            if hasattr(result, "model_dump"):
                result = result.model_dump()

            if isinstance(result, dict):
                success = result.get("successful", result.get("error") is None)
                return {
                    "success": success,
                    "data": result.get("data", result),
                    "error": result.get("error") if not success else None,
                }
            return {"success": True, "data": result, "error": None}

        except Exception as e:
            logger.error(f"[Tools] {tool_slug} failed: {e}")
            return {"success": False, "data": None, "error": str(e)}

    async def execute_any_tool(
        self,
        tool_slug: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute ANY Gmail tool by slug — dynamic access to all 60 tools.

        This is the universal entry point for tools that don't have
        explicit wrappers. The LLM can call this with any valid
        GMAIL_* slug discovered from get_all_tools().
        """
        return await self.execute_tool(tool_slug.upper(), parameters)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Fetch the full Gmail tool catalog from Composio (all 60 tools).

        Returns list of dicts with name, description, parameters.
        """
        try:
            raw_tools = self.composio.tools.get(
                user_id=self.user_id,
                toolkits=["gmail"],
            )
            results = []
            for t in raw_tools:
                if hasattr(t, "model_dump"):
                    results.append(t.model_dump())
                elif isinstance(t, dict):
                    results.append(t)
                else:
                    results.append({
                        "name": getattr(t, "name", getattr(t, "slug", str(t))),
                        "description": getattr(t, "description", ""),
                    })
            return results
        except Exception as e:
            logger.error(f"[Tools] Failed to get all tools: {e}")
            return []

    # ------------------------------------------------------------------
    # Email fetching & search
    # ------------------------------------------------------------------

    async def fetch_emails(
        self,
        query: str = "label:inbox",
        max_results: int = 10,
        include_payload: bool = False,
    ) -> Dict[str, Any]:
        """Fetch emails using GMAIL_FETCH_EMAILS tool."""
        return await self.execute_tool("GMAIL_FETCH_EMAILS", {
            "query": query,
            "max_results": max_results,
            "include_payload": include_payload,
        })

    async def fetch_message_by_id(self, message_id: str) -> Dict[str, Any]:
        """Fetch single email by message ID."""
        return await self.execute_tool("GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", {
            "message_id": message_id,
        })

    async def fetch_message_by_thread(self, thread_id: str) -> Dict[str, Any]:
        """Fetch messages in a thread."""
        return await self.execute_tool("GMAIL_FETCH_MESSAGE_BY_THREAD_ID", {
            "thread_id": thread_id,
        })

    async def list_messages(
        self,
        query: str = "",
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """List Gmail messages with optional query filter."""
        return await self.execute_tool("GMAIL_LIST_MESSAGES", {
            "query": query,
            "max_results": max_results,
        })

    async def list_history(self, start_history_id: str) -> Dict[str, Any]:
        """List mailbox history changes since a given history ID."""
        return await self.execute_tool("GMAIL_LIST_HISTORY", {
            "start_history_id": start_history_id,
        })

    # ------------------------------------------------------------------
    # Send, reply, forward
    # ------------------------------------------------------------------

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        is_html: bool = False,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send email using GMAIL_SEND_EMAIL tool."""
        params: Dict[str, Any] = {
            "to": to,
            "subject": subject,
            "body": body,
            "is_html": is_html,
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
        cc: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Reply to email thread."""
        params: Dict[str, Any] = {
            "thread_id": thread_id,
            "body": body,
            "recipient_email": to,
        }
        if cc:
            params["cc"] = cc
        return await self.execute_tool("GMAIL_REPLY_TO_THREAD", params)

    async def forward_message(
        self,
        message_id: str,
        to: str,
        additional_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward an email message to another recipient."""
        params: Dict[str, Any] = {
            "message_id": message_id,
            "to": to,
        }
        if additional_message:
            params["additional_message"] = additional_message
        return await self.execute_tool("GMAIL_FORWARD_MESSAGE", params)

    # ------------------------------------------------------------------
    # Draft management
    # ------------------------------------------------------------------

    async def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create email draft."""
        params: Dict[str, Any] = {"to": to, "subject": subject, "body": body}
        if cc:
            params["cc"] = cc
        if thread_id:
            params["thread_id"] = thread_id
        return await self.execute_tool("GMAIL_CREATE_EMAIL_DRAFT", params)

    async def get_draft(self, draft_id: str) -> Dict[str, Any]:
        """Get a single draft's full content."""
        return await self.execute_tool("GMAIL_GET_DRAFT", {"draft_id": draft_id})

    async def update_draft(
        self,
        draft_id: str,
        to: Optional[str] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing draft."""
        params: Dict[str, Any] = {"draft_id": draft_id}
        if to:
            params["to"] = to
        if subject:
            params["subject"] = subject
        if body:
            params["body"] = body
        return await self.execute_tool("GMAIL_UPDATE_DRAFT", params)

    async def list_drafts(self, max_results: int = 10) -> Dict[str, Any]:
        """List email drafts."""
        return await self.execute_tool("GMAIL_LIST_DRAFTS", {
            "max_results": max_results,
        })

    async def send_draft(self, draft_id: str) -> Dict[str, Any]:
        """Send an existing draft."""
        return await self.execute_tool("GMAIL_SEND_DRAFT", {"draft_id": draft_id})

    async def delete_draft(self, draft_id: str) -> Dict[str, Any]:
        """Delete a draft."""
        return await self.execute_tool("GMAIL_DELETE_DRAFT", {"draft_id": draft_id})

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    async def add_label_to_email(
        self,
        message_id: str,
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add/remove labels on a message."""
        params: Dict[str, Any] = {"message_id": message_id}
        if add_label_ids:
            params["add_label_ids"] = add_label_ids
        if remove_label_ids:
            params["remove_label_ids"] = remove_label_ids
        return await self.execute_tool("GMAIL_ADD_LABEL_TO_EMAIL", params)

    async def modify_thread_labels(
        self,
        thread_id: str,
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Modify labels on a thread."""
        params: Dict[str, Any] = {"thread_id": thread_id}
        if add_label_ids:
            params["add_label_ids"] = add_label_ids
        if remove_label_ids:
            params["remove_label_ids"] = remove_label_ids
        return await self.execute_tool("GMAIL_MODIFY_THREAD_LABELS", params)

    async def list_labels(self) -> Dict[str, Any]:
        """List all Gmail labels."""
        return await self.execute_tool("GMAIL_LIST_LABELS", {})

    async def get_label(self, label_id: str) -> Dict[str, Any]:
        """Get label details by ID."""
        return await self.execute_tool("GMAIL_GET_LABEL", {"label_id": label_id})

    async def create_label(self, name: str) -> Dict[str, Any]:
        """Create a new label."""
        return await self.execute_tool("GMAIL_CREATE_LABEL", {"name": name})

    async def delete_label(self, label_id: str) -> Dict[str, Any]:
        """Permanently delete a label."""
        return await self.execute_tool("GMAIL_DELETE_LABEL", {"label_id": label_id})

    async def patch_label(
        self,
        label_id: str,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch/rename a label."""
        params: Dict[str, Any] = {"label_id": label_id}
        if name:
            params["name"] = name
        return await self.execute_tool("GMAIL_PATCH_LABEL", params)

    # ------------------------------------------------------------------
    # Message management (trash, delete, batch)
    # ------------------------------------------------------------------

    async def delete_message(self, message_id: str) -> Dict[str, Any]:
        """Permanently delete a message."""
        return await self.execute_tool("GMAIL_DELETE_MESSAGE", {
            "message_id": message_id,
        })

    async def move_to_trash(self, message_id: str) -> Dict[str, Any]:
        """Move message to trash."""
        return await self.execute_tool("GMAIL_MOVE_TO_TRASH", {
            "message_id": message_id,
        })

    async def untrash_message(self, message_id: str) -> Dict[str, Any]:
        """Restore a message from trash."""
        return await self.execute_tool("GMAIL_UNTRASH_MESSAGE", {
            "message_id": message_id,
        })

    async def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        """Permanently delete an entire thread."""
        return await self.execute_tool("GMAIL_DELETE_THREAD", {
            "thread_id": thread_id,
        })

    async def trash_thread(self, thread_id: str) -> Dict[str, Any]:
        """Move an entire thread to trash."""
        return await self.execute_tool("GMAIL_TRASH_THREAD", {
            "thread_id": thread_id,
        })

    async def untrash_thread(self, thread_id: str) -> Dict[str, Any]:
        """Restore a thread from trash."""
        return await self.execute_tool("GMAIL_UNTRASH_THREAD", {
            "thread_id": thread_id,
        })

    async def batch_delete_messages(
        self,
        message_ids: List[str],
    ) -> Dict[str, Any]:
        """Batch delete multiple messages."""
        return await self.execute_tool("GMAIL_BATCH_DELETE_MESSAGES", {
            "ids": message_ids,
        })

    async def batch_modify_messages(
        self,
        message_ids: List[str],
        add_label_ids: Optional[List[str]] = None,
        remove_label_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Batch modify labels on multiple messages."""
        params: Dict[str, Any] = {"ids": message_ids}
        if add_label_ids:
            params["add_label_ids"] = add_label_ids
        if remove_label_ids:
            params["remove_label_ids"] = remove_label_ids
        return await self.execute_tool("GMAIL_BATCH_MODIFY_MESSAGES", params)

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    async def create_filter(
        self,
        criteria: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a Gmail filter rule."""
        return await self.execute_tool("GMAIL_CREATE_FILTER", {
            "criteria": criteria,
            "action": action,
        })

    async def list_filters(self) -> Dict[str, Any]:
        """List all Gmail filters."""
        return await self.execute_tool("GMAIL_LIST_FILTERS", {})

    async def get_filter(self, filter_id: str) -> Dict[str, Any]:
        """Get a filter by ID."""
        return await self.execute_tool("GMAIL_GET_FILTER", {"filter_id": filter_id})

    async def delete_filter(self, filter_id: str) -> Dict[str, Any]:
        """Delete a filter."""
        return await self.execute_tool("GMAIL_DELETE_FILTER", {"filter_id": filter_id})

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    async def list_threads(
        self,
        query: str = "",
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """List email threads."""
        return await self.execute_tool("GMAIL_LIST_THREADS", {
            "query": query,
            "max_results": max_results,
        })

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------

    async def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
    ) -> Dict[str, Any]:
        """Download attachment."""
        return await self.execute_tool("GMAIL_GET_ATTACHMENT", {
            "message_id": message_id,
            "attachment_id": attachment_id,
        })

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    async def get_contacts(self, max_results: int = 100) -> Dict[str, Any]:
        """Get contacts list."""
        return await self.execute_tool("GMAIL_GET_CONTACTS", {
            "max_results": max_results,
        })

    async def search_people(self, query: str) -> Dict[str, Any]:
        """Search contacts."""
        return await self.execute_tool("GMAIL_SEARCH_PEOPLE", {"query": query})

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def get_profile(self) -> Dict[str, Any]:
        """Get Gmail profile."""
        return await self.execute_tool("GMAIL_GET_PROFILE", {})

    async def get_vacation_settings(self) -> Dict[str, Any]:
        """Get vacation/auto-reply settings."""
        return await self.execute_tool("GMAIL_GET_VACATION_SETTINGS", {})

    async def update_vacation_settings(
        self,
        enable_auto_reply: bool,
        response_subject: Optional[str] = None,
        response_body_html: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update vacation/auto-reply settings."""
        params: Dict[str, Any] = {"enableAutoReply": enable_auto_reply}
        if response_subject:
            params["responseSubject"] = response_subject
        if response_body_html:
            params["responseBodyHtml"] = response_body_html
        return await self.execute_tool("GMAIL_UPDATE_VACATION_SETTINGS", params)

    async def get_auto_forwarding(self) -> Dict[str, Any]:
        """Get auto-forwarding settings."""
        return await self.execute_tool("GMAIL_GET_AUTO_FORWARDING", {})

    async def list_forwarding_addresses(self) -> Dict[str, Any]:
        """List all forwarding addresses."""
        return await self.execute_tool("GMAIL_LIST_FORWARDING_ADDRESSES", {})

    async def get_language_settings(self) -> Dict[str, Any]:
        """Get language settings."""
        return await self.execute_tool("GMAIL_GET_LANGUAGE_SETTINGS", {})

    async def list_send_as_aliases(self) -> Dict[str, Any]:
        """List send-as aliases."""
        return await self.execute_tool("GMAIL_LIST_SEND_AS_ALIASES", {})

    async def stop_watch_notifications(self) -> Dict[str, Any]:
        """Stop push notifications."""
        return await self.execute_tool("GMAIL_STOP_WATCH_NOTIFICATIONS", {})

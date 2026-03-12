"""
Gmail Agent - BaseAgent Implementation

Extends BaseAgent to provide Gmail operations with Composio tools.
Handles email search, send, reply, draft management, etc.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import AgentResponse, ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .config import logger
from .service import GmailService
from .memory import agent_memory

logging.getLogger("gmail_agent").setLevel(logging.INFO)


@dataclass
class GmailAgentConfig(AgentConfig):
    """Configuration for Gmail Agent."""
    max_concurrent_fetches: int = 5
    max_search_results: int = 50
    enable_memory: bool = True


class GmailAgent(BaseAgent):
    """
    Gmail Agent - Composio-native Gmail operations.
    
    Features:
    - Email search with LLM optimization
    - Send and reply emails
    - Draft management
    - Attachment handling
    - Email summarization
    - Action extraction
    - Label management
    """
    
    def __init__(
        self,
        agent_id: str = "gmail_agent",
        agent_name: str = "Gmail Agent",
        services: Optional[AgentServices] = None,
        config: Optional[GmailAgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services,
            config=config or GmailAgentConfig(),
        )
        
        # Gmail service (per-user, lazy-loaded)
        self._services: Dict[str, GmailService] = {}
        self.memory = agent_memory
        
        logger.info(f"GmailAgent initialized with config: {self.config}")

    def _get_prompt_guidance(self, request=None) -> str:
        """
        Gmail-specific rules for the shared BaseAgent prompts.

        Gmail access is already bound to the authenticated Orbimesh user via
        request.user_id / context.user_id and the active Composio connection.
        The model should not invent a separate "Gmail user ID" requirement for
        inbox reads or searches.
        """
        return (
            "- The Gmail account is already authenticated through the active "
            "Composio connection for this Orbimesh user.\n"
            "- Use the connected Gmail account by default for inbox reads, "
            "searches, drafts, labels, and settings.\n"
            "- Do NOT ask the user for their Gmail address, Gmail user ID, or "
            "confirmation that the connected account exists just to read or "
            "search their own inbox.\n"
            "- Only ask for an email address when the task specifically needs "
            "a recipient, sender filter, CC/BCC target, or another explicit "
            "email field required by the Gmail action."
        )
    
    def register_capabilities(self) -> None:
        """Register Gmail capabilities."""
        # Capabilities will be auto-discovered from decorated methods
        pass
    
    async def _initialize_resources(self):
        """Initialize Gmail service resources on startup."""
        logger.info("Initializing Gmail Agent resources...")
        # Gmail service is lazy-loaded per-user on demand via _get_service()
        # Stock initialization just prepares the agent state
        self.status = "initialized"
        logger.info("Gmail Agent resources initialized")
    
    async def _cleanup_resources(self):
        """Cleanup Gmail resources on shutdown."""
        logger.info("Cleaning up Gmail Agent resources...")
        # Close any cached service instances
        for user_id, service in self._services.items():
            logger.info(f"Closing Gmail service for user: {user_id}")
        self._services.clear()
        logger.info("Gmail Agent resources cleaned up")
    
    def _get_service(self, user_id: str) -> GmailService:
        """
        Get or create Gmail service for user.
        
        Args:
            user_id: User ID from ExecutionContext (required)
        """
        if user_id not in self._services:
            self._services[user_id] = GmailService(user_id)
        return self._services[user_id]
    
    def _extract_prompt(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from various parameter fields."""
        fields = ['prompt', 'query', 'instruction', 'p', 'q', 'content', 'message']
        for field in fields:
            if request.get(field):
                return str(request[field])
        return None
    
    @capability(
        name="search_emails",
        description="Search emails with natural language query",
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
            ParameterSchema(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=10,
            ),
        ]
    )
    async def search_emails(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Search emails capability."""
        try:
            # Get user_id from ExecutionContext (set by BaseAgent)
            user_id = context.user_id

            
            # Extract query from params
            query = params.get("query") or params.get("prompt", "")
            
            if not query:
                return AgentResponse.error("Query parameter required")
            
            service = self._get_service(user_id)
            result = await service.search_emails(
                query=query,
                max_results=params.get("max_results", 10)
            )
            
            if result.get("success"):
                return AgentResponse.success(
                    result={
                        "messages": result.get("messages", []),
                        "total_count": result.get("total_count", 0),
                        "query": result.get("query_used", query),
                    },
                    summary=f"Found {result.get('total_count', 0)} emails"
                )
            else:
                return AgentResponse.error(result.get("error", "Search failed"))
        except Exception as e:
            logger.error(f"[search_emails] Error: {e}")
            return AgentResponse.error(f"Search failed: {str(e)}")
    
    @capability(
        name="send_email",
        description="Send an email",
        parameters=[
            ParameterSchema(
                name="to",
                type="string",
                description="Recipient email",
                required=True,
            ),
            ParameterSchema(
                name="subject",
                type="string",
                description="Email subject",
                required=True,
            ),
            ParameterSchema(
                name="body",
                type="string",
                description="Email body",
                required=True,
            ),
            ParameterSchema(
                name="cc",
                type="string",
                description="CC recipients",
                required=False,
            ),
            ParameterSchema(
                name="bcc",
                type="string",
                description="BCC recipients",
                required=False,
            ),
        ]
    )
    async def send_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Send email capability."""
        try:
            # Get user_id from ExecutionContext (set by BaseAgent)
            user_id = context.user_id

            
            to = params.get("to")
            subject = params.get("subject")
            body = params.get("body")
            
            if not all([to, subject, body]):
                return AgentResponse.error(
                    "Missing required parameters: to, subject, body"
                )
            
            service = self._get_service(user_id)
            result = await service.send_email(
                to=to,
                subject=subject,
                body=body,
                cc=params.get("cc"),
                bcc=params.get("bcc")
            )
            
            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary="Email sent successfully"
                )
            else:
                return AgentResponse.error(result.get("error", "Send failed"))
        except Exception as e:
            logger.error(f"[send_email] Error: {e}")
            return AgentResponse.error(f"Send failed: {str(e)}")
    
    @capability(
        name="reply_email",
        description="Reply to an email",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Message ID of the email to reply to",
                required=True,
            ),
            ParameterSchema(
                name="body",
                type="string",
                description="Reply body text",
                required=True,
            ),
            ParameterSchema(
                name="to",
                type="string",
                description=(
                    "Recipient email address for the reply. "
                    "If omitted, defaults to empty string — caller should populate "
                    "from the original sender to avoid a Composio error."
                ),
                required=False,
                default="",
            ),
        ]
    )
    async def reply_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Reply to email capability."""
        try:
            # Get user_id from ExecutionContext (set by BaseAgent)
            user_id = context.user_id
            

            message_id = params.get("message_id")
            body = params.get("body")
            
            if not message_id or not body:
                return AgentResponse.error(
                    "Missing required parameters: message_id, body"
                )
            
            service = self._get_service(user_id)

            # Auto-fetch sender when `to` is not provided
            to = (params.get("to") or "").strip()
            if not to:
                import re as _re
                email_result = await service.get_email(message_id=message_id)
                if email_result.get("success"):
                    msg = email_result.get("message", {})
                    raw_sender = msg.get("sender") or msg.get("from", "")
                    m = _re.search(r"<([^>]+)>", str(raw_sender))
                    to = m.group(1) if m else str(raw_sender).strip()
                    logger.info(f"[reply_email] Auto-resolved sender: {to}")

            result = await service.reply_to_email(
                thread_id=message_id,
                message_id=message_id,
                body=body,
                to=to
            )
            
            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary="Reply sent successfully"
                )
            else:
                return AgentResponse.error(result.get("error", "Reply failed"))
        except Exception as e:
            logger.error(f"[reply_email] Error: {e}")
            return AgentResponse.error(f"Reply failed: {str(e)}")
    
    @capability(
        name="get_email",
        description="Get full email details",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Message ID",
                required=True,
            ),
        ]
    )
    async def get_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Get email capability."""
        try:
            # Get user_id from ExecutionContext (set by BaseAgent)
            user_id = context.user_id
            

            message_id = params.get("message_id")
            
            if not message_id:
                return AgentResponse.error("Missing message_id parameter")
            
            service = self._get_service(user_id)
            result = await service.get_email(message_id=message_id)
            
            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("message", {}),
                    summary="Email retrieved"
                )
            else:
                return AgentResponse.error(result.get("error", "Failed to retrieve email"))
        except Exception as e:
            logger.error(f"[get_email] Error: {e}")
            return AgentResponse.error(f"Failed to retrieve email: {str(e)}")

    @capability(
        name="summarize_emails",
        description="Summarize one or more emails using AI. Pass a list of message IDs to get a concise summary of each.",
        parameters=[
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs to summarize",
                required=True,
            ),
        ]
    )
    async def summarize_emails(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Summarize emails using AI."""
        try:
            user_id = context.user_id
            message_ids: List[str] = params.get("message_ids", [])
            if not message_ids:
                return AgentResponse.error("message_ids parameter is required")

            service = self._get_service(user_id)
            result = await service.summarize_emails(message_ids)

            if result.get("success"):
                return AgentResponse.success(
                    result={
                        "summary": result.get("summary", ""),
                        "emails_summarized": result.get("emails_summarized", 0),
                    },
                    summary=f"Summarized {result.get('emails_summarized', 0)} emails"
                )
            return AgentResponse.error(result.get("error", "Summarization failed"))
        except Exception as e:
            logger.error(f"[summarize_emails] Error: {e}")
            return AgentResponse.error(f"Summarization failed: {str(e)}")

    @capability(
        name="draft_smart_reply",
        description="Use AI to draft a reply to an email and save it as a Gmail draft.",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Message ID of the email to reply to",
                required=True,
            ),
            ParameterSchema(
                name="instructions",
                type="string",
                description="Instructions for tone/content of the AI-drafted reply (optional)",
                required=False,
            ),
        ]
    )
    async def draft_smart_reply(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Draft a smart AI reply and save as Gmail draft."""
        try:
            user_id = context.user_id
            message_id = params.get("message_id")
            if not message_id:
                return AgentResponse.error("message_id parameter is required")

            service = self._get_service(user_id)
            result = await service.draft_smart_reply(
                message_id=message_id,
                user_instructions=params.get("instructions"),
            )

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("draft", {}),
                    summary="AI draft reply created and saved to Gmail Drafts"
                )
            return AgentResponse.error(result.get("error", "Draft creation failed"))
        except Exception as e:
            logger.error(f"[draft_smart_reply] Error: {e}")
            return AgentResponse.error(f"Draft creation failed: {str(e)}")

    @capability(
        name="extract_action_items",
        description="Extract to-dos and action items from emails using AI.",
        parameters=[
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs to analyze for action items",
                required=True,
            ),
        ]
    )
    async def extract_action_items(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Extract action items from emails using AI."""
        try:
            user_id = context.user_id
            message_ids: List[str] = params.get("message_ids", [])
            if not message_ids:
                return AgentResponse.error("message_ids parameter is required")

            service = self._get_service(user_id)
            result = await service.extract_action_items(message_ids)

            if result.get("success"):
                return AgentResponse.success(
                    result={
                        "action_items": result.get("action_items", []),
                        "by_email": result.get("by_email", {}),
                        "total": result.get("total_actions", 0),
                    },
                    summary=f"Found {result.get('total_actions', 0)} action items"
                )
            return AgentResponse.error(result.get("error", "Extraction failed"))
        except Exception as e:
            logger.error(f"[extract_action_items] Error: {e}")
            return AgentResponse.error(f"Extraction failed: {str(e)}")

    # ------------------------------------------------------------------
    # NEW CAPABILITIES (v2 — covers remaining Composio Gmail tools)
    # ------------------------------------------------------------------

    @capability(
        name="forward_email",
        description="Forward an email to another recipient",
        parameters=[
            ParameterSchema(
                name="message_id",
                type="string",
                description="Message ID of the email to forward",
                required=True,
            ),
            ParameterSchema(
                name="to",
                type="string",
                description="Recipient email to forward to",
                required=True,
            ),
            ParameterSchema(
                name="additional_message",
                type="string",
                description="Optional message to include above the forwarded email",
                required=False,
            ),
        ]
    )
    async def forward_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Forward an email to another recipient."""
        try:
            user_id = context.user_id
            message_id = params.get("message_id")
            to = params.get("to")
            if not message_id or not to:
                return AgentResponse.error("Missing required: message_id, to")

            service = self._get_service(user_id)
            result = await service.tools.forward_message(
                message_id=message_id,
                to=to,
                additional_message=params.get("additional_message"),
            )
            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Email forwarded to {to}"
                )
            return AgentResponse.error(result.get("error", "Forward failed"))
        except Exception as e:
            logger.error(f"[forward_email] Error: {e}")
            return AgentResponse.error(f"Forward failed: {str(e)}")

    @capability(
        name="manage_drafts",
        description="Manage email drafts: get, update, list, send, or delete a draft",
        parameters=[
            ParameterSchema(
                name="action",
                type="string",
                description="Action: 'get', 'update', 'list', 'send', or 'delete'",
                required=True,
            ),
            ParameterSchema(
                name="draft_id",
                type="string",
                description="Draft ID (required for get/update/send/delete)",
                required=False,
            ),
            ParameterSchema(
                name="to",
                type="string",
                description="Recipient (for update)",
                required=False,
            ),
            ParameterSchema(
                name="subject",
                type="string",
                description="Subject (for update)",
                required=False,
            ),
            ParameterSchema(
                name="body",
                type="string",
                description="Body (for update)",
                required=False,
            ),
        ]
    )
    async def manage_drafts(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Manage email drafts."""
        try:
            user_id = context.user_id
            action = (params.get("action") or "list").lower()
            draft_id = params.get("draft_id")
            service = self._get_service(user_id)

            if action == "list":
                result = await service.list_drafts()
            elif action == "get" and draft_id:
                result = await service.tools.get_draft(draft_id)
            elif action == "update" and draft_id:
                result = await service.tools.update_draft(
                    draft_id=draft_id,
                    to=params.get("to"),
                    subject=params.get("subject"),
                    body=params.get("body"),
                )
            elif action == "send" and draft_id:
                result = await service.send_draft(draft_id)
            elif action == "delete" and draft_id:
                result = await service.delete_draft(draft_id)
            else:
                return AgentResponse.error(
                    f"Invalid action '{action}' or missing draft_id"
                )

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Draft {action} completed"
                )
            return AgentResponse.error(result.get("error", f"Draft {action} failed"))
        except Exception as e:
            logger.error(f"[manage_drafts] Error: {e}")
            return AgentResponse.error(f"Draft operation failed: {str(e)}")

    @capability(
        name="manage_labels",
        description="Manage Gmail labels: list, create, delete, or rename labels",
        parameters=[
            ParameterSchema(
                name="action",
                type="string",
                description="Action: 'list', 'create', 'delete', or 'rename'",
                required=True,
            ),
            ParameterSchema(
                name="label_name",
                type="string",
                description="Label name (for create/rename)",
                required=False,
            ),
            ParameterSchema(
                name="label_id",
                type="string",
                description="Label ID (for delete/rename)",
                required=False,
            ),
        ]
    )
    async def manage_labels(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Manage Gmail labels."""
        try:
            user_id = context.user_id
            action = (params.get("action") or "list").lower()
            service = self._get_service(user_id)

            if action == "list":
                result = await service.list_labels()
            elif action == "create" and params.get("label_name"):
                result = await service.create_label(params["label_name"])
            elif action == "delete" and params.get("label_id"):
                result = await service.tools.delete_label(params["label_id"])
            elif action == "rename" and params.get("label_id") and params.get("label_name"):
                result = await service.tools.patch_label(
                    label_id=params["label_id"],
                    name=params["label_name"],
                )
            else:
                return AgentResponse.error(
                    f"Invalid action '{action}' or missing required parameters"
                )

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Label {action} completed"
                )
            return AgentResponse.error(result.get("error", f"Label {action} failed"))
        except Exception as e:
            logger.error(f"[manage_labels] Error: {e}")
            return AgentResponse.error(f"Label operation failed: {str(e)}")

    @capability(
        name="batch_operations",
        description="Perform batch operations: archive, delete, label, or modify multiple emails at once",
        parameters=[
            ParameterSchema(
                name="action",
                type="string",
                description="Action: 'delete', 'archive', 'label', 'mark_read', 'mark_unread', 'star', 'unstar'",
                required=True,
            ),
            ParameterSchema(
                name="message_ids",
                type="array",
                description="List of message IDs to apply the batch action to",
                required=True,
            ),
            ParameterSchema(
                name="label_ids",
                type="array",
                description="Label IDs to add (for 'label' action)",
                required=False,
            ),
        ]
    )
    async def batch_operations(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Perform batch operations on multiple emails."""
        try:
            user_id = context.user_id
            action = (params.get("action") or "").lower()
            message_ids = params.get("message_ids", [])
            if not message_ids:
                return AgentResponse.error("message_ids required")

            service = self._get_service(user_id)

            label_action_map = {
                "archive": (None, ["INBOX"]),
                "mark_read": (None, ["UNREAD"]),
                "mark_unread": (["UNREAD"], None),
                "star": (["STARRED"], None),
                "unstar": (None, ["STARRED"]),
            }

            if action == "delete":
                result = await service.tools.batch_delete_messages(message_ids)
            elif action == "label" and params.get("label_ids"):
                result = await service.tools.batch_modify_messages(
                    message_ids=message_ids,
                    add_label_ids=params["label_ids"],
                )
            elif action in label_action_map:
                add_ids, remove_ids = label_action_map[action]
                result = await service.tools.batch_modify_messages(
                    message_ids=message_ids,
                    add_label_ids=add_ids,
                    remove_label_ids=remove_ids,
                )
            else:
                return AgentResponse.error(
                    f"Unknown action '{action}'. Use: delete, archive, label, "
                    "mark_read, mark_unread, star, unstar"
                )

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Batch {action} applied to {len(message_ids)} emails"
                )
            return AgentResponse.error(result.get("error", f"Batch {action} failed"))
        except Exception as e:
            logger.error(f"[batch_operations] Error: {e}")
            return AgentResponse.error(f"Batch operation failed: {str(e)}")

    @capability(
        name="manage_filters",
        description="Manage Gmail filters: list, create, or delete filter rules",
        parameters=[
            ParameterSchema(
                name="action",
                type="string",
                description="Action: 'list', 'create', 'get', or 'delete'",
                required=True,
            ),
            ParameterSchema(
                name="filter_id",
                type="string",
                description="Filter ID (for get/delete)",
                required=False,
            ),
            ParameterSchema(
                name="criteria",
                type="object",
                description="Filter criteria (for create): {from, to, subject, query, hasAttachment, etc.}",
                required=False,
            ),
            ParameterSchema(
                name="filter_action",
                type="object",
                description="Filter action (for create): {addLabelIds, removeLabelIds, forward, skipInbox, markRead, star, etc.}",
                required=False,
            ),
        ]
    )
    async def manage_filters(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Manage Gmail filters."""
        try:
            user_id = context.user_id
            action = (params.get("action") or "list").lower()
            service = self._get_service(user_id)

            if action == "list":
                result = await service.tools.list_filters()
            elif action == "get" and params.get("filter_id"):
                result = await service.tools.get_filter(params["filter_id"])
            elif action == "delete" and params.get("filter_id"):
                result = await service.tools.delete_filter(params["filter_id"])
            elif action == "create" and params.get("criteria") and params.get("filter_action"):
                result = await service.tools.create_filter(
                    criteria=params["criteria"],
                    action=params["filter_action"],
                )
            else:
                return AgentResponse.error(
                    f"Invalid action '{action}' or missing required parameters"
                )

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Filter {action} completed"
                )
            return AgentResponse.error(result.get("error", f"Filter {action} failed"))
        except Exception as e:
            logger.error(f"[manage_filters] Error: {e}")
            return AgentResponse.error(f"Filter operation failed: {str(e)}")

    @capability(
        name="get_settings",
        description="Get Gmail account settings: vacation auto-reply, forwarding, language, profile, send-as aliases",
        parameters=[
            ParameterSchema(
                name="setting",
                type="string",
                description="Setting to retrieve: 'vacation', 'forwarding', 'language', 'profile', 'aliases'",
                required=True,
            ),
        ]
    )
    async def get_settings(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Get Gmail account settings."""
        try:
            user_id = context.user_id
            setting = (params.get("setting") or "").lower()
            service = self._get_service(user_id)

            setting_methods = {
                "vacation": service.tools.get_vacation_settings,
                "forwarding": service.tools.get_auto_forwarding,
                "language": service.tools.get_language_settings,
                "profile": service.tools.get_profile,
                "aliases": service.tools.list_send_as_aliases,
            }

            method = setting_methods.get(setting)
            if not method:
                return AgentResponse.error(
                    f"Unknown setting '{setting}'. Use: vacation, forwarding, "
                    "language, profile, aliases"
                )

            result = await method()
            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Retrieved {setting} settings"
                )
            return AgentResponse.error(result.get("error", f"Failed to get {setting}"))
        except Exception as e:
            logger.error(f"[get_settings] Error: {e}")
            return AgentResponse.error(f"Settings retrieval failed: {str(e)}")

    @capability(
        name="execute_gmail_tool",
        description=(
            "Execute ANY Gmail tool by its Composio slug. "
            "Use this for advanced/uncommon actions not covered by other capabilities. "
            "Discover available tools with the 'discover_tools' query 'gmail'."
        ),
        parameters=[
            ParameterSchema(
                name="tool_slug",
                type="string",
                description="Composio tool slug (e.g., 'GMAIL_IMPORT_MESSAGE', 'GMAIL_INSERT_MESSAGE')",
                required=True,
            ),
            ParameterSchema(
                name="parameters",
                type="object",
                description="Tool parameters as key-value pairs",
                required=False,
            ),
        ]
    )
    async def execute_gmail_tool(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Execute any Gmail tool by slug — universal fallback for all 60 tools."""
        try:
            user_id = context.user_id
            tool_slug = params.get("tool_slug", "")
            tool_params = params.get("parameters", {})

            if not tool_slug:
                return AgentResponse.error("Missing tool_slug parameter")

            if not tool_slug.upper().startswith("GMAIL_"):
                return AgentResponse.error(
                    f"Tool slug must start with 'GMAIL_', got '{tool_slug}'"
                )

            service = self._get_service(user_id)
            result = await service.tools.execute_any_tool(tool_slug, tool_params)

            if result.get("success"):
                return AgentResponse.success(
                    result=result.get("data", {}),
                    summary=f"Executed {tool_slug}"
                )
            return AgentResponse.error(result.get("error", f"{tool_slug} failed"))
        except Exception as e:
            logger.error(f"[execute_gmail_tool] Error: {e}")
            return AgentResponse.error(f"Tool execution failed: {str(e)}")

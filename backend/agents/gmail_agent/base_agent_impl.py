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


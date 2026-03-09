"""
Gmail Agent - BaseAgent Implementation

Extends BaseAgent to provide Gmail operations with Composio tools.
Handles email search, send, reply, draft management, etc.
"""

import logging
from typing import Dict, Any, Optional
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
    
    def _get_service(self, user_id: str = "default") -> GmailService:
        """Get or create Gmail service for user."""
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
            ParameterSchema(
                name="user_id",
                type="string",
                description="User ID",
                required=False,
                default="default",
            ),
        ]
    )
    async def search_emails(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Search emails capability."""
        try:
            # params is passed directly by Capability.execute(); fall back to context.metadata
            if not params:
                params = context.metadata or {}
            user_id = params.get("user_id", "default")
            
            # Extract query from context or params
            # The orchestrator passes it in context.metadata
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
            ParameterSchema(
                name="user_id",
                type="string",
                description="User ID",
                required=False,
                default="default",
            ),
        ]
    )
    async def send_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Send email capability."""
        try:
            if not params:
                params = context.metadata or {}
            user_id = params.get("user_id", "default")
            
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
                description="Message ID to reply to",
                required=True,
            ),
            ParameterSchema(
                name="body",
                type="string",
                description="Reply body",
                required=True,
            ),
            ParameterSchema(
                name="user_id",
                type="string",
                description="User ID",
                required=False,
                default="default",
            ),
        ]
    )
    async def reply_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Reply to email capability."""
        try:
            if not params:
                params = context.metadata or {}
            message_id = params.get("message_id")
            body = params.get("body")
            user_id = params.get("user_id", "default")
            
            if not message_id or not body:
                return AgentResponse.error(
                    "Missing required parameters: message_id, body"
                )
            
            service = self._get_service(user_id)
            result = await service.reply_to_email(
                thread_id=message_id,
                message_id=message_id,
                body=body,
                to=params.get("to", "")
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
            ParameterSchema(
                name="user_id",
                type="string",
                description="User ID",
                required=False,
                default="default",
            ),
        ]
    )
    async def get_email(self, params: Dict[str, Any], context: ExecutionContext) -> AgentResponse:
        """Get email capability."""
        try:
            if not params:
                params = context.metadata or {}
            message_id = params.get("message_id")
            user_id = params.get("user_id", "default")
            
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


"""
General Agent - BaseAgent Implementation
Universal fallback agent for any Composio-powered integration.

Capabilities:
- Dynamic app detection (Slack, Notion, GitHub, etc.)
- Automatic connection verification
- Tool discovery and execution
- Multi-app support
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .service import GeneralAgentService
from .tool_cache import ToolCache

logger = logging.getLogger("agents.general_agent")


@dataclass
class GeneralAgentConfig(AgentConfig):
    """Configuration for General Agent."""
    tool_cache_ttl: int = 300  # 5 minutes
    supported_apps: List[str] = None
    
    def __post_init__(self):
        if self.supported_apps is None:
            self.supported_apps = [
                "slack", "notion", "github", "linear", "jira", "asana",
                "trello", "figma", "discord", "twitter", "instagram"
            ]


class GeneralAgent(BaseAgent):
    """
    Universal fallback agent for Composio integrations.
    
    Handles requests for any app not covered by specialized agents.
    Features:
    - Automatic app detection from user prompt
    - Connection verification with user prompting
    - Dynamic tool discovery
    - LLM-powered tool selection
    """
    
    def __init__(
        self,
        agent_id: str = "general_agent",
        agent_name: str = "General Agent",
        services: Optional[AgentServices] = None,
        config: Optional[GeneralAgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services or AgentServices.create_default(),
            config=config or GeneralAgentConfig(),
        )
        
        self.description = "Universal agent for any Composio-powered integration"
        
        # Initialize tool cache (shared across users)
        self.tool_cache = ToolCache(ttl_seconds=self.config.tool_cache_ttl)
        
        # Service instances (per-user)
        self._service_cache: Dict[str, GeneralAgentService] = {}
        
        logger.info(f"GeneralAgent initialized with {len(self.config.supported_apps)} supported apps")
    
    async def _initialize_resources(self):
        """Initialize agent-specific resources."""
        # Initialize Composio dependencies
        from services.integrations.composio_tools import get_tool_manager
        from services.integrations.composio_auth import get_auth_manager
        
        self.tool_manager = get_tool_manager()
        self.auth_manager = get_auth_manager()
        
        logger.info("General Agent resources initialized")
    
    def _get_service(self, user_id: str) -> GeneralAgentService:
        """Get or create service instance for user."""
        if user_id not in self._service_cache:
            self._service_cache[user_id] = GeneralAgentService(
                user_id=user_id,
                tool_cache=self.tool_cache,
                tool_manager=self.tool_manager,
                auth_manager=self.auth_manager,
            )
        return self._service_cache[user_id]
    
    # ============================================================================
    # CAPABILITIES
    # ============================================================================
    
    @capability(
        name="execute_composio_action",
        description="Execute an action on any Composio-supported app (Slack, Notion, GitHub, etc.)",
        parameters=[
            ParameterSchema(
                name="prompt",
                type="string",
                description="Natural language instruction (e.g., 'Send message to #general on Slack')",
                required=True,
            ),
            ParameterSchema(
                name="app_name",
                type="string",
                description="App to use (slack, notion, github, etc.). Auto-detected if not provided.",
                required=False,
            ),
            ParameterSchema(
                name="action_params",
                type="object",
                description="Structured parameters for the action",
                required=False,
            ),
        ],
    )
    async def execute_composio_action(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute an action on any Composio-supported app.
        
        Workflow:
        1. Extract or detect app name from prompt
        2. Check if user has connection for that app
        3. If not connected, return needs_input
        4. If connected, discover tools and execute
        """
        prompt = params.get("prompt", "")
        app_name = params.get("app_name")
        action_params = params.get("action_params", {})
        
        if not prompt and not app_name:
            return {
                "success": False,
                "error": "Either 'prompt' or 'app_name' must be provided"
            }
        
        # Get service for this user
        service = self._get_service(context.user_id)
        
        # Execute through service
        result = await service.execute(
            prompt=prompt,
            app_name=app_name,
            action_params=action_params,
            context=context,
        )
        
        return result
    
    @capability(
        name="check_app_connection",
        description="Check if user has connected a specific app (Slack, Notion, etc.)",
        parameters=[
            ParameterSchema(
                name="app_name",
                type="string",
                description="App slug to check (e.g., 'slack', 'notion', 'github')",
                required=True,
            ),
        ],
    )
    async def check_app_connection(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Check if user has connected an app."""
        app_name = params.get("app_name", "").lower()
        
        if not app_name:
            return {
                "success": False,
                "error": "Missing 'app_name' parameter"
            }
        
        try:
            # Check connection via auth manager
            status = self.auth_manager.check_connection_status(
                user_id=context.user_id,
                app_slug=app_name
            )
            
            is_connected = app_name in status.get("connected_apps", [])
            
            if is_connected:
                return {
                    "success": True,
                    "connected": True,
                    "app_name": app_name,
                    "message": f"{app_name.title()} is connected"
                }
            else:
                # Generate connect URL
                frontend_url = os.getenv('FRONTEND_URL', 'https://app.orbimesh.com')
                connect_url = f"{frontend_url}/connections/{app_name}"
                
                return {
                    "success": False,
                    "connected": False,
                    "app_name": app_name,
                    "error": f"No {app_name.title()} connection found",
                    "connect_url": connect_url,
                    "message": f"Please connect your {app_name.title()} account"
                }
        
        except Exception as e:
            logger.error(f"Error checking connection for {app_name}: {e}")
            return {
                "success": False,
                "error": f"Failed to check connection: {str(e)}"
            }
    
    @capability(
        name="list_supported_apps",
        description="List all apps supported by this agent",
        parameters=[],
    )
    async def list_supported_apps(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """List all supported apps."""
        return {
            "success": True,
            "apps": self.config.supported_apps,
            "total_count": len(self.config.supported_apps),
            "message": f"Supports {len(self.config.supported_apps)} Composio apps"
        }
    
    @capability(
        name="get_available_tools",
        description="Get available tools for a specific app the user has connected",
        parameters=[
            ParameterSchema(
                name="app_name",
                type="string",
                description="App slug (e.g., 'slack', 'notion')",
                required=True,
            ),
        ],
    )
    async def get_available_tools(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Get list of available tools for an app."""
        app_name = params.get("app_name", "").lower()
        
        if not app_name:
            return {
                "success": False,
                "error": "Missing 'app_name' parameter"
            }
        
        service = self._get_service(context.user_id)
        
        try:
            # Check connection first
            connection_check = await self.check_app_connection(
                {"app_name": app_name}, context
            )
            
            if not connection_check.get("connected"):
                return connection_check
            
            # Get tools from cache or Composio
            tools = await service._get_tools_for_app(app_name)
            
            return {
                "success": True,
                "app_name": app_name,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description if hasattr(tool, 'description') else ""
                    }
                    for tool in tools
                ],
                "tool_count": len(tools),
            }
        
        except Exception as e:
            logger.error(f"Error getting tools for {app_name}: {e}")
            return {
                "success": False,
                "error": f"Failed to get tools: {str(e)}"
            }


# Singleton instance
_agent_instance = None


def get_agent() -> GeneralAgent:
    """Get or create singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = GeneralAgent()
    return _agent_instance

"""
Integrations Agent - BaseAgent Implementation
Universal fallback agent for any Composio-powered integration.
Features in-chat OAuth, app detection, and session persistence.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentServices, AgentConfig
from backend.agents.base.types import ExecutionContext
from backend.agents.base.capability import capability, ParameterSchema

from .service import IntegrationsAgentService
from .tool_cache import ToolCache

logger = logging.getLogger("agents.integrations_agent")


@dataclass
class IntegrationsAgentConfig(AgentConfig):
    """Configuration for Integrations Agent."""
    tool_cache_ttl: int = 300  # 5 minutes


class IntegrationsAgent(BaseAgent):
    """
    Universal fallback agent with in-chat OAuth and session persistence.
    Handles requests for any Composio app not covered by dedicated agents.
    """
    
    def __init__(
        self,
        agent_id: str = "integrations_agent",
        agent_name: str = "Integrations Agent",
        services: Optional[AgentServices] = None,
        config: Optional[IntegrationsAgentConfig] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            services=services or AgentServices.create_default(),
            config=config or IntegrationsAgentConfig(),
        )
        
        self.description = "Universal agent for any Composio-powered integration"
        
        # Initialize tool cache (shared across users)
        self.tool_cache = ToolCache(ttl_seconds=self.config.tool_cache_ttl)
        
        # Service instances (per-user)
        self._service_cache: Dict[str, IntegrationsAgentService] = {}

        logger.info("IntegrationsAgent initialized (dynamic toolkit discovery enabled)")
    
    async def _initialize_resources(self):
        """Initialize agent-specific resources."""
        # Initialize Composio dependencies
        from services.integrations.composio_tools import get_tool_manager
        from services.integrations.composio_auth import get_auth_manager
        
        self.tool_manager = get_tool_manager()
        self.auth_manager = get_auth_manager()
        
        logger.info("Integrations Agent resources initialized")
    
    def _get_service(self, user_id: str) -> IntegrationsAgentService:
        """Get or create IntegrationsAgentService instance for user."""
        if user_id not in self._service_cache:
            self._service_cache[user_id] = IntegrationsAgentService(
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
        name="discover_tools",
        description="Search for available tools/integrations by natural language query",
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                description="Natural language search (e.g., 'send slack message', 'create jira ticket')",
                required=True,
            ),
        ],
    )
    async def discover_tools(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Search for available Composio tools by natural language query."""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Missing 'query' parameter"}

        try:
            results = self.tool_manager.search_tools(query=query)
            return {
                "success": True,
                "query": query,
                "tools": [
                    {
                        "name": r.get("name") or r.get("slug", ""),
                        "description": r.get("description", "")[:200],
                    }
                    for r in results[:15]
                ],
                "total_found": len(results),
                "message": f"Found {len(results)} tools matching '{query}'",
            }
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            return {"success": False, "error": str(e)}

    @capability(
        name="manage_connections",
        description="Initiate or check connection to a third-party app (OAuth)",
        parameters=[
            ParameterSchema(
                name="app_name",
                type="string",
                description="App slug to connect (e.g., 'slack', 'notion', 'github')",
                required=True,
            ),
            ParameterSchema(
                name="action",
                type="string",
                description="'connect', 'disconnect', or 'check'",
                required=False,
            ),
        ],
    )
    async def manage_connections(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """Manage user connections to third-party apps."""
        app_name = params.get("app_name", "").lower()
        action = params.get("action", "connect").lower()

        if not app_name:
            return {"success": False, "error": "Missing 'app_name' parameter"}

        try:
            if action == "check":
                return await self.check_app_connection({"app_name": app_name}, context)

            elif action == "disconnect":
                result = self.auth_manager.disconnect_app(context.user_id, app_name)
                return result

            else:  # connect
                # Check if already connected
                connection = self.auth_manager.get_connection_for_agent(
                    context.user_id, app_name
                )
                if connection:
                    return {
                        "success": True,
                        "connected": True,
                        "message": f"{app_name.title()} is already connected",
                    }

                # Initiate auth flow
                auth_result = self.auth_manager.start_auth_flow(
                    context.user_id, app_name
                )
                if auth_result.get("success"):
                    return {
                        "success": True,
                        "needs_approval": True,
                        "auth_url": auth_result.get("redirect_url"),
                        "app_name": app_name,
                        "message": (
                            f"Please connect {app_name.title()}: "
                            f"{auth_result.get('redirect_url', 'Go to Connections page')}"
                        ),
                    }
                else:
                    return {
                        "success": False,
                        "error": auth_result.get("error", "Failed to start auth flow"),
                    }

        except Exception as e:
            logger.error(f"Connection management failed for {app_name}: {e}")
            return {"success": False, "error": str(e)}

    @capability(
        name="list_supported_apps",
        description="List available integrations and their connection status",
        parameters=[
            ParameterSchema(
                name="search",
                type="string",
                description="Optional search query to filter apps",
                required=False,
            ),
        ],
    )
    async def list_supported_apps(
        self, params: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """List available integrations, optionally with search."""
        search = params.get("search")

        try:
            toolkits = self.tool_manager.list_toolkits(search=search)
            return {
                "success": True,
                "apps": [
                    {
                        "name": tk.get("name", tk.get("slug", "")),
                        "slug": tk.get("slug", tk.get("name", "")).lower(),
                        "description": tk.get("description", "")[:150],
                    }
                    for tk in toolkits[:30]
                ],
                "total_count": len(toolkits),
                "message": f"Found {len(toolkits)} available integrations",
            }
        except Exception as e:
            logger.error(f"Failed to list toolkits: {e}")
            return {"success": False, "error": str(e)}
    
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


def get_agent() -> IntegrationsAgent:
    """Get or create singleton IntegrationsAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntegrationsAgent()
    return _agent_instance

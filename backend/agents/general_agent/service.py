# agents/general_agent/service.py
"""
General Agent Service

Core business logic for the General Fallback Agent.
Handles tool discovery, connection checking, and execution.

Updated for BaseAgent compatibility.
"""

import logging
import re
from typing import Dict, Any, Optional, List
import os

from backend.agents.base.types import ExecutionContext

logger = logging.getLogger("general_agent")

class GeneralAgentService:
    """
    Service layer for General Fallback Agent.
    
    Responsibilities:
    - Determine which app the user wants to interact with
    - Check if user has required connection
    - Discover available tools from Composio
    - Execute tools with user parameters
    - Handle errors gracefully
    """
    
    def __init__(
        self,
        user_id: str,
        tool_cache,
        tool_manager=None,
        auth_manager=None
    ):
        """
        Initialize service for a specific user.
        
        Args:
            user_id: User ID for connection checking
            tool_cache: Shared ToolCache instance
            tool_manager: ComposioToolManager instance
            auth_manager: ComposioAuthManager instance
        """
        self.user_id = user_id
        self.tool_cache = tool_cache
        
        # Use injected managers or initialize
        if tool_manager:
            self.tool_manager = tool_manager
        else:
            from services.integrations.composio_tools import get_tool_manager
            self.tool_manager = get_tool_manager()
        
        if auth_manager:
            self.auth_manager = auth_manager
        else:
            from services.integrations.composio_auth import get_auth_manager
            self.auth_manager = get_auth_manager()
        
        logger.info(f"[GeneralAgentService] Initialized for user {user_id}")
    
    async def execute(
        self,
        prompt: str,
        app_name: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute a user prompt.
        
        Workflow:
        1. Parse prompt to determine app (or use provided app_name)
        2. Check if user has connection for that app
        3. If not, return error with connect URL
        4. If yes, discover tools and execute
        5. Return result
        
        Args:
            prompt: Natural language instruction
            app_name: Optional app name (auto-detected if not provided)
            action_params: Optional structured parameters
            context: Execution context with user_id, thread_id, etc.
        
        Returns:
            Dict with success, data, error, etc.
        """
        try:
            # Step 1: Determine which app the user wants to use
            if not app_name:
                app_name = self._extract_app_from_prompt(prompt)
            
            if not app_name:
                return {
                    "success": False,
                    "error": "Could not determine which app to use. Please mention the app name explicitly (e.g., 'Send a Slack message', 'Create a Notion page')."
                }
            
            logger.info(f"[Execute] Using app: {app_name} for prompt: {prompt[:100]}")
            
            # Step 2: Check if user has connection for this app
            connection_check = await self._check_connection(app_name)
            
            if not connection_check["connected"]:
                # User needs to connect the app
                frontend_url = os.getenv('FRONTEND_URL', 'https://app.orbimesh.com')
                connect_url = f"{frontend_url}/connections/{app_name.lower()}"
                
                return {
                    "success": False,
                    "error": f"No {app_name.title()} connection found",
                    "message": f"Please connect your {app_name.title()} account to continue",
                    "connect_url": connect_url,
                    "needs_connection": True
                }
            
            # Step 3: Get tools for this app (cached)
            tools = await self._get_tools_for_app(app_name)
            
            if not tools:
                return {
                    "success": False,
                    "error": f"No tools available for {app_name.title()}. This may be a temporary issue."
                }
            
            logger.info(f"[Execute] Found {len(tools)} tools for {app_name}")
            
            # Step 4: Execute using LLM tool selection
            # For now, using simple heuristic - replace with LLM in production
            result = await self._execute_with_tools(
                prompt=prompt,
                tools=tools,
                app_name=app_name,
                action_params=action_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[Execute] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }
    
    def _extract_app_from_prompt(self, prompt: str) -> Optional[str]:
        """
        Extract app name from user prompt using simple pattern matching.
        
        Examples:
        - "Send a Slack message" → "Slack"
        - "Create Notion page" → "Notion"
        - "Open GitHub PR" → "GitHub"
        
        Args:
            prompt: User's natural language instruction
        
        Returns:
            App name or None if not found
        """
        prompt_lower = prompt.lower()
        
        # Common app patterns
        app_patterns = {
            "slack": ["slack"],
            "notion": ["notion"],
            "github": ["github", "git hub"],
            "linear": ["linear"],
            "jira": ["jira"],
            "asana": ["asana"],
            "trello": ["trello"],
            "figma": ["figma"],
            "discord": ["discord"],
            "hubspot": ["hubspot"],
            "salesforce": ["salesforce"],
            "zendesk": ["zendesk"],
            "intercom": ["intercom"],
        }
        
        for app_name, patterns in app_patterns.items():
            for pattern in patterns:
                if pattern in prompt_lower:
                    return app_name.capitalize()
        
        return None
    
    async def _check_connection(self, app_name: str) -> Dict[str, bool]:
        """
        Check if user has active connection for app.
        
        Args:
            app_name: App name (e.g., "Slack", "Notion")
        
        Returns:
            {"connected": True/False}
        """
        try:
            status = self.auth_manager.check_connection_status(
                user_id=self.user_id,
                app_slug=app_name.lower()
            )
            
            # Check if app is in connected_apps list
            connected_apps = status.get("connected_apps", [])
            is_connected = app_name.lower() in [app.lower() for app in connected_apps]
            
            return {"connected": is_connected}
        except Exception as e:
            logger.error(f"[CheckConnection] Error: {e}")
            return {"connected": False}
    
    async def _get_tools_for_app(self, app_name: str) -> List[Any]:
        """
        Get tools for app from cache or Composio API.
        
        Args:
            app_name: App name (e.g., "Slack")
        
        Returns:
            List of LangChain tools
        """
        # Check cache first
        cached_tools = self.tool_cache.get(self.user_id, app_name)
        if cached_tools is not None:
            logger.info(f"[GetTools] Cache hit for {app_name} (user {self.user_id})")
            return cached_tools
        
        # Cache miss - fetch from Composio
        logger.info(f"[GetTools] Cache miss for {app_name}, fetching from Composio")
        
        try:
            # Get tools using tool manager
            tools = self.tool_manager.get_tools_for_user(
                user_id=self.user_id,
                toolkits=[app_name.lower()]
            )
            
            # Cache the result
            if tools:
                self.tool_cache.set(self.user_id, app_name, tools)
            
            return tools or []
        except Exception as e:
            logger.error(f"[GetTools] Error fetching tools: {e}", exc_info=True)
            return []
    
    async def _execute_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        app_name: str,
        action_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the user's prompt using available tools.
        
        Uses simple heuristic for tool selection.
        TODO: Replace with LLM-based tool selection for production.
        
        Args:
            prompt: User instruction
            tools: Available tools from Composio
            app_name: App being used
            action_params: Optional structured parameters
        
        Returns:
            Result dict with success, data, error
        """
        try:
            # Select tool based on heuristic
            tool_name = self._select_tool_heuristic(prompt, tools, app_name)
            
            if not tool_name:
                return {
                    "success": False,
                    "error": f"Could not determine which {app_name.title()} action to take. Please be more specific."
                }
            
            logger.info(f"[ExecuteWithTools] Selected tool: {tool_name}")
            
            # Extract parameters from prompt
            params = self._extract_parameters(prompt, action_params)
            
            logger.info(f"[ExecuteWithTools] Executing {tool_name} with params: {params}")
            
            # Execute the tool
            # Note: This is a simplified execution - in production, use LangChain Agent
            selected_tool = next((t for t in tools if hasattr(t, 'name') and t.name == tool_name), None)
            
            if not selected_tool:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found in available tools"
                }
            
            # Execute tool directly
            result = await selected_tool.ainvoke(params)
            
            return {
                "success": True,
                "data": result,
                "tool_used": tool_name,
                "message": f"Successfully executed {tool_name}"
            }
            
        except Exception as e:
            logger.error(f"[ExecuteWithTools] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def _select_tool_heuristic(self, prompt: str, tools: List[Any], app_name: str) -> Optional[str]:
        """
        Simple heuristic for tool selection.
        
        TODO: Replace with LLM-based selection for production.
        """
        prompt_lower = prompt.lower()
        
        # Slack-specific heuristics
        if app_name.lower() == "slack":
            if "send" in prompt_lower and "message" in prompt_lower:
                return "SLACK_SEND_MESSAGE"
            elif "list" in prompt_lower and "channel" in prompt_lower:
                return "SLACK_LIST_CHANNELS"
        
        # Notion-specific heuristics
        elif app_name.lower() == "notion":
            if "create" in prompt_lower and "page" in prompt_lower:
                return "NOTION_CREATE_PAGE"
            elif "search" in prompt_lower:
                return "NOTION_SEARCH"
        
        # GitHub-specific heuristics
        elif app_name.lower() == "github":
            if "create" in prompt_lower and ("issue" in prompt_lower or "pr" in prompt_lower):
                return "GITHUB_CREATE_ISSUE"
            elif "list" in prompt_lower and "repo" in prompt_lower:
                return "GITHUB_LIST_REPOS"
        
        # Fallback: Use first tool if available
        if tools:
            first_tool = tools[0]
            if hasattr(first_tool, 'name'):
                return first_tool.name
        
        return None
    
    def _extract_parameters(self, prompt: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract parameters from prompt and payload.
        
        TODO: Use LLM to extract structured parameters.
        """
        params = payload.copy() if payload else {}
        
        # Simple extraction examples
        # In production, use LLM for this
        
        # Extract channel from "send to #general"
        channel_match = re.search(r'#(\w+)', prompt)
        if channel_match:
            params["channel"] = channel_match.group(1)
        
        # Extract quoted text as message
        message_match = re.search(r'"([^"]+)"', prompt)
        if message_match:
            params["text"] = message_match.group(1)
        elif "'" in prompt:
            message_match = re.search(r"'([^']+)'", prompt)
            if message_match:
                params["text"] = message_match.group(1)
        
        return params

# agents/general_agent/service.py
"""
General Agent Service

Core business logic for the General Fallback Agent.
Handles tool discovery, connection checking, and execution.
"""

import logging
import re
from typing import Dict, Any, Optional, List
import os

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
    
    def __init__(self, user_id: str, tool_cache):
        """
        Initialize service for a specific user.
        
        Args:
            user_id: User ID for connection checking
            tool_cache: Shared ToolCache instance
        """
        self.user_id = user_id
        self.tool_cache = tool_cache
        
        # Initialize Composio client
        from services.integrations.composio_tools import ComposioToolManager
        self.composio = ComposioToolManager()
        
        # Initialize connection service for checking
        from services.integrations.composio_auth import ComposioAuthManager
        self.auth_service = ComposioAuthManager()
        
        logger.info(f"[GeneralAgentService] Initialized for user {user_id}")
    
    async def execute(
        self,
        prompt: str,
        payload: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a user prompt.
        
        Workflow:
        1. Parse prompt to determine app (e.g., "Slack", "Notion")
        2. Check if user has connection for that app
        3. If not, return needs_input with connect URL
        4. If yes, discover tools and execute
        5. Return result
        
        Args:
            prompt: Natural language instruction
            payload: Optional structured data
            task_id: Task ID for multi-turn
            thread_id: Thread ID for session context
        
        Returns:
            UAP response dict
        """
        try:
            # Step 1: Determine which app the user wants to use
            app_name = self._extract_app_from_prompt(prompt)
            
            if not app_name:
                return {
                    "success": False,
                    "result": None,
                    "status": "error",
                    "error": "Could not determine which app to use. Please mention the app name explicitly (e.g., 'Send a Slack message', 'Create a Notion page')."
                }
            
            logger.info(f"[Execute] Detected app: {app_name} for prompt: {prompt[:100]}")
            
            # Step 2: Check if user has connection for this app
            connection_status = await self._check_connection(app_name)
            
            if not connection_status["connected"]:
                # User needs to connect the app
                connect_url = f"{os.getenv('FRONTEND_URL', 'https://app.orbimesh.com')}/connections/{app_name.lower()}"
                
                return {
                    "success": False,
                    "result": None,
                    "status": "needs_input",
                    "question": f"Please connect your {app_name} account to continue.",
                    "error": f"No {app_name} connection found. Connect at: {connect_url}"
                }
            
            # Step 3: Get tools for this app (cached)
            tools = await self._get_tools_for_app(app_name)
            
            if not tools:
                return {
                    "success": False,
                    "result": None,
                    "status": "error",
                    "error": f"No tools available for {app_name}. This may be a temporary issue."
                }
            
            logger.info(f"[Execute] Found {len(tools)} tools for {app_name}")
            
            # Step 4: Use LLM to select and execute the right tool
            result = await self._execute_with_tools(
                prompt=prompt,
                tools=tools,
                app_name=app_name,
                payload=payload
            )
            
            return result
            
        except ValueError as e:
            # User-friendly errors
            logger.warning(f"[Execute] ValueError: {e}")
            return {
                "success": False,
                "result": None,
                "status": "error",
                "error": str(e)
            }
        except Exception as e:
            # Unexpected errors
            logger.error(f"[Execute] Unexpected error: {e}", exc_info=True)
            return {
                "success": False,
                "result": None,
                "status": "error",
                "error": f"Internal error: {str(e)}"
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
            status = await self.auth_service.check_connection_status(
                user_id=self.user_id,
                app_slug=app_name.lower()
            )
            return {"connected": status.get("is_active", False)}
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
            tools = self.composio.get_tools_for_user(
                user_id=self.user_id,
                toolkits=[app_name.lower()]
            )
            
            # Cache the result
            self.tool_cache.set(self.user_id, app_name, tools)
            
            return tools
        except Exception as e:
            logger.error(f"[GetTools] Error fetching tools: {e}")
            return []
    
    async def _execute_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        app_name: str,
        payload: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the user's prompt using available tools.
        
        This is a simplified implementation. In production, you'd use:
        - LangChain Agent with ReAct pattern
        - OpenAI function calling
        - Or similar LLM-based tool selection
        
        Args:
            prompt: User instruction
            tools: Available tools from Composio
            app_name: App being used
            payload: Optional structured data
        
        Returns:
            UAP response dict
        """
        try:
            # For now, we'll use a simple heuristic to select the tool
            # In production, replace this with proper LLM-based tool selection
            
            # Example: If prompt contains "send message" and app is Slack,
            # use SLACK_SEND_MESSAGE tool
            
            tool_name = self._select_tool_heuristic(prompt, tools, app_name)
            
            if not tool_name:
                return {
                    "success": False,
                    "result": None,
                    "status": "error",
                    "error": f"Could not determine which {app_name} action to take. Please be more specific."
                }
            
            # Extract parameters from prompt (simplified)
            params = self._extract_parameters(prompt, payload)
            
            # Execute the tool
            result = self.composio.execute_tool(
                user_id=self.user_id,
                tool_slug=tool_name,
                arguments=params
            )
            
            if result.get("success", False):
                return {
                    "success": True,
                    "result": result.get("data"),
                    "status": "completed"
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "status": "error",
                    "error": result.get("error", "Tool execution failed")
                }
            
        except Exception as e:
            logger.error(f"[ExecuteWithTools] Error: {e}", exc_info=True)
            return {
                "success": False,
                "result": None,
                "status": "error",
                "error": str(e)
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

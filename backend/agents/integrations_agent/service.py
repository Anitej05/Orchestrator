# agents/integrations_agent/service.py
"""
Integrations Agent Service

Core business logic for the Integrations Agent (universal fallback + in-chat OAuth).
Handles app detection, connection checking, session management, and execution.

Architecture: Tier 3 – Dedicated Agent (as per MASTER_IMPLEMENTATION_PLAN_v2)
"""

import logging
import re
from typing import Dict, Any, Optional, List
import os

from backend.agents.base.types import ExecutionContext

logger = logging.getLogger("integrations_agent")

class IntegrationsAgentService:
    """
    Service layer for the Integrations Agent (universal fallback + in-chat OAuth).

    Responsibilities:
    - Detect which Composio app the task requires (via AppDetector)
    - Check per-user connection status
    - Return inline OAuth URL when connection is missing (in-chat OAuth flow)
    - Manage per-user, per-app session context (via ComposioSessionManager)
    - Discover available Composio tools and execute them
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
        
        logger.info(f"[IntegrationsAgentService] Initialized for user {user_id}")
    
    async def execute(
        self,
        prompt: str,
        app_name: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
        # UAP-style kwargs (forwarded from agent.py /execute endpoint)
        payload: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with in-chat OAuth support.

        Workflow:
          1. Detect which Composio app the task needs (AppDetector)
          2. Check whether the user has an active connection for that app
          3a. Connection missing  → return pending_auth with auth URL (in-chat OAuth)
          3b. Connection present  → load/create session, fetch tools, execute, save context

        Returns (connection missing):
          {
            "status": "pending_auth",
            "needs_approval": True,
            "auth_url": "https://connect.composio.dev/link/...",
            "app_slug": "slack",
            "app_name": "Slack",
            "message": "Please authorise Slack to continue"
          }

        Returns (success):
          {
            "success": True,
            "status": "completed",
            "result": { ... },
            "session_id": "...",
            "app_slug": "slack"
          }
        """
        try:
            # Merge payload kwargs so the method works when called from agent.py
            if payload:
                if not action_params:
                    action_params = payload
                if not app_name and "app_name" in payload:
                    app_name = payload["app_name"]

            # ----------------------------------------------------------------
            # Step 1: Detect which app the task requires
            # ----------------------------------------------------------------
            from .app_detector import get_app_detector

            detector = get_app_detector()
            if app_name:
                # Caller already knows the app; wrap in detection result format
                detection = {
                    "app_slug": app_name.lower(),
                    "app_name": app_name,
                    "confidence": 1.0,
                    "method": "INTEGRATIONS_AGENT",
                    "agent_id": "integrations_agent",
                }
            else:
                detection = detector.detect_app_from_task(prompt)

            if not detection.get("app_slug"):
                return {
                    "success": False,
                    "status": "error",
                    "error": (
                        "Could not determine which app to use. "
                        "Please mention the app name explicitly "
                        "(e.g. 'Send a Slack message', 'Create a Notion page')."
                    ),
                }

            app_slug = detection["app_slug"]
            app_display_name = detection["app_name"]

            logger.info(
                f"[IntegrationsAgentService] Detected app: {app_slug} "
                f"(confidence={detection.get('confidence', 0):.2f}) "
                f"for user {self.user_id}"
            )

            # ----------------------------------------------------------------
            # Step 2: Check connection status for this user + app
            # ----------------------------------------------------------------
            connection = self.auth_manager.get_connection_for_agent(self.user_id, app_slug)

            if not connection:
                # ----------------------------------------------------------------
                # Step 3a: Connection missing → return inline auth link
                # ----------------------------------------------------------------
                logger.info(
                    f"[IntegrationsAgentService] No {app_slug} connection for "
                    f"user {self.user_id} — initiating in-chat OAuth"
                )
                try:
                    auth_result = self.auth_manager.start_auth_flow(self.user_id, app_slug)
                    auth_url = auth_result.get("redirect_url") if auth_result.get("success") else None
                except Exception as auth_err:
                    logger.warning(f"[IntegrationsAgentService] start_auth_flow error: {auth_err}")
                    auth_url = None

                return {
                    "success": False,
                    "status": "pending_auth",
                    "needs_approval": True,
                    "auth_url": auth_url,
                    "app_slug": app_slug,
                    "app_name": app_display_name,
                    "message": (
                        f"Please authorise {app_display_name} to continue. "
                        + (f"Click here: {auth_url}" if auth_url else "Go to your Connections page.")
                    ),
                    # Structured for Brain to handle via requires_approval flow
                    "result": None,
                    "error": f"No active {app_display_name} connection",
                }

            # ----------------------------------------------------------------
            # Step 3b: Connected — load/create session & execute
            # ----------------------------------------------------------------
            from .session_manager import get_session_manager

            session_mgr = get_session_manager()
            session = session_mgr.get_or_create_session(self.user_id, app_slug)

            logger.info(
                f"[IntegrationsAgentService] Using session {session['session_id']} "
                f"(new={session['is_new']})"
            )

            # Fetch tools
            tools = await self._get_tools_for_app(app_slug)
            if not tools:
                return {
                    "success": False,
                    "status": "error",
                    "error": (
                        f"No tools available for {app_display_name}. "
                        "This may be a temporary issue — please try again."
                    ),
                }

            logger.info(
                f"[IntegrationsAgentService] Found {len(tools)} tools for {app_slug}"
            )

            # Execute
            exec_result = await self._execute_with_tools(
                prompt=prompt,
                tools=tools,
                app_name=app_slug,
                action_params=action_params,
            )

            # Persist any context the execution surfaced
            if exec_result.get("success") and exec_result.get("context"):
                session_mgr.save_context(self.user_id, app_slug, exec_result["context"])

            return {
                "success": exec_result.get("success", False),
                "status": "completed" if exec_result.get("success") else "error",
                "result": exec_result.get("data"),
                "session_id": session["session_id"],
                "app_slug": app_slug,
                "tool_used": exec_result.get("tool_used"),
                "message": exec_result.get("message", ""),
                "error": exec_result.get("error"),
            }

        except Exception as e:
            logger.error(f"[IntegrationsAgentService.execute] Unexpected error: {e}", exc_info=True)
            return {
                "success": False,
                "status": "error",
                "error": f"Execution failed: {str(e)}",
            }
    
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

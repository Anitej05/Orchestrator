# agents/integrations_agent/service.py
"""
Integrations Agent Service

Core business logic for the Integrations Agent (universal fallback + in-chat OAuth).
Handles dynamic tool discovery, LLM-driven tool selection, connection checking,
session management, and execution.

Architecture: Tier 3 – Dedicated Agent (as per MASTER_IMPLEMENTATION_PLAN_v2)
"""

import json
import logging
import os
import re
from typing import Dict, Any, Optional, List

from backend.base_agent.types import ExecutionContext

logger = logging.getLogger("integrations_agent")

# LLM model selection - uses inference_service now
# No hardcoded model - let inference_service handle provider selection


class IntegrationsAgentService:
    """
    Service layer for the Integrations Agent (universal fallback + in-chat OAuth).

    Dynamic execution flow (v3):
      1. Detect which app the task needs (AppDetector fast-path OR Composio search)
      2. Check per-user connection status
      3a. Missing connection → return inline OAuth URL (in-chat OAuth)
      3b. Connected → dynamically discover tools, LLM selects best tool,
          LLM extracts parameters, execute, save context

    Key difference from v1: No hardcoded tool/app maps. Uses Composio's
    search API for tool discovery and an LLM for tool selection + param extraction.
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
        Execute a task with dynamic tool discovery and in-chat OAuth.

        Workflow:
          1. Detect which Composio app the task needs (AppDetector)
          2. Check whether the user has an active connection for that app
          3a. Connection missing  → return pending_auth with auth URL (in-chat OAuth)
          3b. Connection present  → dynamically discover tools, LLM selects + extracts
              params, execute, save context

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
                # Fast-path detection failed — try dynamic search via Composio
                logger.info(
                    "[IntegrationsAgentService] AppDetector returned no app, "
                    "trying dynamic tool search"
                )
                return await self._execute_dynamic_search(
                    prompt=prompt,
                    action_params=action_params,
                )

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

            # Fetch tools dynamically
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

            # Execute with LLM-driven tool selection
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

    # ------------------------------------------------------------------
    # Dynamic search fallback (when AppDetector can't identify the app)
    # ------------------------------------------------------------------

    async def _execute_dynamic_search(
        self,
        prompt: str,
        action_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fallback when AppDetector can't identify the app.

        Uses Composio's search API to find relevant tools directly from
        the user's prompt, then executes the best match.
        """
        try:
            # Search for tools matching the prompt
            search_results = self.tool_manager.search_tools(query=prompt)

            if not search_results:
                return {
                    "success": False,
                    "status": "error",
                    "error": (
                        "Could not find any matching integrations for your request. "
                        "Please mention the app name explicitly "
                        "(e.g., 'Send a Slack message', 'Create a Notion page')."
                    ),
                }

            # Extract the app/toolkit from the first result
            top_result = search_results[0]
            tool_name = top_result.get("name", top_result.get("slug", ""))

            # Derive app slug from tool name (e.g., SLACK_SEND_MESSAGE → slack)
            app_slug = tool_name.split("_", 1)[0].lower() if "_" in tool_name else ""

            if not app_slug:
                return {
                    "success": False,
                    "status": "error",
                    "error": "Could not determine which app to use from search results.",
                }

            logger.info(
                f"[DynamicSearch] Found tool {tool_name} (app: {app_slug}) "
                f"from {len(search_results)} search results"
            )

            # Check connection
            connection = self.auth_manager.get_connection_for_agent(self.user_id, app_slug)
            if not connection:
                try:
                    auth_result = self.auth_manager.start_auth_flow(self.user_id, app_slug)
                    auth_url = auth_result.get("redirect_url") if auth_result.get("success") else None
                except Exception:
                    auth_url = None

                return {
                    "success": False,
                    "status": "pending_auth",
                    "needs_approval": True,
                    "auth_url": auth_url,
                    "app_slug": app_slug,
                    "app_name": app_slug.title(),
                    "message": (
                        f"I found the right tool ({tool_name}), but you need to "
                        f"connect {app_slug.title()} first. "
                        + (f"Click here: {auth_url}" if auth_url else "Go to your Connections page.")
                    ),
                    "result": None,
                    "error": f"No active {app_slug.title()} connection",
                }

            # Build adapters from search results (only matching app)
            matching_tools = [
                r for r in search_results
                if (r.get("name", "") or r.get("slug", "")).upper().startswith(app_slug.upper())
            ]

            adapters = []
            for r in matching_tools[:10]:  # Limit to top 10
                from services.integrations.composio_tools import ComposioActionAdapter
                name = r.get("name") or r.get("slug", "")
                adapters.append(
                    ComposioActionAdapter(
                        action_name=name,
                        composio_client=self.tool_manager._composio,
                        user_id=self.user_id,
                        connected_account=self.tool_manager._resolve_connected_account_id(
                            self.user_id, name
                        ),
                        description=r.get("description", ""),
                        parameters=r.get("parameters", {}),
                    )
                )

            if not adapters:
                return {
                    "success": False,
                    "status": "error",
                    "error": f"Found tools but none matched app {app_slug}.",
                }

            # Execute with LLM
            return await self._execute_with_tools(
                prompt=prompt,
                tools=adapters,
                app_name=app_slug,
                action_params=action_params,
            )

        except Exception as e:
            logger.error(f"[DynamicSearch] Error: {e}", exc_info=True)
            return {
                "success": False,
                "status": "error",
                "error": f"Dynamic tool search failed: {str(e)}",
            }

    # ------------------------------------------------------------------
    # Tool retrieval
    # ------------------------------------------------------------------

    async def _get_tools_for_app(self, app_name: str) -> List[Any]:
        """
        Get tools for app from cache or Composio API.

        Args:
            app_name: App name (e.g., "Slack")

        Returns:
            List of ComposioActionAdapter tools
        """
        # Check cache first
        cached_tools = self.tool_cache.get(self.user_id, app_name)
        if cached_tools is not None:
            logger.info(f"[GetTools] Cache hit for {app_name} (user {self.user_id})")
            return cached_tools

        # Cache miss - fetch from Composio
        logger.info(f"[GetTools] Cache miss for {app_name}, fetching from Composio")

        try:
            # Get tools using tool manager (v3 API)
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

    # ------------------------------------------------------------------
    # LLM-driven tool selection & execution
    # ------------------------------------------------------------------

    async def _execute_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        app_name: str,
        action_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the user's prompt using LLM-driven tool selection.

        1. Build a tool catalog from available tools
        2. Ask LLM to select the best tool and extract parameters
        3. Execute the selected tool
        """
        try:
            # Build tool descriptions for LLM
            tool_info = []
            tool_map = {}
            for t in tools:
                name = getattr(t, "name", str(t))
                desc = getattr(t, "description", "")
                params = getattr(t, "parameters", {})
                tool_info.append({
                    "name": name,
                    "description": desc,
                    "parameters": params,
                })
                tool_map[name] = t

            if not tool_info:
                return {
                    "success": False,
                    "error": f"No tools available for {app_name}",
                }

            # Use LLM to select tool and extract parameters
            selection = await self._llm_select_tool(
                prompt=prompt,
                tool_catalog=tool_info,
                app_name=app_name,
                provided_params=action_params,
            )

            if not selection or not selection.get("tool_name"):
                # Fallback: use first tool with basic params
                logger.warning("[ExecuteWithTools] LLM selection failed, using fallback")
                selection = {
                    "tool_name": tool_info[0]["name"],
                    "parameters": action_params or {},
                }

            tool_name = selection["tool_name"]
            params = selection.get("parameters", {})

            # Merge with explicit action_params (user-provided take priority)
            if action_params:
                params.update(action_params)

            logger.info(f"[ExecuteWithTools] LLM selected: {tool_name} with {len(params)} params")

            # Find and execute the tool
            selected_tool = tool_map.get(tool_name)
            if not selected_tool:
                # Try case-insensitive match
                for name, tool in tool_map.items():
                    if name.upper() == tool_name.upper():
                        selected_tool = tool
                        tool_name = name
                        break

            if not selected_tool:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found in available tools. Available: {list(tool_map.keys())[:5]}",
                }

            # Execute tool directly
            result = await selected_tool.ainvoke(params)

            return {
                "success": True,
                "data": result,
                "tool_used": tool_name,
                "message": f"Successfully executed {tool_name}",
            }

        except Exception as e:
            logger.error(f"[ExecuteWithTools] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }

    async def _llm_select_tool(
        self,
        prompt: str,
        tool_catalog: List[Dict[str, Any]],
        app_name: str,
        provided_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to select the best tool and extract parameters from the prompt.

        Args:
            prompt: User's original request
            tool_catalog: List of available tool descriptions
            app_name: The app being used (for context)
            provided_params: Any parameters already provided by the caller

        Returns:
            Dict with "tool_name" and "parameters", or None on failure
        """
        try:
            from backend.services.inference_service import inference_service, InferencePriority
            from langchain_core.messages import HumanMessage, SystemMessage

            # Build a concise tool list for the LLM
            tool_list_str = "\n".join(
                f"- {t['name']}: {t.get('description', 'No description')[:120]}"
                for t in tool_catalog[:20]  # Limit to 20 tools
            )

            system_prompt = (
                "You are a tool-selection assistant. Given a user request and a list of "
                "available API tools, select the BEST matching tool and extract the "
                "required parameters from the user's message.\n\n"
                "Respond with ONLY a JSON object (no markdown, no explanation):\n"
                '{\n'
                '  "tool_name": "EXACT_TOOL_NAME_FROM_LIST",\n'
                '  "parameters": { "param1": "value1", ... },\n'
                '  "reasoning": "one line explaining why"\n'
                '}\n\n'
                "Rules:\n"
                "- tool_name MUST exactly match one of the tool names below\n"
                "- Extract parameters from the user message (names, dates, text, etc.)\n"
                "- If a parameter isn't in the message, omit it\n"
                "- For channel/workspace references like #general, use the name without #\n"
            )

            user_message = (
                f"App: {app_name}\n"
                f"User request: {prompt}\n\n"
                f"Available tools:\n{tool_list_str}"
            )

            if provided_params:
                user_message += f"\n\nAlready provided parameters: {json.dumps(provided_params)}"

            # Use inference_service instead of direct Groq client
            response = await inference_service.generate(
                messages=[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ],
                priority=InferencePriority.SPEED,
                temperature=0.0,
                max_tokens=500,
                json_mode=True,
                strip_markdown=True,
            )

            # Parse JSON from response
            result = json.loads(response)

            # Validate tool_name exists
            valid_names = {t["name"] for t in tool_catalog}
            if result.get("tool_name") not in valid_names:
                # Try case-insensitive
                for name in valid_names:
                    if name.upper() == result.get("tool_name", "").upper():
                        result["tool_name"] = name
                        break
                else:
                    logger.warning(
                        f"[LLM] Selected tool '{result.get('tool_name')}' "
                        f"not in catalog. Falling back to first tool."
                    )
                    result["tool_name"] = tool_catalog[0]["name"]

            logger.info(
                f"[LLM] Selected: {result['tool_name']} "
                f"(reason: {result.get('reasoning', 'N/A')})"
            )
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[LLM] Failed to parse response as JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"[LLM] Tool selection failed: {e}")
            return None

"""
Composio ToolSet Manager (Multi-App)

Provides tool execution, discovery, search, and session management for
authenticated users.

Uses composio v0.10.x SDK (v3 API):
    composio.tools.get(...)        → discover available tools/actions
    composio.tools.execute(...)    → execute an action for a user
    composio.create(user_id=...)   → create a Composio session with meta-tools
    composio.toolkits.list(...)    → list available toolkits
"""

import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger("composio_tools")

# Import Composio exceptions for better error messages
try:
    from composio import exceptions as composio_exceptions
    COMPOSIO_EXCEPTIONS_AVAILABLE = True
except ImportError:
    COMPOSIO_EXCEPTIONS_AVAILABLE = False
    composio_exceptions = None  # type: ignore


class ComposioActionAdapter:
    """
    Thin wrapper around a Composio action/tool.

    Exposes `.name` and `.ainvoke()` so it can be used wherever
    LangChain BaseTool objects are expected (e.g., IntegrationsAgentService).

    Also exposes `.description` and `.parameters` when available for
    LLM-driven tool selection.
    """

    def __init__(
        self,
        action_name: str,
        composio_client,
        user_id: str,
        connected_account: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = action_name
        self._client = composio_client
        self._user_id = user_id
        self._connected_account = connected_account
        self.description = description or ""
        self.parameters = parameters or {}

    async def ainvoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action via the v3 SDK tools.execute() API."""
        try:
            result = self._client.tools.execute(
                slug=self.name,
                arguments=params or {},
                connected_account_id=self._connected_account,
            )
            # Handle ToolExecutionResponse object
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            logger.error(f"Tool execution failed for {self.name}: {e}")
            raise

    def __repr__(self) -> str:
        desc_preview = self.description[:60] + "..." if len(self.description) > 60 else self.description
        return f"ComposioActionAdapter(name={self.name!r}, desc={desc_preview!r})"


class ComposioToolManager:
    """
    Manages Composio tool discovery, execution, search, and sessions.

    Uses composio v0.10.x (v3 API):
        composio.tools.get(...)        → discover tools by toolkit, slug, or search
        composio.tools.execute(...)    → execute a tool for a user
        composio.create(user_id=...)   → create a session with meta-tools
        composio.toolkits.list(...)    → list all available toolkits
    """

    def __init__(self):
        # Get COMPOSIO_API_KEY via credential_manager (DB → .env fallback)
        try:
            from backend.services.credential_service import credential_manager
            self.api_key = credential_manager.get("agent", "integrations", "COMPOSIO_API_KEY")
        except Exception:
            self.api_key = os.getenv("COMPOSIO_API_KEY")
        if not self.api_key:
            raise ValueError("COMPOSIO_API_KEY required")
        from composio import Composio
        self._composio = Composio(api_key=self.api_key)
        logger.info("Initialized Composio v3 client for tools")

    # ------------------------------------------------------------------
    # Error formatting
    # ------------------------------------------------------------------

    def _format_composio_error(self, error: Exception) -> str:
        """Convert SDK exceptions to readable messages."""
        if not COMPOSIO_EXCEPTIONS_AVAILABLE or not composio_exceptions:
            return str(error)
        if isinstance(error, getattr(composio_exceptions, "ApiKeyNotProvidedError", type(None))):
            return "Composio API key not configured."
        elif isinstance(error, getattr(composio_exceptions, "ComposioSDKTimeoutError", type(None))):
            return "Request timeout. Please try again."
        elif isinstance(error, getattr(composio_exceptions, "NoItemsFound", type(None))):
            return "No connected accounts found for user. Please connect the app first."
        err = str(error)
        if "401" in err or "unauthorized" in err.lower():
            return "Authentication failed. User may need to reconnect the app."
        if "not found" in err.lower():
            return "User not connected to this app. Please connect first."
        return err

    # ------------------------------------------------------------------
    # Connection resolution (for backward compat with existing agents)
    # ------------------------------------------------------------------

    def _resolve_connected_account_id(self, user_id: str, action_or_toolkit: str) -> Optional[str]:
        """Resolve connected account ID from our DB connection store."""
        try:
            from services.integrations.composio_auth import get_auth_manager

            token = (action_or_toolkit or "").strip().upper()
            app_slug = token.split("_", 1)[0].lower() if "_" in token else token.lower()
            if app_slug == "zoho":
                app_slug = "zohobooks"

            connection = get_auth_manager().get_connection_for_agent(user_id, app_slug)
            if not connection:
                return None
            return connection.get("connection_id")
        except Exception as e:
            logger.debug(f"Could not resolve connected_account_id for {user_id}/{action_or_toolkit}: {e}")
            return None

    # ------------------------------------------------------------------
    # Tool discovery (v3 API)
    # ------------------------------------------------------------------

    def get_tools_for_user(
        self,
        user_id: str,
        toolkits: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> List[ComposioActionAdapter]:
        """
        Get Composio tools for a user as ComposioActionAdapter objects.

        Uses composio v3 `tools.get()` API with search, toolkits, or specific tool slugs.

        Args:
            user_id: User ID (used for connection resolution)
            toolkits: App slugs like ["gmail", "github"]
            tools: Specific tool slugs like ["GMAIL_SEND_EMAIL"]
            search: Natural language search query (e.g., "send email")

        Returns:
            List of ComposioActionAdapter objects with .name and .ainvoke()
        """
        try:
            raw_tools = []

            if search:
                # Natural language search across all tools
                raw_tools = self._composio.tools.get(search=search)
                logger.info(f"Search '{search}' returned {len(raw_tools)} tools")

            elif tools:
                # Get specific tools by slug
                raw_tools = self._composio.tools.get(tools=tools)
                logger.info(f"Fetched {len(raw_tools)} specific tools")

            elif toolkits:
                # Get tools for specific toolkits
                raw_tools = self._composio.tools.get(toolkits=toolkits)
                logger.info(f"Fetched {len(raw_tools)} tools for toolkits {toolkits}")

            else:
                logger.warning("No toolkits, tools, or search specified — returning empty list")
                return []

            # Wrap raw tools into adapters
            adapters = []
            for tool in raw_tools:
                name = None
                description = ""
                parameters = {}

                if hasattr(tool, "name"):
                    name = tool.name
                elif hasattr(tool, "slug"):
                    name = tool.slug
                elif isinstance(tool, dict):
                    name = tool.get("name") or tool.get("slug")
                elif isinstance(tool, str):
                    name = tool

                if hasattr(tool, "description"):
                    description = tool.description or ""
                elif isinstance(tool, dict):
                    description = tool.get("description", "")

                if hasattr(tool, "parameters"):
                    parameters = tool.parameters or {}
                elif isinstance(tool, dict):
                    parameters = tool.get("parameters", {})

                if name:
                    # Resolve connected account for tool execution
                    connected_account = self._resolve_connected_account_id(user_id, name)
                    adapters.append(
                        ComposioActionAdapter(
                            action_name=name,
                            composio_client=self._composio,
                            user_id=user_id,
                            connected_account=connected_account,
                            description=description,
                            parameters=parameters,
                        )
                    )

            logger.info(f"Wrapped {len(adapters)} tools for user {user_id}")
            return adapters

        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to get tools for {user_id}: {error_msg}", exc_info=True)
            return []

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search Composio's tool catalog by natural language query.

        Unlike get_tools_for_user(), this returns raw tool metadata dicts
        (not adapters), suitable for display in UI or tool selection.

        Args:
            query: Natural language search (e.g., "send slack message")

        Returns:
            List of tool metadata dicts with name, description, toolkit, etc.
        """
        try:
            raw_tools = self._composio.tools.get(search=query)
            results = []
            for tool in raw_tools:
                info = {}
                if hasattr(tool, "model_dump"):
                    info = tool.model_dump()
                elif isinstance(tool, dict):
                    info = tool
                else:
                    info = {
                        "name": getattr(tool, "name", getattr(tool, "slug", str(tool))),
                        "description": getattr(tool, "description", ""),
                    }
                results.append(info)
            return results
        except Exception as e:
            logger.error(f"Tool search failed for '{query}': {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Tool execution (v3 API)
    # ------------------------------------------------------------------

    def execute_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a Composio tool for a user.

        Uses composio v3 `tools.execute()` API.

        Args:
            user_id: User ID (used to resolve connected account)
            tool_slug: Tool identifier (e.g., "GMAIL_SEND_EMAIL")
            arguments: Tool parameters
        """
        try:
            connected_account_id = self._resolve_connected_account_id(user_id, tool_slug)

            result = self._composio.tools.execute(
                slug=tool_slug.upper(),
                arguments=arguments or {},
                connected_account_id=connected_account_id,
            )

            # Handle ToolExecutionResponse
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {"result": result}

            success = result_dict.get("successful", "error" not in result_dict)
            return {
                "success": success,
                "data": result_dict.get("data", result_dict),
                "error": result_dict.get("error"),
            }

        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Tool execution failed for {user_id}/{tool_slug}: {error_msg}", exc_info=True)
            return {"success": False, "data": {}, "error": error_msg}

    # ------------------------------------------------------------------
    # Session management (v3 API)
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        toolkits: Optional[List[str]] = None,
    ) -> Any:
        """
        Create a Composio session for a user with access to meta-tools.

        Sessions provide COMPOSIO_SEARCH_TOOLS, COMPOSIO_MULTI_EXECUTE_TOOL,
        COMPOSIO_MANAGE_CONNECTIONS, etc.

        Args:
            user_id: User ID to scope the session
            toolkits: Optional list of toolkit slugs to restrict access

        Returns:
            Composio session object with .tools(), .mcp, .authorize(), .toolkits()
        """
        try:
            kwargs: Dict[str, Any] = {"user_id": user_id}
            if toolkits:
                kwargs["toolkits"] = toolkits

            session = self._composio.create(**kwargs)
            logger.info(f"Created Composio session for user {user_id} (toolkits={toolkits or 'all'})")
            return session

        except Exception as e:
            logger.error(f"Session creation failed for {user_id}: {e}", exc_info=True)
            raise

    def get_session_meta_tools(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get meta-tool schemas for LLM consumption.

        Returns the schemas for COMPOSIO_SEARCH_TOOLS, COMPOSIO_MULTI_EXECUTE_TOOL,
        COMPOSIO_MANAGE_CONNECTIONS, etc. — formatted as tool definitions that can be
        passed to an LLM.

        Args:
            user_id: User ID for session scoping

        Returns:
            List of tool schema dicts for LLM function calling
        """
        try:
            raw_tools = self._composio.tools.get_raw_tool_router_meta_tools()
            schemas = []
            for tool in raw_tools:
                if hasattr(tool, "model_dump"):
                    schemas.append(tool.model_dump())
                elif isinstance(tool, dict):
                    schemas.append(tool)
                else:
                    schemas.append({"name": str(tool)})
            logger.info(f"Retrieved {len(schemas)} meta-tool schemas")
            return schemas
        except Exception as e:
            logger.error(f"Failed to get meta-tool schemas: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Toolkit discovery (v3 API)
    # ------------------------------------------------------------------

    def list_toolkits(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available Composio toolkits (apps).

        Args:
            search: Optional search query to filter toolkits

        Returns:
            List of toolkit info dicts
        """
        try:
            if search:
                toolkits = self._composio.toolkits.list(search=search)
            else:
                toolkits = self._composio.toolkits.list()

            results = []
            if hasattr(toolkits, "items"):
                items = toolkits.items
            elif isinstance(toolkits, list):
                items = toolkits
            else:
                items = [toolkits] if toolkits else []

            for tk in items:
                if hasattr(tk, "model_dump"):
                    results.append(tk.model_dump())
                elif isinstance(tk, dict):
                    results.append(tk)
                else:
                    results.append({"name": str(tk)})

            return results
        except Exception as e:
            logger.error(f"Failed to list toolkits: {e}", exc_info=True)
            return []

    def get_toolkit_connection_status(
        self,
        user_id: str,
        toolkit_slugs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get connection status for toolkits for a specific user.

        Args:
            user_id: User ID
            toolkit_slugs: Optional list of specific toolkit slugs to check

        Returns:
            List of dicts with toolkit name, slug, and connection status
        """
        try:
            session = self.create_session(user_id)
            toolkits = session.toolkits()

            results = []
            if hasattr(toolkits, "items"):
                items = toolkits.items
            elif isinstance(toolkits, list):
                items = toolkits
            else:
                items = [toolkits] if toolkits else []

            for tk in items:
                slug = getattr(tk, "slug", getattr(tk, "name", ""))
                if toolkit_slugs and slug not in toolkit_slugs:
                    continue

                is_active = False
                connected_account_id = None
                if hasattr(tk, "connection"):
                    conn = tk.connection
                    is_active = getattr(conn, "is_active", False)
                    if hasattr(conn, "connected_account"):
                        connected_account_id = getattr(conn.connected_account, "id", None)

                results.append({
                    "name": getattr(tk, "name", slug),
                    "slug": slug,
                    "is_connected": is_active,
                    "connected_account_id": connected_account_id,
                })

            return results
        except Exception as e:
            logger.error(f"Failed to get toolkit status for {user_id}: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Convenience helpers (backward compat)
    # ------------------------------------------------------------------

    def get_invoice_tools_for_user(self, user_id: str) -> List[ComposioActionAdapter]:
        """Convenience: get Zoho Books (invoice) actions for a user."""
        return self.get_tools_for_user(user_id, toolkits=["zohobooks"])


# ---------------------------------------------------------------------------
# Singleton + convenience helpers
# ---------------------------------------------------------------------------

_tool_manager: Optional[ComposioToolManager] = None


def get_tool_manager() -> ComposioToolManager:
    """Get or create singleton ComposioToolManager."""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ComposioToolManager()
    return _tool_manager


def get_tools_for_user(
    user_id: str,
    toolkits: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
) -> List[ComposioActionAdapter]:
    """Convenience wrapper around ComposioToolManager.get_tools_for_user()."""
    return get_tool_manager().get_tools_for_user(user_id, toolkits=toolkits, tools=tools)


def search_tools(query: str) -> List[Dict[str, Any]]:
    """Convenience wrapper around ComposioToolManager.search_tools()."""
    return get_tool_manager().search_tools(query)

"""
Composio ToolSet Manager (Multi-App)

Provides tool execution and discovery for authenticated users.
Uses composio_core v0.7.x SDK (composio.actions API).
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
    Thin wrapper around a Composio action name (v0.7.x SDK).

    Exposes `.name` and `.ainvoke()` so it can be used wherever
    LangChain BaseTool objects are expected (e.g. IntegrationsAgentService).
    """

    def __init__(self, action_name: str, composio_client, entity_id: str, connected_account: Optional[str] = None):
        self.name = action_name
        self._client = composio_client
        self._entity_id = entity_id
        self._connected_account = connected_account

    async def ainvoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action via the v0.7.x SDK."""
        from composio import Action
        result = self._client.actions.execute(
            action=Action[self.name],
            params=params or {},
            entity_id=self._entity_id,
            connected_account=self._connected_account,
        )
        return result

    def __repr__(self) -> str:
        return f"ComposioActionAdapter(name={self.name!r})"


class ComposioToolManager:
    """
    Manages Composio action discovery and execution for authenticated users.

    Uses composio_core v0.7.x:
        composio.actions.get(apps=[...])   → discover available actions
        composio.actions.execute(...)      → run an action for a user
    """

    def __init__(self):
        self.api_key = os.getenv("COMPOSIO_API_KEY")
        if not self.api_key:
            raise ValueError("COMPOSIO_API_KEY required")
        from composio import Composio
        self._composio = Composio(api_key=self.api_key)
        logger.info("Initialized Composio client for tools")

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

    def _resolve_connected_account_id(self, user_id: str, action_or_toolkit: str) -> Optional[str]:
        """Resolve connected account id from our DB connection store."""
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

    def get_tools_for_user(
        self,
        user_id: str,
        toolkits: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
    ) -> List[ComposioActionAdapter]:
        """
        Get Composio actions for a user as ComposioActionAdapter objects.

        Uses composio.actions.get(apps=[toolkit]) from the v0.7.x SDK.
        Returns objects with .name and .ainvoke() compatible with
        IntegrationsAgentService._execute_with_tools().

        Args:
            user_id: User ID (used as entity_id for execution)
            toolkits: App slugs like ["gmail", "github"]
            tools: Specific action slugs like ["GMAIL_SEND_EMAIL"]
        """
        try:
            if tools:
                adapters = [
                    ComposioActionAdapter(
                        slug.upper(),
                        self._composio,
                        user_id,
                        connected_account=self._resolve_connected_account_id(user_id, slug),
                    )
                    for slug in tools
                ]
                logger.info(f"Wrapped {len(adapters)} specific actions for user {user_id}")
                return adapters

            if toolkits:
                adapters = []
                for toolkit in toolkits:
                    action_models = self._composio.actions.get(
                        apps=[toolkit.upper()],
                        limit=20,
                    )
                    if not isinstance(action_models, list):
                        action_models = [action_models] if action_models else []
                    for am in action_models:
                        name = getattr(am, "name", None)
                        if name:
                            adapters.append(
                                ComposioActionAdapter(
                                    name,
                                    self._composio,
                                    user_id,
                                    connected_account=self._resolve_connected_account_id(user_id, toolkit),
                                )
                            )
                logger.info(f"Fetched {len(adapters)} actions for {toolkits} (user {user_id})")
                return adapters

            logger.warning("No toolkits or tools specified — returning empty list")
            return []

        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to get tools for {user_id}: {error_msg}", exc_info=True)
            return []

    def execute_tool(
        self,
        user_id: str,
        tool_slug: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a Composio action for a user.

        Uses composio.actions.execute(action=Action[slug], params=..., entity_id=...)
        from the v0.7.x SDK.
        """
        try:
            from composio import Action
            result = self._composio.actions.execute(
                action=Action[tool_slug.upper()],
                params=arguments or {},
                entity_id=user_id,
                connected_account_id=self._resolve_connected_account_id(user_id, tool_slug),
            )
            success = "error" not in result if isinstance(result, dict) else True
            return {
                "success": success,
                "data": result if isinstance(result, dict) else {"result": result},
                "error": result.get("error") if isinstance(result, dict) else None,
            }
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Tool execution failed for {user_id}/{tool_slug}: {error_msg}", exc_info=True)
            return {"success": False, "data": {}, "error": error_msg}

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

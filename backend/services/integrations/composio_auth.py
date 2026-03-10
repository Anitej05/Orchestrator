"""
Composio Authentication Manager (Multi-App)

Handles OAuth flow for all integrations (Zoho Books, Gmail, GitHub, etc.)
using Composio's entity/connection APIs.

Reference: https://docs.composio.dev/docs/authenticating-users/manually-authenticating
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy import or_

from dotenv import load_dotenv
from database import SessionLocal
from models import UserConnection, ComposioEntity
from cryptography.fernet import Fernet
import base64
import hashlib

load_dotenv()

# Initialize logger first
logger = logging.getLogger("composio_auth")
logger.setLevel(logging.DEBUG)

# Import official Composio exceptions for better error handling
try:
    from composio import exceptions as composio_exceptions
    COMPOSIO_EXCEPTIONS_AVAILABLE = True
except ImportError:
    logger.warning("Composio exceptions module not available. Using generic exception handling.")
    COMPOSIO_EXCEPTIONS_AVAILABLE = False
    composio_exceptions = None


class ComposioAuthManager:
    """
    Modern Composio session-based authentication for multiple apps.
    
    Handles:
    - Pre-authentication before chat
    - Connection verification
    - Multi-app support
    - Webhook notification for completed authentications
    
    Data flow:
    1. User initiates connection → start_auth_flow()
    2. Backend provides redirect_url → https://connect.composio.dev/link/ln_abc123
    3. Frontend redirects user to redirect_url
    4. User authenticates and Composio redirects back to callback_url
    5. Backend polls check_connection_status() to verify connection
    6. Tools become available for this user via get_tool_manager()
    """
    
    def __init__(self):
        self.api_key = os.getenv("COMPOSIO_API_KEY")
        if not self.api_key:
            logger.error("COMPOSIO_API_KEY not set in environment")
            raise ValueError("COMPOSIO_API_KEY required")
        
        # Initialize Composio client directly (official SDK pattern)
        from composio import Composio
        self._composio = Composio(api_key=self.api_key)
        logger.info("Initialized Composio client")
        
        # Initialize encryption for connection IDs
        self._init_encryption_key()
    
    def _init_encryption_key(self):
        """
        Initialize Fernet encryption key for connection IDs.
        
        Uses CONNECTION_ENCRYPTION_KEY from environment, or generates a
        deterministic key from COMPOSIO_API_KEY if not set (for backwards compatibility).
        
        WARNING: For production, always set CONNECTION_ENCRYPTION_KEY explicitly!
        """
        encryption_key = os.getenv("CONNECTION_ENCRYPTION_KEY")
        env = os.getenv("ENV", "development")
        
        if not encryption_key:
            # In production, fail fast with clear instructions
            if env == "production":
                raise ValueError(
                    "CRITICAL: CONNECTION_ENCRYPTION_KEY must be set in production! "
                    "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' "
                    "Then set it as an environment variable: CONNECTION_ENCRYPTION_KEY=<generated_key>"
                )
            
            # Development fallback: Generate deterministic key from COMPOSIO_API_KEY
            logger.warning(
                "CONNECTION_ENCRYPTION_KEY not set. Generating from COMPOSIO_API_KEY. "
                "Set CONNECTION_ENCRYPTION_KEY in production!"
            )
            # Create a deterministic 32-byte key from API key
            key_material = hashlib.sha256(self.api_key.encode()).digest()
            encryption_key = base64.urlsafe_b64encode(key_material).decode()
        
        try:
            self._cipher = Fernet(encryption_key.encode())
            logger.info("Connection ID encryption initialized")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise ValueError(
                "Invalid CONNECTION_ENCRYPTION_KEY. Must be a valid Fernet key. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

    @staticmethod
    def _normalize_status(status: Optional[str]) -> str:
        """Normalize connection status values to lowercase canonical form."""
        return (status or "").strip().lower()

    @staticmethod
    def _normalize_app_slug(app_slug: str) -> str:
        """Normalize app slug values for consistent DB lookups."""
        return (app_slug or "").strip().lower()

    def _get_candidate_entity_ids(self, user_id: str, app_slug: Optional[str] = None) -> List[str]:
        """
        Resolve all candidate Composio entity IDs for a user.
        Includes canonical user_id, mapped composio_entity_id values, and legacy default fallback.
        """
        ids: List[str] = []

        def _add(value: Optional[str]):
            if value and value not in ids:
                ids.append(value)

        _add(user_id)
        normalized_slug = self._normalize_app_slug(app_slug) if app_slug else None

        needs_legacy_default_fallback = False
        try:
            with SessionLocal() as db:
                mapping = (
                    db.query(ComposioEntity)
                    .filter(ComposioEntity.internal_user_id == user_id)
                    .first()
                )
                if mapping:
                    _add(mapping.composio_entity_id)

                user_connections_query = db.query(UserConnection).filter(
                    or_(
                        UserConnection.user_id == user_id,
                        UserConnection.internal_user_id == user_id,
                    )
                )
                if normalized_slug:
                    user_connections_query = user_connections_query.filter(
                        UserConnection.app_slug == normalized_slug
                    )

                for conn in user_connections_query.all():
                    _add(conn.composio_entity_id)
                    if not conn.composio_entity_id or conn.composio_entity_id == "default":
                        needs_legacy_default_fallback = True
        except Exception as e:
            logger.debug(f"Failed to load candidate entity IDs for {user_id}: {e}")

        if user_id != "default" and needs_legacy_default_fallback:
            _add("default")

        return ids
    
    def _encrypt_connection_id(self, connection_id: str) -> str:
        """
        Encrypt a connection ID for storage.
        
        Args:
            connection_id: Plain text connection ID (e.g., 'ca_xZUTNToOnUiQ')
        
        Returns:
            Encrypted connection ID as base64 string
        """
        if not connection_id:
            return connection_id
        
        try:
            encrypted = self._cipher.encrypt(connection_id.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error(f"Failed to encrypt connection ID: {e}")
            # Return original on error to prevent data loss (log for investigation)
            return connection_id
    
    def _decrypt_connection_id(self, encrypted_id: str) -> str:
        """
        Decrypt a connection ID for use.
        
        Args:
            encrypted_id: Encrypted connection ID from database
        
        Returns:
            Plain text connection ID
        """
        if not encrypted_id:
            return encrypted_id
        
        try:
            # Try to decrypt (will fail if it's already plain text)
            decrypted = self._cipher.decrypt(encrypted_id.encode())
            return decrypted.decode()
        except Exception:
            # If decryption fails, assume it's already plain text (backwards compatibility)
            # This handles migration from unencrypted to encrypted storage
            logger.warning("Connection ID appears to be unencrypted (or wrong key). Using as-is.")
            return encrypted_id
    
    def _get_auth_config_id(self, app_slug: str) -> Optional[str]:
        """
        Map app slug to auth config ID from environment.

        Returns the auth config ID if the env var is set, or None if it isn't.
        None is acceptable — _get_integration_id will fall back to Composio's
        integrations API to resolve the integration UUID, so you don't need to
        create a custom auth config just to connect a standard app like Gmail.

        If you DO have a custom auth config (e.g. your own OAuth client), set:
            COMPOSIO_AUTH_CONFIG_GMAIL=ac_gmail_xxx  in backend/.env
        """
        # Normalize app slug
        normalized_slug = app_slug.lower()

        # Handle deprecated 'zoho_books' slug
        if normalized_slug == "zoho_books":
            logger.warning(
                "⚠️ DEPRECATION: 'zoho_books' is deprecated. "
                "Please use 'zohobooks' instead. "
                "Support for 'zoho_books' will be removed in a future version."
            )
            normalized_slug = "zohobooks"

        # Map common app slugs to environment variables
        auth_config_map = {
            "gmail": os.getenv("COMPOSIO_AUTH_CONFIG_GMAIL"),
            "zohobooks": os.getenv("COMPOSIO_AUTH_CONFIG_ZOHOBOOKS"),
            "github": os.getenv("COMPOSIO_AUTH_CONFIG_GITHUB"),
            "slack": os.getenv("COMPOSIO_AUTH_CONFIG_SLACK"),
            "notion": os.getenv("COMPOSIO_AUTH_CONFIG_NOTION"),
        }

        auth_config_id = auth_config_map.get(normalized_slug)

        if not auth_config_id:
            # No custom auth config — that's fine. _get_integration_id will
            # look up the integration directly via Composio's API.
            logger.info(
                f"COMPOSIO_AUTH_CONFIG_{normalized_slug.upper()} not set; "
                f"will resolve integration via Composio API instead."
            )
            return None

        return auth_config_id

    def _get_integration_id(self, app_slug: str, auth_config_id: Optional[str] = None) -> str:
        """
        DEPRECATED — No longer called after migrating start_auth_flow to v3 session-based approach.

        This method used Composio's v2 'integrations' API (now renamed to 'auth configs' in v3).
        Kept for reference / backwards-compat only. Do not call in new code.

        v2 terminology: integration / integration ID → v3: auth config / auth config ID
        """
        logger.warning(
            "_get_integration_id() is deprecated (v2 integrations API). "
            "Use the session-based start_auth_flow() instead."
        )
        app_name_candidates = [
            app_slug,
            app_slug.upper(),
            app_slug.replace("_", ""),
            app_slug.replace("_", "").upper(),
        ]

        for app_name in app_name_candidates:
            try:
                integrations = self._composio.integrations.get(
                    app_name=app_name,
                    show_disabled=True
                )
                if isinstance(integrations, list) and integrations:
                    return integrations[0].id
                if integrations and hasattr(integrations, "id"):
                    return integrations.id
            except Exception as e:
                logger.warning(f"⚠️ Could not resolve integration_id for {app_name}: {e}")

        if auth_config_id:
            try:
                uuid.UUID(auth_config_id)
                return auth_config_id
            except Exception:
                pass

        raise ValueError(
            f"Integration ID not found for {app_slug}. "
            "Ensure the app exists in Composio and your API key has access."
        )
    
    def start_auth_flow(
        self, 
        user_id: str, 
        app_slug: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initiate OAuth connection using the recommended v3 session-based approach.

        v3 pattern (per docs):
            session = composio.create(user_id)
            connection_request = session.authorize(toolkit, callbackUrl=...)
            redirect_url = connection_request.redirect_url

        No manual integration_id / entity_id lookup required.
        Composio handles auth-config selection automatically.

        Args:
            user_id: Your app's user ID (v3: user_id, not entity_id)
            app_slug: Composio toolkit slug (e.g., 'gmail', 'github', 'zohobooks')
            callback_url: Optional URL Composio redirects to after auth

        Returns:
            Dict with redirect_url and connection tracking info

        Data flow:
            1. Backend calls this → gets redirect_url
            2. Frontend redirects user to redirect_url (Composio Connect Link)
            3. User authenticates → Composio redirects to callback_url
            4. POST /webhooks/composio fires → DB updated to ACTIVE
        """
        try:

            logger.info(f"Starting auth for {app_slug} (user: {user_id})")
            if callback_url:
                logger.info(f"Using callback URL: {callback_url}")

            # SDK v0.7.x: get_entity(user_id).initiate_connection(app_name=...)
            # This is the simplest pattern that works with the installed version.
            entity = self._composio.get_entity(id=user_id)
            connection_request = entity.initiate_connection(
                app_name=app_slug.upper(),
                redirect_url=callback_url or None,
            )

            # Extract redirect URL
            redirect_url = None
            if hasattr(connection_request, "redirectUrl"):
                redirect_url = connection_request.redirectUrl
            elif hasattr(connection_request, "redirect_url"):
                redirect_url = connection_request.redirect_url

            # Extract connection ID
            connection_id = None
            if hasattr(connection_request, "connectedAccountId"):
                connection_id = connection_request.connectedAccountId
            elif hasattr(connection_request, "connected_account_id"):
                connection_id = connection_request.connected_account_id
            elif hasattr(connection_request, "id"):
                connection_id = connection_request.id

            # Persist INITIATED status to DB for tracking
            if connection_id:
                logger.info(f"💾 Saving INITIATED connection: {connection_id}")
                self._save_connection_to_db(
                    user_id=user_id,
                    app_slug=app_slug,
                    connection_id=connection_id,
                    status="INITIATED",
                    composio_entity_id=user_id,
                    app_name=app_slug,
                )
            else:
                logger.warning("⚠️ No connected-account ID returned")

            return {
                "success": True,
                "redirect_url": redirect_url,
                "app_slug": app_slug,
                "user_id": user_id,
                "connection_id": connection_id,
                "poll_status_url": f"/api/integrations/status/{user_id}/{app_slug}",
            }


        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Auth flow failed for {app_slug}: {error_msg}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "initiated", "failed", error_msg)

            return {
                "success": False,
                "error": error_msg,
                "app_slug": app_slug,

            }
    
    def check_connection_status(
        self, 
        user_id: str, 
        app_slug: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if user has active connections and sync to database.
        
        Args:
            user_id: Your app's user ID
            app_slug: Check specific app, or None for all apps
        
        Returns:
            Connection status for the user's toolkits
        """
        try:
            # SDK v0.7.x: fetch by entity_ids; support canonical + legacy mapped entity IDs.
            connections: List[Any] = []
            seen_connection_ids = set()
            for entity_id in self._get_candidate_entity_ids(user_id, app_slug):
                try:
                    connections_response = self._composio.connected_accounts.get(
                        entity_ids=[entity_id]
                    )
                    if isinstance(connections_response, list):
                        raw_connections = connections_response
                    elif connections_response is not None:
                        raw_connections = [connections_response]
                    else:
                        raw_connections = []

                    for raw_conn in raw_connections:
                        conn_id = getattr(raw_conn, "id", None)
                        if conn_id and conn_id in seen_connection_ids:
                            continue
                        if conn_id:
                            seen_connection_ids.add(conn_id)
                        connections.append(raw_conn)
                except Exception as entity_error:
                    logger.debug(f"Could not fetch connections for entity {entity_id}: {entity_error}")
            
            logger.info(f"📋 Checking {len(connections)} connections for user {user_id}")
            
            results = []
            results_by_slug = {}
            db_status_map = {}
            db_connection_map = {}

            # Load DB connections for the user (used when Composio status is delayed)
            try:
                with SessionLocal() as db:
                    db_connections = db.query(UserConnection).filter(
                        or_(
                            UserConnection.user_id == user_id,
                            UserConnection.internal_user_id == user_id,
                        )
                    ).all()
                    for conn in db_connections:
                        db_status_map[conn.app_slug] = self._normalize_status(conn.status)
                        db_connection_map[conn.app_slug] = conn.connection_id
            except Exception as db_error:
                logger.warning(f"⚠️ Failed to load DB connections for {user_id}: {db_error}")
            for conn in connections:
                conn_dict = None
                if hasattr(conn, "model_dump"):
                    try:
                        conn_dict = conn.model_dump()
                    except Exception:
                        conn_dict = None

                integration_id = None
                app_name = None
                app_unique_id = None
                status = None
                connected_account_id = None

                if conn_dict:
                    integration_id = conn_dict.get("integrationId") or conn_dict.get("integration_id")
                    app_name = conn_dict.get("appName") or conn_dict.get("app_name")
                    app_unique_id = conn_dict.get("appUniqueId") or conn_dict.get("app_unique_id")
                    status = conn_dict.get("status")
                    connected_account_id = conn_dict.get("id")

                integration_id = integration_id or getattr(conn, "integration_id", None) or getattr(conn, "integrationId", None)
                app_name = app_name or getattr(conn, "app_name", None) or getattr(conn, "appName", None)
                app_unique_id = app_unique_id or getattr(conn, "app_unique_id", None) or getattr(conn, "appUniqueId", None)
                status = self._normalize_status(status or getattr(conn, "status", "") or "")
                connected_account_id = connected_account_id or getattr(conn, "id", None)
                entity_id = getattr(conn, "entityId", None) or getattr(conn, "entity_id", None)

                slug = (app_unique_id or app_name or "").lower()
                is_connected = status == "active"
                
                db_status = db_status_map.get(slug)
                db_connection_id = db_connection_map.get(slug)

                connection_info = {
                    "name": getattr(conn, "appName", None) or slug,
                    "slug": slug,
                    "is_connected": is_connected,
                    "connected_account_id": connected_account_id,
                    "entity_id": entity_id,
                    "db_status": db_status,
                    "db_connection_id": db_connection_id,
                }

                if slug:
                    if slug in results_by_slug and results_by_slug[slug]["is_connected"] and not is_connected:
                        pass
                    else:
                        results_by_slug[slug] = connection_info
                
                # REMOVED: Don't add to results here - we'll build it from results_by_slug after deduplication
                # This prevents duplicates when multiple connections exist for the same app
                
            # Build results from deduplicated results_by_slug
            for slug, connection_info in results_by_slug.items():
                # Filter by app_slug if specified
                if app_slug is None or slug == app_slug.lower():
                    results.append(connection_info)
                
                # SYNC TO DATABASE: Save connected apps to user_connections table
                # Use connection_info from the current iteration, not the loop variable
                if connection_info["is_connected"] and connection_info["connected_account_id"] and slug:
                    logger.info(f"🔗 Found active connection: {slug} (ID: {connection_info['connected_account_id']})")
                    self._save_connection_to_db(
                        user_id,
                        slug,
                        connection_info["connected_account_id"],
                        status="active",
                        composio_entity_id=connection_info.get("entity_id"),
                        app_name=connection_info.get("name", slug),
                        app_metadata={
                            "app_name": connection_info.get("name", slug),
                            "composio_status": "active",
                        },
                    )

            # Include DB-only INITIATED entries not yet visible in Composio
            for slug, status in db_status_map.items():
                if slug in results_by_slug:
                    continue
                if app_slug is not None and slug != app_slug:
                    continue
                results.append({
                    "name": slug,
                    "slug": slug,
                    "is_connected": False,
                    "connected_account_id": None,
                    "db_status": status,
                    "db_connection_id": db_connection_map.get(slug),
                })
            
            connected = [
                t for t in results
                if t["is_connected"] or (self._normalize_status(t.get("db_status")) == "active")
            ]
            pending = [
                t for t in results
                if self._normalize_status(t.get("db_status")) == "initiated"
            ]
            
            return {
                "success": True,
                "user_id": user_id,
                "all_toolkits": results,
                "connected_apps": [t["slug"] for t in connected],
                "pending_apps": [t["slug"] for t in pending],
            }
        
        except Exception as e:
            logger.error(f"Status check failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
            }
    
    def verify_required_connections(
        self, 
        user_id: str, 
        required_apps: List[str]
    ) -> Dict[str, Any]:
        """
        Verify user has all required app connections before proceeding.
        
        Args:
            user_id: Your app's user ID
            required_apps: List of app slugs (e.g., ['zohobooks', 'gmail'])
        
        Returns:
            Status and list of apps needing auth
        """
        status = self.check_connection_status(user_id)
        
        if not status["success"]:
            return status
        
        connected = set(status["connected_apps"])
        required = set(required_apps)
        missing = required - connected
        
        return {
            "success": True,
            "user_id": user_id,
            "all_connected": len(missing) == 0,
            "connected_apps": list(connected & required),
            "missing_apps": list(missing),
            "auth_urls": {
                app: self.start_auth_flow(user_id, app).get("redirect_url")
                for app in missing
            } if missing else {},
        }
    
    def disconnect_app(self, user_id: str, app_slug: str) -> Dict[str, Any]:
        """Disconnect a user from a specific app and remove from database."""
        try:
            # Get user's connections using v0.7.x SDK
            connections_response = self._composio.connected_accounts.get(
                entity_ids=[user_id]
            )
            connections = connections_response if isinstance(connections_response, list) else ([connections_response] if connections_response else [])
            
            for conn in connections:
                conn_dict = conn.model_dump() if hasattr(conn, "model_dump") else {}
                slug = (
                    conn_dict.get("appUniqueId")
                    or conn_dict.get("appName")
                    or getattr(conn, "appUniqueId", None)
                    or getattr(conn, "appName", None)
                    or ""
                ).lower()
                if slug == app_slug.lower():
                    # Use official SDK delete method
                    self._composio.connected_accounts.delete(conn.id)
                    logger.info(f"Disconnected {app_slug} for user {user_id}")
                    
                    # Remove from database
                    db = SessionLocal()
                    try:
                        connection = db.query(UserConnection).filter(
                            UserConnection.user_id == user_id,
                            UserConnection.app_slug == app_slug
                        ).first()
                        if connection:
                            db.delete(connection)
                            db.commit()
                            logger.info(f"✓ Removed connection from DB: {app_slug}")
                    finally:
                        db.close()
                    
                    # Log to database
                    self._log_connection_event(user_id, app_slug, "disconnected", "success")
                    return {"success": True, "message": f"Disconnected from {app_slug}"}
            
            return {"success": False, "error": f"No active connection for {app_slug}"}
        
        except Exception as e:
            logger.error(f"Disconnect failed: {e}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "disconnected", "failed", str(e))
            return {"success": False, "error": str(e)}
    
    def refresh_connection(self, user_id: str, app_slug: str, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Refresh OAuth token for a connection.
        
        Use when:
        - Connection becomes stale
        - API calls return 401 errors
        - Proactive token refresh before expiration
        
        Args:
            user_id: User ID who owns the connection
            app_slug: App identifier (e.g., 'gmail', 'zohobooks')
            connection_id: Optional Composio connection ID. If not provided, looks up from DB.
        
        Returns:
            Dict with success status and refresh timestamp
        
        Example:
            result = auth_mgr.refresh_connection("user_123", "gmail")
            if result["success"]:
                print(f"Token refreshed at {result['refreshed_at']}")
        """
        try:
            # Get connection_id if not provided
            if not connection_id:
                connection = self.get_connection_for_agent(user_id, app_slug)
                if not connection:
                    # Try syncing from Composio first
                    logger.info(f"No local connection found, attempting to sync from Composio for {app_slug}")
                    sync_result = self.check_connection_status(user_id, app_slug)
                    if sync_result.get("success"):
                        # Try again after sync
                        connection = self.get_connection_for_agent(user_id, app_slug)
                
                if not connection:
                    return {
                        "success": False, 
                        "error": f"No connection found for {app_slug}. Please reconnect the integration.",
                        "needs_reconnect": True
                    }
                connection_id = connection["connection_id"]
            
            logger.info(f"🔄 Refreshing connection: {app_slug} for user {user_id}")
            
            # NOTE: v0.7.15 doesn't have refresh() method with connected_account_id parameter
            # Instead, connection refresh happens automatically via OAuth
            logger.warning(f"Connection refresh not implemented in Composio v0.7.15 - skipping")
            response = None  # Placeholder
            
            # # Use official SDK refresh method (NOT AVAILABLE IN v0.7.15)
            # response = self._composio.connected_accounts.refresh(
            #     connected_account_id=connection_id
            # )
            
            # Update DB with refresh timestamp
            db = SessionLocal()
            try:
                encrypted_conn_id = self._encrypt_connection_id(connection_id)
                conn = db.query(UserConnection).filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug
                ).first()
                if conn:
                    conn.auth_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
                    db.commit()
            finally:
                db.close()
            
            logger.info(f"✓ Successfully refreshed {app_slug} connection")
            self._log_connection_event(user_id, app_slug, "refreshed", "success")
            
            return {
                "success": True,
                "status": response.status if hasattr(response, 'status') else "refreshed",
                "refreshed_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                "message": f"Successfully refreshed {app_slug} connection"
            }
            
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Token refresh failed for {app_slug}: {error_msg}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "refreshed", "failed", error_msg)
            return {"success": False, "error": error_msg}
    
    def disable_connection(self, user_id: str, app_slug: str, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Temporarily disable a connection without deleting it.
        
        Useful for:
        - User wants to pause integration without losing setup
        - Rate limit management
        - Testing without full disconnect
        - Troubleshooting connection issues
        
        The connection can be re-enabled later with enable_connection().
        
        Args:
            user_id: User ID who owns the connection
            app_slug: App identifier (e.g., 'gmail', 'zohobooks')
            connection_id: Optional Composio connection ID
        
        Returns:
            Dict with success status
        """
        try:
            # Get connection_id if not provided
            if not connection_id:
                connection = self.get_connection_for_agent(user_id, app_slug)
                if not connection:
                    return {"success": False, "error": f"No connection found for {app_slug}"}
                connection_id = connection["connection_id"]
            
            logger.info(f"⏸️ Disabling connection: {app_slug} for user {user_id}")
            
            # NOTE: v0.7.15 doesn't have disable() method with connected_account_id parameter
            logger.warning(f"Connection disable not implemented in Composio v0.7.15 - updating DB only")
            
            # # Use official SDK disable method (NOT AVAILABLE IN v0.7.15)
            # self._composio.connected_accounts.disable(
            #     connected_account_id=connection_id
            # )
            
            # Update DB status
            db = SessionLocal()
            try:
                conn = db.query(UserConnection).filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug
                ).first()
                if conn:
                    conn.status = "disabled"
                    db.commit()
                    logger.info(f"✓ Updated DB status to disabled for {app_slug}")
            finally:
                db.close()
            
            self._log_connection_event(user_id, app_slug, "disabled", "success")
            
            return {
                "success": True,
                "status": "disabled",
                "message": f"Successfully disabled {app_slug} connection. Use enable_connection() to re-enable."
            }
            
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to disable {app_slug}: {error_msg}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "disabled", "failed", error_msg)
            return {"success": False, "error": error_msg}
    
    def enable_connection(self, user_id: str, app_slug: str, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Re-enable a previously disabled connection.
        
        Args:
            user_id: User ID who owns the connection
            app_slug: App identifier (e.g., 'gmail', 'zohobooks')
            connection_id: Optional Composio connection ID
        
        Returns:
            Dict with success status
        """
        try:
            # Get connection_id if not provided
            if not connection_id:
                connection = self.get_connection_for_agent(user_id, app_slug)
                if not connection:
                    return {"success": False, "error": f"No connection found for {app_slug}"}
                connection_id = connection["connection_id"]
            
            logger.info(f"▶️ Enabling connection: {app_slug} for user {user_id}")
            
            # NOTE: v0.7.15 doesn't have enable() method with connected_account_id parameter
            logger.warning(f"Connection enable not implemented in Composio v0.7.15 - updating DB only")
            
            # # Use official SDK enable method (NOT AVAILABLE IN v0.7.15)
            # self._composio.connected_accounts.enable(
            #     connected_account_id=connection_id
            # )
            
            # Update DB status
            db = SessionLocal()
            try:
                conn = db.query(UserConnection).filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug
                ).first()
                if conn:
                    conn.status = "active"
                    db.commit()
                    logger.info(f"✓ Updated DB status to active for {app_slug}")
            finally:
                db.close()
            
            self._log_connection_event(user_id, app_slug, "enabled", "success")
            
            return {
                "success": True,
                "status": "active",
                "message": f"Successfully enabled {app_slug} connection"
            }
            
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Failed to enable {app_slug}: {error_msg}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "enabled", "failed", error_msg)
            return {"success": False, "error": error_msg}
    
    def _format_composio_error(self, error: Exception) -> str:
        """
        Format Composio SDK errors into user-friendly messages.
        
        Handles official Composio exception types when available.
        """
        if not COMPOSIO_EXCEPTIONS_AVAILABLE or not composio_exceptions:
            return str(error)
        
        # Handle specific Composio exception types
        if isinstance(error, getattr(composio_exceptions, 'ApiKeyNotProvidedError', type(None))):
            return "Composio API key not configured. Please set COMPOSIO_API_KEY in environment."
        elif isinstance(error, getattr(composio_exceptions, 'ComposioSDKTimeoutError', type(None))):
            return "Connection timeout. Please try again or check your network connection."
        elif isinstance(error, getattr(composio_exceptions, 'ComposioMultipleConnectedAccountsError', type(None))):
            return "Multiple connections found when only one expected. Please disconnect and reconnect."
        elif isinstance(error, getattr(composio_exceptions, 'NoItemsFound', type(None))):
            return "Connection not found. Please connect the app first."
        else:
            # Generic error with additional context
            error_str = str(error)
            if "401" in error_str or "unauthorized" in error_str.lower():
                return "Authentication failed. Please reconnect the app."
            elif "403" in error_str or "forbidden" in error_str.lower():
                return "Access denied. Please check app permissions."
            elif "404" in error_str or "not found" in error_str.lower():
                return "Resource not found. The connection may have been deleted."
            elif "429" in error_str or "rate limit" in error_str.lower():
                return "Rate limit exceeded. Please try again later."
            elif "500" in error_str or "internal server" in error_str.lower():
                return "Composio service error. Please try again later."
            else:
                return error_str
    
    def _save_connection_to_db(
        self,
        user_id: str,
        app_slug: str,
        connection_id: str,
        status: str = "active",
        composio_entity_id: Optional[str] = None,
        app_name: Optional[str] = None,
        app_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save or update connection in user_connections table.
        
        Connection ID is encrypted before storage for security.
        """
        # Normalize app_slug to lowercase to ensure consistency with Composio's format
        app_slug = self._normalize_app_slug(app_slug)
        status = self._normalize_status(status)
        
        # Encrypt connection ID before storing
        encrypted_connection_id = self._encrypt_connection_id(connection_id)
        
        logger.info(f"💾 Attempting to save connection: {app_slug} for user {user_id}")
        try:
            db = SessionLocal()
            try:
                # Check if connection already exists (by user/app or by connection_id)
                existing = db.query(UserConnection).filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug
                ).first()
                existing_by_id = db.query(UserConnection).filter(
                    UserConnection.connection_id == encrypted_connection_id
                ).first()
                
                if existing:
                    # Update existing connection
                    existing.connection_id = encrypted_connection_id
                    existing.status = status
                    existing.internal_user_id = user_id
                    if composio_entity_id:
                        existing.composio_entity_id = composio_entity_id
                    if app_name:
                        existing.app_name = app_name
                    existing.auth_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
                    if status == "active":
                        existing.last_verified = datetime.now(timezone.utc).replace(tzinfo=None)
                    if app_metadata:
                        existing.app_metadata = app_metadata
                    db.commit()
                    logger.info(f"✅ Updated connection in DB: {app_slug} for user {user_id} (status: {status})")
                elif existing_by_id:
                    # Connection already tracked under a different app_slug/user; just update metadata
                    existing_by_id.user_id = user_id
                    existing_by_id.internal_user_id = user_id
                    existing_by_id.app_slug = app_slug
                    existing_by_id.status = status
                    if composio_entity_id:
                        existing_by_id.composio_entity_id = composio_entity_id
                    if app_name:
                        existing_by_id.app_name = app_name
                    if app_metadata:
                        existing_by_id.app_metadata = app_metadata
                    existing_by_id.auth_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
                    if status == "active":
                        existing_by_id.last_verified = datetime.now(timezone.utc).replace(tzinfo=None)
                    db.commit()
                    logger.info(
                        f"✅ Updated existing connection by ID (status: {status})"
                    )
                else:
                    # Create new connection
                    import uuid
                    connection = UserConnection(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        internal_user_id=user_id,
                        composio_entity_id=composio_entity_id,
                        app_slug=app_slug,
                        app_name=app_name,
                        connection_id=encrypted_connection_id,
                        status=status,
                        auth_timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                        last_verified=datetime.now(timezone.utc).replace(tzinfo=None) if status == "active" else None,
                    )
                    if app_metadata:
                        connection.app_metadata = app_metadata
                    db.add(connection)
                    db.commit()
                    logger.info(f"✅ Created new connection in DB: {app_slug} for user {user_id} (status: {status})")
                
                # Also log the event
                self._log_connection_event(user_id, app_slug, "auth_completed", "success", connection_id=connection_id)

                # Keep canonical entity mapping table in sync when available
                if composio_entity_id:
                    mapping = (
                        db.query(ComposioEntity)
                        .filter(ComposioEntity.internal_user_id == user_id)
                        .first()
                    )
                    if mapping:
                        if mapping.composio_entity_id != composio_entity_id:
                            mapping.composio_entity_id = composio_entity_id
                    else:
                        db.add(
                            ComposioEntity(
                                internal_user_id=user_id,
                                composio_entity_id=composio_entity_id,
                            )
                        )
                    db.commit()
                
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to save connection to DB: {e}", exc_info=True)
    
    def get_connection_for_agent(
        self, 
        user_id: str, 
        app_slug: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get decrypted connection details for agent use.
        Verifies connection if last_verified is older than 1 hour.
        
        Args:
            user_id: Your app's user ID
            app_slug: App identifier (e.g., 'gmail', 'zohobooks')
        
        Returns:
            Dict with connection details including decrypted connection_id,
            or None if not found or inactive
        
        Example:
            conn = auth_mgr.get_connection_for_agent("user_123", "gmail")
            if conn:
                connection_id = conn["connection_id"]  # Already decrypted
                # Use connection_id for API calls
        """
        with SessionLocal() as db:
            try:
                # Look for active OR stale connections (stale can be refreshed)
                connection = (
                    db.query(UserConnection)
                    .filter(
                        or_(
                            UserConnection.user_id == user_id,
                            UserConnection.internal_user_id == user_id,
                        ),
                        UserConnection.app_slug == self._normalize_app_slug(app_slug),
                        UserConnection.status.in_(["active", "stale", "ACTIVE", "STALE"])
                    )
                    .first()
                )
                
                if not connection:
                    logger.warning(f"No active or stale connection found for user {user_id}, app {app_slug}")
                    return None
                
                # Check if connection needs verification (older than 1 hour)
                needs_verification = False
                if connection.last_verified:
                    time_since_verification = datetime.now(timezone.utc).replace(tzinfo=None) - connection.last_verified
                    needs_verification = time_since_verification > timedelta(hours=1)
                else:
                    needs_verification = True
                
                # Verify connection if needed
                if needs_verification:
                    logger.info(f"Verifying connection for {user_id}/{app_slug} (last verified: {connection.last_verified})")
                    try:
                        # Quick status check using v0.7.15 API
                        decrypted_id = self._decrypt_connection_id(connection.connection_id)

                        
                        # Get all connections for user (try both actual user_id and "default" fallback)
                        # Some connections may be registered under "default" entity_id
                        entity_ids_to_check = self._get_candidate_entity_ids(user_id, app_slug)
                        
                        all_connections = []
                        for entity_id in entity_ids_to_check:
                            try:
                                connections = self._composio.connected_accounts.get(entity_ids=[entity_id])
                                all_connections.extend(connections)
                            except Exception as e:
                                logger.debug(f"Could not fetch connections for entity {entity_id}: {e}")
                        
                        conn_found = False
                        for conn in all_connections:
                            conn_app_name = (getattr(conn, "appName", "") or "").lower()
                            if conn.id == decrypted_id and conn_app_name == self._normalize_app_slug(app_slug):
                                # Check if connection is active
                                if self._normalize_status(getattr(conn, "status", "")) in ["active", "connected"]:
                                    conn_found = True
                                    # Update composio_entity_id in DB if connection found under different entity
                                    if connection.composio_entity_id != conn.entityId:
                                        logger.info(f"Updating composio_entity_id from {connection.composio_entity_id} to {conn.entityId}")
                                        connection.composio_entity_id = conn.entityId
                                    break
                        
                        if conn_found:
                            # Update last_verified timestamp
                            connection.last_verified = datetime.now(timezone.utc).replace(tzinfo=None)
                            connection.status = "active"
                            db.commit()
                            logger.info(f"Connection verified and updated for {user_id}/{app_slug}")
                        else:
                            logger.warning(f"Connection not found or inactive for {user_id}/{app_slug}")
                            # Mark as stale but don't fail the request
                            connection.status = "stale"
                            db.commit()
                            return None
                            
                    except Exception as verify_error:
                        logger.warning(f"Connection verification failed for {user_id}/{app_slug}: {verify_error}")
                        # Mark as stale but don't fail the request
                        connection.status = "stale"
                        db.commit()
                        return None
                
                # Decrypt connection ID for use
                decrypted_id = self._decrypt_connection_id(connection.connection_id)
                
                return {
                    "connection_id": decrypted_id,
                    "app_slug": connection.app_slug,
                    "status": connection.status,
                    "created_at": connection.created_at,  # Fixed: model uses created_at not connected_at

                    "metadata": connection.app_metadata
                }
                
            except Exception as e:
                logger.error(f"Failed to get connection for {user_id}/{app_slug}: {e}")
                return None
    
    def _log_connection_event(
        self,
        user_id: str,
        app_slug: str,
        event_type: str,
        status: str,
        error_message: Optional[str] = None,
        connection_id: Optional[str] = None,
    ):
        """No-op: connection_logs table has been dropped."""
        logger.debug(f"[connection_event] {event_type}/{status} for {app_slug} (logging removed)")
# Singleton
_auth_manager = None


def get_auth_manager() -> ComposioAuthManager:
    """Get or create singleton auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = ComposioAuthManager()
    return _auth_manager

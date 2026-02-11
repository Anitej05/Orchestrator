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
from datetime import datetime, timedelta

from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import SessionLocal
from models import UserConnection, ConnectionLog
from cryptography.fernet import Fernet
import base64
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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
            logger.warning(f"Connection ID appears to be unencrypted (or wrong key). Using as-is.")
            return encrypted_id
    
    def _get_auth_config_id(self, app_slug: str) -> str:
        """
        Map app slug to auth config ID from environment.
        
        Auth configs must be created in Composio dashboard first:
        https://app.composio.dev → Integrations → Apps → Configure
        
        Args:
            app_slug: App identifier (e.g., 'gmail', 'zohobooks')
                Note: Use 'zohobooks' (normalized). 'zoho_books' is deprecated.
        
        Returns:
            Auth config ID (e.g., 'ac_gmail_123')
        
        Raises:
            ValueError: If auth config not found for app
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
            raise ValueError(
                f"No auth config found for '{app_slug}'. "
                f"Please create auth config in Composio dashboard and add "
                f"COMPOSIO_AUTH_CONFIG_{normalized_slug.upper()} to .env file. "
                f"Supported apps: {', '.join(auth_config_map.keys())}"
            )
        
        return auth_config_id

    def _get_integration_id(self, app_slug: str, auth_config_id: Optional[str] = None) -> str:
        """
        Resolve Composio integration ID (UUID) for an app.

        Tries Composio integrations API by app name. If not found, falls back
        to auth_config_id only if it is a valid UUID.
        """
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
        Initiate OAuth connection for a single app using official SDK.
        
        Args:
            user_id: Your app's user ID
            app_slug: Composio app slug (e.g., 'zohobooks', 'gmail', 'github')
            callback_url: Optional custom callback URL
        
        Returns:
            Dict with redirect_url and connection tracking info
        
        Prerequisites:
            - Auth configs must be created in Composio dashboard
            - Environment variables set: COMPOSIO_AUTH_CONFIG_{APP}
        
        Data flow:
            1. Backend calls this → gets redirect_url
            2. Frontend redirects user to redirect_url
            3. User authenticates on Composio Connect Link
            4. User redirected back to callback_url
            5. Backend polls check_connection_status() to verify
        """
        try:
            # Get auth config ID for this app (may be used as fallback)
            auth_config_id = self._get_auth_config_id(app_slug)
            integration_id = self._get_integration_id(app_slug, auth_config_id)
            
            logger.info(f"Starting auth for {app_slug} (user: {user_id})")
            logger.info(f"Using integration_id: {integration_id}")
            if callback_url:
                logger.info(f"Using callback URL: {callback_url}")
            
            # Use official SDK method: connected_accounts.initiate()
            # Parameters: integration_id, entity_id, redirect_url
            connection_request = self._composio.connected_accounts.initiate(
                integration_id=integration_id,
                entity_id=user_id,
                redirect_url=callback_url
            )
            
            # Extract connection_id (SDK returns connectedAccountId in v3)
            connection_id = None
            if hasattr(connection_request, "connectedAccountId"):
                connection_id = connection_request.connectedAccountId
            elif hasattr(connection_request, "connected_account_id"):
                connection_id = connection_request.connected_account_id
            elif hasattr(connection_request, "id"):
                connection_id = connection_request.id
            
            # Save connection to database with INITIATED status
            if connection_id:
                logger.info(f"💾 Saving INITIATED connection: {connection_id}")
                self._save_connection_to_db(
                    user_id=user_id,
                    app_slug=app_slug,
                    connection_id=connection_id,
                    status="INITIATED"
                )
            else:
                logger.warning(f"⚠️ No connection_id returned")
            
            redirect_url = None
            if hasattr(connection_request, "redirectUrl"):
                redirect_url = connection_request.redirectUrl
            elif hasattr(connection_request, "redirect_url"):
                redirect_url = connection_request.redirect_url

            return {
                "success": True,
                "redirect_url": redirect_url,
                "app_slug": app_slug,
                "user_id": user_id,
                "connection_id": connection_id,
                "poll_status_url": f"/api/integrations/status/{user_id}/{app_slug}",
            }
        
        except ValueError as e:
            # Auth config not found
            logger.error(f"Auth config error: {e}")
            return {
                "success": False,
                "error": str(e),
                "app_slug": app_slug,
            }
        except Exception as e:
            error_msg = self._format_composio_error(e)
            logger.error(f"Auth flow failed for {app_slug}: {error_msg}", exc_info=True)
            self._log_connection_event(user_id, app_slug, "initiated", "failed", error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "app_slug": app_slug,
                "troubleshooting": "Check that auth config is created in Composio dashboard and COMPOSIO_AUTH_CONFIG_{APP} is set in .env"
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
            # Use official SDK method: connected_accounts.get() with entity_ids parameter
            connections_response = self._composio.connected_accounts.get(
                entity_ids=[user_id],
                active=False  # Get all connections, not just active ones
            )
            
            # Handle both single connection and list of connections
            if isinstance(connections_response, list):
                connections = connections_response
            else:
                connections = [connections_response] if connections_response else []
            
            logger.info(f"📋 Checking {len(connections)} connections for user {user_id}")
            
            results = []
            results_by_slug = {}
            db_status_map = {}
            db_connection_map = {}

            # Load DB connections for the user (used when Composio status is delayed)
            try:
                with SessionLocal() as db:
                    db_connections = db.query(UserConnection).filter(
                        UserConnection.user_id == user_id
                    ).all()
                    for conn in db_connections:
                        db_status_map[conn.app_slug] = conn.status
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
                status = (status or getattr(conn, "status", "") or "").lower()
                connected_account_id = connected_account_id or getattr(conn, "id", None)

                slug = (app_unique_id or app_name or "").lower()
                is_connected = status == "active"
                
                db_status = db_status_map.get(slug)
                db_connection_id = db_connection_map.get(slug)

                connection_info = {
                    "name": getattr(conn, "appName", None) or slug,
                    "slug": slug,
                    "is_connected": is_connected,
                    "connected_account_id": connected_account_id,
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
                if t["is_connected"] or (t.get("db_status") == "active")
            ]
            pending = [
                t for t in results
                if t.get("db_status") == "INITIATED"
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
            # Get user's connections using official SDK
            connections_response = self._composio.connected_accounts.list(
                user_ids=[user_id]
            )
            
            for conn in connections_response.items:
                integration_id = conn.integration_id if hasattr(conn, 'integration_id') else None
                slug = (integration_id or "").lower()
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
            
            # Use official SDK refresh method
            response = self._composio.connected_accounts.refresh(
                connected_account_id=connection_id
            )
            
            # Update DB with refresh timestamp
            db = SessionLocal()
            try:
                encrypted_conn_id = self._encrypt_connection_id(connection_id)
                conn = db.query(UserConnection).filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug
                ).first()
                if conn:
                    conn.auth_timestamp = datetime.utcnow()
                    db.commit()
            finally:
                db.close()
            
            logger.info(f"✓ Successfully refreshed {app_slug} connection")
            self._log_connection_event(user_id, app_slug, "refreshed", "success")
            
            return {
                "success": True,
                "status": response.status if hasattr(response, 'status') else "refreshed",
                "refreshed_at": datetime.utcnow().isoformat(),
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
            
            # Use official SDK disable method
            self._composio.connected_accounts.disable(
                connected_account_id=connection_id
            )
            
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
            
            # Use official SDK enable method
            self._composio.connected_accounts.enable(
                connected_account_id=connection_id
            )
            
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
        app_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save or update connection in user_connections table.
        
        Connection ID is encrypted before storage for security.
        """
        # Normalize app_slug to lowercase to ensure consistency with Composio's format
        app_slug = app_slug.lower()
        
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
                    existing.auth_timestamp = datetime.utcnow()
                    if app_metadata:
                        existing.app_metadata = app_metadata
                    db.commit()
                    logger.info(f"✅ Updated connection in DB: {app_slug} for user {user_id} (status: {status})")
                elif existing_by_id:
                    # Connection already tracked under a different app_slug/user; just update metadata
                    existing_by_id.status = status
                    if app_metadata:
                        existing_by_id.app_metadata = app_metadata
                    existing_by_id.auth_timestamp = datetime.utcnow()
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
                        app_slug=app_slug,
                        connection_id=encrypted_connection_id,
                        status=status,
                        auth_timestamp=datetime.utcnow()
                    )
                    if app_metadata:
                        connection.app_metadata = app_metadata
                    db.add(connection)
                    db.commit()
                    logger.info(f"✅ Created new connection in DB: {app_slug} for user {user_id} (status: {status})")
                
                # Also log the event
                self._log_connection_event(user_id, app_slug, "auth_completed", "success", connection_id=connection_id)
                
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
                        UserConnection.user_id == user_id,
                        UserConnection.app_slug == app_slug,
                        UserConnection.status.in_(["active", "stale"])
                    )
                    .first()
                )
                
                if not connection:
                    logger.warning(f"No active or stale connection found for user {user_id}, app {app_slug}")
                    return None
                
                # Check if connection needs verification (older than 1 hour)
                needs_verification = False
                if connection.last_verified:
                    time_since_verification = datetime.utcnow() - connection.last_verified
                    needs_verification = time_since_verification > timedelta(hours=1)
                else:
                    needs_verification = True
                
                # Verify connection if needed
                if needs_verification:
                    logger.info(f"Verifying connection for {user_id}/{app_slug} (last verified: {connection.last_verified})")
                    try:
                        # Quick status check
                        decrypted_id = self._decrypt_connection_id(connection.connection_id)
                        conn_check = self._composio.connected_accounts.get(connected_account_id=decrypted_id)
                        
                        # Update last_verified timestamp
                        connection.last_verified = datetime.utcnow()
                        db.commit()
                        logger.info(f"Connection verified and updated for {user_id}/{app_slug}")
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
                    "connected_at": connection.connected_at,
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
        """Log connection events to database for audit trail."""
        try:
            with SessionLocal() as db:
                log = ConnectionLog(
                    user_id=user_id,
                    app_slug=app_slug,
                    connection_id=connection_id,
                    event_type=event_type,
                    status=status,
                    error_message=error_message,
                )
                db.add(log)
                db.commit()
                logger.debug(f"✓ Logged event: {event_type} for {app_slug}")
        except Exception as e:
            logger.error(f"Failed to log connection event: {e}")
# Singleton
_auth_manager = None


def get_auth_manager() -> ComposioAuthManager:
    """Get or create singleton auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = ComposioAuthManager()
    return _auth_manager

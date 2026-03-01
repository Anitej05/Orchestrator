"""
Integrations Router

Manages OAuth authentication with external services via Composio.
"""

from typing import List, Optional
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.integrations.composio_auth import get_auth_manager

# Set up logging for integrations
logger = logging.getLogger("integrations")
logger.setLevel(logging.DEBUG)

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


# Request/Response Models
class StartAuthRequest(BaseModel):
    app_slug: str  # 'zohobooks', 'gmail', 'github', etc.
    callback_url: Optional[str] = None


class VerifyConnectionsRequest(BaseModel):
    required_apps: List[str]


class AuthResponse(BaseModel):
    success: bool
    redirect_url: Optional[str] = None
    error: Optional[str] = None
    poll_status_url: Optional[str] = None


class ToolkitStatus(BaseModel):
    name: str
    slug: str
    is_connected: bool
    connected_account_id: Optional[str] = None
    db_status: Optional[str] = None
    db_connection_id: Optional[str] = None


class ConnectionStatusResponse(BaseModel):
    success: bool
    user_id: str
    connected_apps: List[str]
    pending_apps: List[str]
    all_toolkits: Optional[List[ToolkitStatus]] = None


# Endpoints
@router.post("/auth/start/{user_id}", response_model=AuthResponse)
async def start_auth(user_id: str, request: StartAuthRequest):
    """
    Start OAuth flow for an app.
    
    Response sends redirect_url to frontend.
    Frontend should:
    1. Redirect user to redirect_url
    2. After user authenticates, poll /integrations/status/{user_id}/{app_slug}
    
    Example:
    ```
    POST /api/integrations/auth/start/user_123
    {
        "app_slug": "zohobooks",
        "callback_url": "http://localhost:3000/auth/callback"
    }
    
    Response:
    {
        "success": true,
        "redirect_url": "https://connect.composio.dev/link/ln_abc123",
        "poll_status_url": "/api/integrations/status/user_123/zohobooks"
    }
    ```
    """
    logger.info(f"🔵 START AUTH - user_id={user_id}, app_slug={request.app_slug}")
    try:
        manager = get_auth_manager()
        logger.debug("✓ Auth manager loaded")
        
        result = manager.start_auth_flow(user_id, request.app_slug, request.callback_url)
        logger.info(f"🟢 Auth flow started - success={result['success']}")
        
        if not result["success"]:
            logger.error(f"❌ Auth flow failed: {result['error']}")
            # Build error response with troubleshooting hints
            error_detail = {
                "error": result["error"],
            }
            if result.get("troubleshooting"):
                error_detail["troubleshooting"] = result["troubleshooting"]
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"✅ Returning redirect URL: {result.get('redirect_url')[:50] if result.get('redirect_url') else 'None'}...")
        return {
            "success": True,
            "redirect_url": result.get("redirect_url"),
            "connection_id": result.get("connection_id"),
            "poll_status_url": result.get("poll_status_url"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Exception in start_auth: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{user_id}", response_model=ConnectionStatusResponse)
async def check_all_connections(user_id: str):
    """
    Check all connection statuses for a user.
    
    Returns list of connected and pending apps.
    """
    logger.info(f"🔵 CHECK ALL CONNECTIONS - user_id={user_id}")
    try:
        manager = get_auth_manager()
        result = manager.check_connection_status(user_id)
        
        if not result["success"]:
            logger.error(f"❌ Status check failed: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        logger.info(f"✅ Connected apps: {result['connected_apps']}, Pending: {result['pending_apps']}")
        return {
            "success": True,
            "user_id": user_id,
            "connected_apps": result["connected_apps"],
            "pending_apps": result["pending_apps"],
            "all_toolkits": result.get("all_toolkits", []),
        }
    except Exception as e:
        logger.error(f"❌ Exception in check_all_connections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{user_id}/{app_slug}", response_model=ConnectionStatusResponse)
async def check_auth_status(user_id: str, app_slug: str):
    """
    Poll connection status for a specific app after redirect.
    
    Use this endpoint after user returns from OAuth redirect.
    """
    logger.info(f"🔵 CHECK STATUS - user_id={user_id}, app_slug={app_slug}")
    try:
        manager = get_auth_manager()
        result = manager.check_connection_status(user_id, app_slug)
        
        if not result["success"]:
            logger.error(f"❌ Status check failed: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        toolkits = result.get("all_toolkits", [])
        connected = [t["slug"] for t in toolkits if t["is_connected"]]
        pending = [t["slug"] for t in toolkits if not t["is_connected"]]
        
        logger.info(f"✅ App status - connected: {connected}, pending: {pending}")
        return {
            "success": True,
            "user_id": user_id,
            "connected_apps": connected,
            "pending_apps": pending,
            "all_toolkits": toolkits,
        }
    except Exception as e:
        logger.error(f"❌ Exception in check_auth_status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify/{user_id}")
async def verify_connections(user_id: str, request: VerifyConnectionsRequest):
    """
    Verify user has all required app connections.
    
    Returns missing apps with auth URLs if needed.
    
    Example:
    ```
    POST /api/integrations/verify/user_123
    {
        "required_apps": ["zohobooks", "gmail"]
    }
    
    Response:
    {
        "success": true,
        "all_connected": true,
        "connected_apps": ["zohobooks", "gmail"],
        "missing_apps": [],
        "auth_urls": {}
    }
    ```
    """
    logger.info(f"🔵 VERIFY CONNECTIONS - user_id={user_id}, required={request.required_apps}")
    try:
        manager = get_auth_manager()
        result = manager.verify_required_connections(user_id, request.required_apps)
        
        logger.info(f"✅ Verification result - all_connected={result.get('all_connected')}")
        return result
    except Exception as e:
        logger.error(f"❌ Exception in verify_connections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/{user_id}")
async def sync_connections(user_id: str):
    """
    Manually sync connection status from Composio to local database.
    
    This is useful after OAuth completion to immediately update the
    connection status from INITIATED to active without waiting for polling.
    
    Example:
    ```
    POST /api/integrations/sync/user_123
    
    Response:
    {
        "success": true,
        "user_id": "user_123",
        "synced_apps": ["zohobooks", "gmail"],
        "connected_apps": ["zohobooks", "gmail"],
        "pending_apps": []
    }
    ```
    """
    logger.info(f"🔵 SYNC CONNECTIONS - user_id={user_id}")
    try:
        manager = get_auth_manager()
        
        # Force a fresh check from Composio which will sync to database
        result = manager.check_connection_status(user_id)
        
        if not result["success"]:
            logger.error(f"❌ Sync failed: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        synced_apps = result.get("connected_apps", [])
        logger.info(f"✅ Synced {len(synced_apps)} connections: {synced_apps}")
        
        return {
            "success": True,
            "user_id": user_id,
            "synced_apps": synced_apps,
            "connected_apps": result.get("connected_apps", []),
            "pending_apps": result.get("pending_apps", []),
            "all_toolkits": result.get("all_toolkits", []),
        }
    except Exception as e:
        logger.error(f"❌ Exception in sync_connections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/disconnect/{user_id}/{app_slug}")
async def disconnect_app(user_id: str, app_slug: str):
    """
    Disconnect user from a specific app.
    
    Example:
    ```
    DELETE /api/integrations/disconnect/user_123/zohobooks
    
    Response:
    {
        "success": true,
        "message": "Disconnected from zohobooks"
    }
    ```
    """
    logger.info(f"🔵 DISCONNECT - user_id={user_id}, app_slug={app_slug}")
    try:
        manager = get_auth_manager()
        result = manager.disconnect_app(user_id, app_slug)
        
        if not result["success"]:
            logger.error(f"❌ Disconnect failed: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        logger.info(f"✅ Successfully disconnected from {app_slug}")
        return result
    except Exception as e:
        logger.error(f"❌ Exception in disconnect_app: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh/{user_id}/{app_slug}")
async def refresh_connection(user_id: str, app_slug: str):
    """
    Refresh OAuth token for a connection.
    
    Use when:
    - Connection becomes stale
    - API calls return 401 errors
    - Proactive token refresh
    
    Example:
    ```
    POST /api/integrations/refresh/user_123/gmail
    
    Response:
    {
        "success": true,
        "message": "Connection refreshed successfully",
        "refreshed_at": "2026-02-07T12:34:56"
    }
    ```
    """
    logger.info(f"🔵 REFRESH CONNECTION - user_id={user_id}, app_slug={app_slug}")
    try:
        manager = get_auth_manager()
        result = manager.refresh_connection(user_id, app_slug)
        
        if not result["success"]:
            logger.error(f"❌ Refresh failed: {result.get('error')}")
            error_response = {
                "success": False,
                "error": result.get("error"),
            }
            if result.get("troubleshooting"):
                error_response["troubleshooting"] = result["troubleshooting"]
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        logger.info(f"✅ Successfully refreshed connection for {app_slug}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Exception in refresh_connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable/{user_id}/{app_slug}")
async def disable_connection(user_id: str, app_slug: str):
    """
    Temporarily disable a connection without deleting.
    
    Useful for:
    - User wants to pause integration
    - Rate limit management
    - Testing without full disconnect
    
    Example:
    ```
    POST /api/integrations/disable/user_123/gmail
    
    Response:
    {
        "success": true,
        "message": "Connection disabled",
        "status": "disabled"
    }
    ```
    """
    logger.info(f"🔵 DISABLE CONNECTION - user_id={user_id}, app_slug={app_slug}")
    try:
        manager = get_auth_manager()
        result = manager.disable_connection(user_id, app_slug)
        
        if not result["success"]:
            logger.error(f"❌ Disable failed: {result.get('error')}")
            error_response = {
                "success": False,
                "error": result.get("error"),
            }
            if result.get("troubleshooting"):
                error_response["troubleshooting"] = result["troubleshooting"]
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        logger.info(f"✅ Successfully disabled connection for {app_slug}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Exception in disable_connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable/{user_id}/{app_slug}")
async def enable_connection(user_id: str, app_slug: str):
    """
    Re-enable a previously disabled connection.
    
    Example:
    ```
    POST /api/integrations/enable/user_123/gmail
    
    Response:
    {
        "success": true,
        "message": "Connection enabled",
        "status": "active"
    }
    ```
    """
    logger.info(f"🔵 ENABLE CONNECTION - user_id={user_id}, app_slug={app_slug}")
    try:
        manager = get_auth_manager()
        result = manager.enable_connection(user_id, app_slug)
        
        if not result["success"]:
            logger.error(f"❌ Enable failed: {result.get('error')}")
            error_response = {
                "success": False,
                "error": result.get("error"),
            }
            if result.get("troubleshooting"):
                error_response["troubleshooting"] = result["troubleshooting"]
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        logger.info(f"✅ Successfully enabled connection for {app_slug}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Exception in enable_connection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

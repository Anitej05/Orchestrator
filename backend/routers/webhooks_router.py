# routers/webhooks_router.py
"""
Webhooks Router

Handles inbound webhook events from Composio.

Composio Dashboard setup:
  1. Settings → Webhooks
  2. Set URL: https://your-api.com/webhooks/composio
  3. Subscribe to: connection.completed | connection.expired | connection.error
  4. Copy the webhook secret → env var: COMPOSIO_WEBHOOK_SECRET

Events handled:
  connection.completed  → mark UserConnection ACTIVE, invalidate session cache
  connection.expired    → mark UserConnection EXPIRED
  connection.error      → mark UserConnection REVOKED / log error
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger("webhooks")

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ---------------------------------------------------------------------------
# Pydantic models for the Composio webhook payload
# ---------------------------------------------------------------------------

class ComposioWebhookPayload(BaseModel):
    event: str                          # e.g. "connection.completed"
    user_id: Optional[str] = None       # local user_id (entity_id Composio knows)
    app_slug: Optional[str] = None      # e.g. "gmail"
    connection_id: Optional[str] = None # Composio connection ID (may be encrypted already)
    error: Optional[str] = None
    # Allow arbitrary extra fields from Composio
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Signature verification helper
# ---------------------------------------------------------------------------

def _verify_composio_signature(
    raw_body: bytes,
    signature_header: Optional[str],
    secret: Optional[str],
) -> bool:
    """
    Verify HMAC-SHA256 signature sent by Composio.
    Returns True when:
    - No secret is configured (dev mode / webhook not yet signed)
    - Signature matches
    Returns False otherwise.
    """
    if not secret:
        logger.debug("[Webhook] COMPOSIO_WEBHOOK_SECRET not set — skipping signature check")
        return True

    if not signature_header:
        logger.warning("[Webhook] Signature header missing")
        return False

    # Composio sends: "sha256=<hex_digest>"
    expected_prefix = "sha256="
    if not signature_header.startswith(expected_prefix):
        return False

    provided_digest = signature_header[len(expected_prefix):]
    mac = hmac.new(secret.encode(), raw_body, hashlib.sha256)
    expected_digest = mac.hexdigest()

    return hmac.compare_digest(provided_digest, expected_digest)


# ---------------------------------------------------------------------------
# Main webhook endpoint
# ---------------------------------------------------------------------------

@router.post("/composio")
async def handle_composio_webhook(
    request: Request,
    x_composio_signature: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Receive and process Composio webhook events.

    Composio calls this URL after OAuth events (connection completed / expired / error).
    The endpoint:
    1. Optionally verifies HMAC-SHA256 signature.
    2. Dispatches to the appropriate handler based on event type.
    3. Returns 200 quickly (Composio retries on non-2xx).
    """
    raw_body = await request.body()

    # Verify signature if secret is configured
    secret = os.getenv("COMPOSIO_WEBHOOK_SECRET")
    if not _verify_composio_signature(raw_body, x_composio_signature, secret):
        logger.warning("[Webhook] Invalid Composio signature — rejecting request")
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = body.get("event", "")
    user_id = body.get("user_id") or body.get("entity_id")
    app_slug = (body.get("app_slug") or body.get("app", "")).lower()
    connection_id = body.get("connection_id")

    logger.info(
        f"[Webhook] 📨 Received event={event!r} user_id={user_id!r} app={app_slug!r}"
    )

    if event == "connection.completed":
        return await _handle_connection_completed(user_id, app_slug, connection_id, body)

    elif event == "connection.expired":
        return await _handle_connection_status_change(user_id, app_slug, "EXPIRED", "connection.expired")

    elif event == "connection.error":
        error_msg = body.get("error", "")
        logger.error(f"[Webhook] Connection error for {user_id}/{app_slug}: {error_msg}")
        return await _handle_connection_status_change(user_id, app_slug, "REVOKED", "connection.error", error_msg)

    else:
        logger.info(f"[Webhook] Unhandled event type: {event!r}")
        return {"status": "received", "event": event, "message": "Event noted but not handled"}


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

async def _handle_connection_completed(
    user_id: Optional[str],
    app_slug: str,
    connection_id: Optional[str],
    raw_body: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mark the user's connection as ACTIVE in our DB.
    Called when Composio reports a successful OAuth completion.
    """
    if not user_id or not app_slug:
        logger.warning("[Webhook] connection.completed missing user_id or app_slug — ignoring")
        return {"status": "ignored", "reason": "missing user_id or app_slug"}

    try:
        from services.integrations.composio_auth import get_auth_manager

        auth_manager = get_auth_manager()
        if connection_id:
            auth_manager._save_connection_to_db(
                user_id=user_id,
                app_slug=app_slug,
                connection_id=connection_id,
                status="active",
                composio_entity_id=user_id,
                app_name=app_slug,
                app_metadata={"source": "webhook", "event": "connection.completed"},
            )
        else:
            auth_manager._log_connection_event(
                user_id=user_id,
                app_slug=app_slug,
                event_type="auth_completed",
                status="failed",
                error_message="Missing connection_id in webhook payload",
            )

        logger.info(
            f"[Webhook] ✅ Connection ACTIVE: user={user_id} app={app_slug}"
        )

        # Invalidate tool cache for this user so fresh tools are fetched
        _invalidate_tool_cache(user_id)

        return {
            "status": "ok",
            "message": f"Connection {app_slug} marked ACTIVE for user {user_id}",
        }

    except Exception as e:
        logger.error(f"[Webhook] connection.completed handler error: {e}", exc_info=True)
        # Return 200 so Composio doesn't retry — we'll reconcile via polling
        return {"status": "error", "message": str(e)}


async def _handle_connection_status_change(
    user_id: Optional[str],
    app_slug: str,
    new_status: str,
    event_type: str,
    error_msg: str = "",
) -> Dict[str, Any]:
    """
    Update the status column on UserConnection (e.g. EXPIRED or REVOKED).
    """
    if not user_id or not app_slug:
        return {"status": "ignored", "reason": "missing user_id or app_slug"}

    new_status = (new_status or "").lower()

    try:
        from database import SessionLocal
        from models import UserConnection

        with SessionLocal() as db:
            conn = (
                db.query(UserConnection)
                .filter(
                    UserConnection.user_id == user_id,
                    UserConnection.app_slug == app_slug,
                )
                .first()
            )

            if conn:
                conn.status = new_status
                conn.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            db.commit()

        logger.info(
            f"[Webhook] Connection {new_status}: user={user_id} app={app_slug}"
        )
        return {"status": "ok", "message": f"Connection marked {new_status}"}

    except Exception as e:
        logger.error(f"[Webhook] status-change handler error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invalidate_tool_cache(user_id: str) -> None:
    """
    Invalidate the integrations agent tool cache for this user.
    Best-effort — no crash if cache is unavailable.
    """
    try:
        from agents.integrations_agent.agent import tool_cache
        tool_cache.invalidate_user(user_id)
        logger.debug(f"[Webhook] Tool cache invalidated for {user_id}")
    except Exception as e:
        logger.debug(f"[Webhook] Could not invalidate tool cache: {e}")

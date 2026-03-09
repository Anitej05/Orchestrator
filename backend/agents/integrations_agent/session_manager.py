# agents/integrations_agent/session_manager.py
"""
Composio Session Manager

Per-user, per-app session management with context persistence.

When a user returns to a conversation the agent can greet them:
  "I see you were working with Google Sheets last time. 
   Want to continue from where you left off?"

Table: integrations_sessions  (created by migration 07_add_integrations_sessions_*)
"""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("integrations_agent.session_manager")


class ComposioSessionManager:
    """
    Manages per-user, per-app sessions for the Integrations Agent.

    Responsibilities:
    - Get or create a Composio session for (user_id, app_slug)
    - Persist execution context so the agent can resume intelligently
    - Expose last_context for continuity prompts
    """

    def get_or_create_session(self, user_id: str, app_slug: str) -> Dict[str, Any]:
        """
        Return an existing session or create a new one.

        Args:
            user_id:  Local system user identifier
            app_slug: Composio app slug, e.g. "gmail", "googlesheets"

        Returns:
            Session dict {user_id, app_slug, session_id, last_context, is_new}
        """
        try:
            from database import SessionLocal
            from models import IntegrationsSession

            app_slug = app_slug.lower()

            with SessionLocal() as db:
                row = (
                    db.query(IntegrationsSession)
                    .filter(
                        IntegrationsSession.user_id == user_id,
                        IntegrationsSession.app_slug == app_slug,
                    )
                    .first()
                )

                if row:
                    logger.info(
                        f"[SessionManager] Resumed session {row.session_id} "
                        f"for {user_id}/{app_slug}"
                    )
                    return {
                        "user_id": user_id,
                        "app_slug": app_slug,
                        "session_id": row.session_id,
                        "last_context": row.last_context or {},
                        "is_new": False,
                    }

                # Create a new session
                new_session_id = str(uuid.uuid4())
                row = IntegrationsSession(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    app_slug=app_slug,
                    session_id=new_session_id,
                    last_context={},
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                    last_used=datetime.now(timezone.utc).replace(tzinfo=None),
                )
                db.add(row)
                db.commit()

                logger.info(
                    f"[SessionManager] Created new session {new_session_id} "
                    f"for {user_id}/{app_slug}"
                )
                return {
                    "user_id": user_id,
                    "app_slug": app_slug,
                    "session_id": new_session_id,
                    "last_context": {},
                    "is_new": True,
                }

        except Exception as e:
            logger.warning(
                f"[SessionManager] DB unavailable, using in-memory session: {e}"
            )
            # Fallback: ephemeral in-memory session (no persistence)
            return {
                "user_id": user_id,
                "app_slug": app_slug,
                "session_id": str(uuid.uuid4()),
                "last_context": {},
                "is_new": True,
            }

    def save_context(self, user_id: str, app_slug: str, context: Dict[str, Any]) -> bool:
        """
        Persist execution context for the next conversation.

        Args:
            user_id:  Local user identifier
            app_slug: Composio app slug
            context:  Dict of context to preserve (e.g. file_id, folder_id)

        Returns:
            True on success, False if DB unavailable
        """
        if not context:
            return True

        try:
            from database import SessionLocal
            from models import IntegrationsSession

            app_slug = app_slug.lower()

            with SessionLocal() as db:
                row = (
                    db.query(IntegrationsSession)
                    .filter(
                        IntegrationsSession.user_id == user_id,
                        IntegrationsSession.app_slug == app_slug,
                    )
                    .first()
                )

                if row:
                    row.last_context = context
                    row.last_used = datetime.now(timezone.utc).replace(tzinfo=None)
                else:
                    row = IntegrationsSession(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        app_slug=app_slug,
                        session_id=str(uuid.uuid4()),
                        last_context=context,
                        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        last_used=datetime.now(timezone.utc).replace(tzinfo=None),
                    )
                    db.add(row)

                db.commit()
                logger.debug(
                    f"[SessionManager] Saved context for {user_id}/{app_slug}: "
                    f"{list(context.keys())}"
                )
                return True

        except Exception as e:
            logger.warning(f"[SessionManager] Failed to save context: {e}")
            return False

    def get_context(self, user_id: str, app_slug: str) -> Dict[str, Any]:
        """
        Retrieve preserved context for the next conversation turn.

        Args:
            user_id:  Local user identifier
            app_slug: Composio app slug

        Returns:
            Context dict (may be empty if no previous session exists)
        """
        try:
            from database import SessionLocal
            from models import IntegrationsSession

            app_slug = app_slug.lower()

            with SessionLocal() as db:
                row = (
                    db.query(IntegrationsSession)
                    .filter(
                        IntegrationsSession.user_id == user_id,
                        IntegrationsSession.app_slug == app_slug,
                    )
                    .first()
                )

                if row and row.last_context:
                    logger.debug(
                        f"[SessionManager] Retrieved context for {user_id}/{app_slug}"
                    )
                    return row.last_context

        except Exception as e:
            logger.warning(f"[SessionManager] Failed to retrieve context: {e}")

        return {}

    def delete_session(self, user_id: str, app_slug: str) -> bool:
        """
        Delete a session (e.g. when user disconnects an app).

        Args:
            user_id:  Local user identifier
            app_slug: Composio app slug

        Returns:
            True on success
        """
        try:
            from database import SessionLocal
            from models import IntegrationsSession

            app_slug = app_slug.lower()

            with SessionLocal() as db:
                rows = db.query(IntegrationsSession).filter(
                    IntegrationsSession.user_id == user_id,
                    IntegrationsSession.app_slug == app_slug,
                ).all()

                for row in rows:
                    db.delete(row)
                db.commit()

                logger.info(
                    f"[SessionManager] Deleted session for {user_id}/{app_slug}"
                )
                return True

        except Exception as e:
            logger.warning(f"[SessionManager] Failed to delete session: {e}")
            return False


# Module-level singleton
_session_manager: Optional[ComposioSessionManager] = None


def get_session_manager() -> ComposioSessionManager:
    """Return the module-level ComposioSessionManager singleton."""
    global _session_manager
    if _session_manager is None:
        _session_manager = ComposioSessionManager()
    return _session_manager

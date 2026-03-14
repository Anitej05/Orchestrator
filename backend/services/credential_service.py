"""
Credential Manager
Centralized, encrypted, database-backed credential storage.

Provides a single entry point for all credential management across agents,
tools, and system-level keys.  Falls back to os.getenv() when the DB does
not have a credential — so the system works out of the box with a .env file
and gradually migrates to DB storage as users save credentials via the UI.
"""

import os
import time
import uuid

import logging
from typing import Dict, Optional

from sqlalchemy.orm import Session

from backend.utils.encryption import encrypt, decrypt

logger = logging.getLogger("uvicorn.error")

# Cache TTL in seconds — avoid repeated DB lookups within a request burst
_CACHE_TTL = 60


class CredentialManager:
    """
    Singleton credential manager.

    Usage::

        from backend.services.credential_service import credential_manager

        # Get a single credential (agent scope)
        key = credential_manager.get("agent", "mail_agent", "COMPOSIO_API_KEY")

        # Get all credentials for a tool
        creds = credential_manager.get_all("tool", "web_search")

        # Save credentials from the frontend
        credential_manager.save("agent", "mail_agent", {"COMPOSIO_API_KEY": "sk-..."})
    """

    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value_dict, timestamp)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_db(self) -> Session:
        """Get a new DB session.  Caller must close it."""
        from database import SessionLocal
        return SessionLocal()

    def _cache_key(self, scope: str, scope_id: str, user_id: str) -> str:
        return f"{scope}:{scope_id}:{user_id}"

    def _get_from_cache(self, key: str) -> Optional[Dict[str, str]]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[1]) < _CACHE_TTL:
            return entry[0]
        return None

    def _set_cache(self, key: str, value: Dict[str, str]):
        self._cache[key] = (value, time.time())

    def _invalidate_cache(self, key: str):
        self._cache.pop(key, None)

    def _decrypt_credentials(self, encrypted_creds: dict) -> Dict[str, str]:
        """Decrypt an encrypted_credentials JSON dict."""
        result = {}
        if not encrypted_creds:
            return result
        for k, enc_val in encrypted_creds.items():
            try:
                result[k] = decrypt(enc_val)
            except Exception as e:
                logger.error(f"Failed to decrypt credential '{k}': {e}")
        return result

    # ------------------------------------------------------------------
    # Public API — Generic scope-based access
    # ------------------------------------------------------------------

    def get(
        self,
        scope: str,
        scope_id: str,
        key: str,
        user_id: str = "system",
        env_fallback: bool = True,
    ) -> Optional[str]:
        """
        Get a single credential value.

        Lookup order:
        1. In-memory cache
        2. Generic ``Credential`` table (new)
        3. Legacy ``AgentCredential`` table (if scope == "agent")
        4. ``os.getenv(key)`` fallback (if env_fallback=True)

        Returns None if not found anywhere.
        """
        all_creds = self.get_all(scope, scope_id, user_id)
        value = all_creds.get(key)
        if value:
            return value

        # .env fallback
        if env_fallback:
            env_val = os.getenv(key)
            if env_val:
                return env_val

        return None

    def get_all(
        self,
        scope: str,
        scope_id: str,
        user_id: str = "system",
    ) -> Dict[str, str]:
        """
        Get all credentials for a given scope/scope_id/user_id as a
        dict of ``{credential_name: plaintext_value}``.
        """
        ck = self._cache_key(scope, scope_id, user_id)
        cached = self._get_from_cache(ck)
        if cached is not None:
            return cached

        db = self._get_db()
        try:
            creds = self._load_from_generic_table(db, scope, scope_id, user_id)
            self._set_cache(ck, creds)
            return creds
        except Exception as e:
            logger.warning(f"DB credential lookup failed ({scope}/{scope_id}): {e}")
            return {}
        finally:
            db.close()

    def save(
        self,
        scope: str,
        scope_id: str,
        credentials: Dict[str, str],
        user_id: str = "system",
    ) -> bool:
        """
        Save (upsert) encrypted credentials.
        """
        from models import Credential

        db = self._get_db()
        try:
            encrypted = {}
            for k, v in credentials.items():
                if v:  # Only encrypt non-empty
                    encrypted[k] = encrypt(v)

            existing = (
                db.query(Credential)
                .filter_by(scope=scope, scope_id=scope_id, user_id=user_id)
                .first()
            )

            if existing:
                # Merge: update existing keys, add new ones
                merged = dict(existing.encrypted_credentials or {})
                merged.update(encrypted)
                existing.encrypted_credentials = merged
                existing.is_active = True
                from datetime import datetime
                existing.updated_at = datetime.utcnow()
            else:
                new_cred = Credential(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    scope=scope,
                    scope_id=scope_id,
                    encrypted_credentials=encrypted,
                    is_active=True,
                )
                db.add(new_cred)

            db.commit()
            self._invalidate_cache(self._cache_key(scope, scope_id, user_id))
            logger.info(f"Saved credentials for {scope}/{scope_id} (user={user_id})")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save credentials: {e}", exc_info=True)
            return False
        finally:
            db.close()

    def delete(
        self,
        scope: str,
        scope_id: str,
        user_id: str = "system",
    ) -> bool:
        """Delete credentials for a given scope."""
        from models import Credential

        db = self._get_db()
        try:
            deleted = (
                db.query(Credential)
                .filter_by(scope=scope, scope_id=scope_id, user_id=user_id)
                .delete()
            )
            db.commit()
            self._invalidate_cache(self._cache_key(scope, scope_id, user_id))
            if deleted:
                logger.info(f"Deleted credentials for {scope}/{scope_id}")
                return True
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete credentials: {e}", exc_info=True)
            return False
        finally:
            db.close()

    def has_valid(
        self,
        scope: str,
        scope_id: str,
        user_id: str = "system",
    ) -> bool:
        """Check if valid (non-empty, active) credentials exist."""
        creds = self.get_all(scope, scope_id, user_id)
        return bool(creds)

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_from_generic_table(
        self, db: Session, scope: str, scope_id: str, user_id: str
    ) -> Dict[str, str]:
        """Load from the new generic ``credentials`` table."""
        from models import Credential

        row = (
            db.query(Credential)
            .filter_by(scope=scope, scope_id=scope_id, user_id=user_id, is_active=True)
            .first()
        )
        if not row:
            return {}
        return self._decrypt_credentials(row.encrypted_credentials)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
credential_manager = CredentialManager()


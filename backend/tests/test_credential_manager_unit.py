"""
Unit tests for CredentialManager.
Tests cover: get/save/delete, env fallback, caching, scope system, and legacy compat.
"""

import os
import sys
import uuid
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure backend is on the path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ---------------------------------------------------------------------------
# Use SQLite in-memory for tests — must set before importing database/models
# ---------------------------------------------------------------------------
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ENCRYPTION_KEY"] = "test-key-DoNotUseInProd-32chars00="

from database import Base, engine, SessionLocal
from models import Credential, AgentCredential
from backend.utils.encryption import encrypt, decrypt


@pytest.fixture(autouse=True)
def setup_database():
    """Create all tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def manager():
    """Fresh CredentialManager instance with empty cache."""
    from backend.services.credential_service import CredentialManager
    mgr = CredentialManager()
    # Override _get_db to use our test session
    mgr._get_db = lambda: SessionLocal()
    return mgr


# ---------------------------------------------------------------------------
# Core CRUD
# ---------------------------------------------------------------------------

class TestSaveAndGet:
    def test_save_and_get_single(self, manager):
        """Save a credential and retrieve it."""
        manager.save("agent", "gmail_agent", {"COMPOSIO_API_KEY": "sk-123"}, "user1")
        val = manager.get("agent", "gmail_agent", "COMPOSIO_API_KEY", "user1", env_fallback=False)
        assert val == "sk-123"

    def test_save_and_get_all(self, manager):
        """Save multiple credentials and retrieve all."""
        manager.save("tool", "web_search", {"SERP_KEY": "key1", "CUSTOM": "key2"}, "user1")
        all_creds = manager.get_all("tool", "web_search", "user1")
        assert all_creds == {"SERP_KEY": "key1", "CUSTOM": "key2"}

    def test_get_missing_returns_none(self, manager):
        """Getting a non-existent credential returns None."""
        val = manager.get("agent", "nonexistent", "API_KEY", "user1", env_fallback=False)
        assert val is None

    def test_upsert_merges(self, manager):
        """Saving again merges new keys without removing old ones."""
        manager.save("agent", "ag1", {"key1": "v1"}, "u1")
        manager.save("agent", "ag1", {"key2": "v2"}, "u1")
        creds = manager.get_all("agent", "ag1", "u1")
        assert "key1" in creds
        assert "key2" in creds


class TestDelete:
    def test_delete_removes_credential(self, manager):
        """Delete should remove the credential row."""
        manager.save("agent", "ag1", {"k": "v"}, "u1")
        assert manager.has_valid("agent", "ag1", "u1")
        deleted = manager.delete("agent", "ag1", "u1")
        assert deleted is True
        assert manager.has_valid("agent", "ag1", "u1") is False

    def test_delete_nonexistent_returns_false(self, manager):
        """Deleting what doesn't exist returns False."""
        assert manager.delete("agent", "no_such", "u1") is False


class TestHasValid:
    def test_has_valid_true(self, manager):
        manager.save("system", "global", {"SHARED_KEY": "abc"}, "system")
        assert manager.has_valid("system", "global", "system") is True

    def test_has_valid_false(self, manager):
        assert manager.has_valid("system", "global", "system") is False


# ---------------------------------------------------------------------------
# .env Fallback
# ---------------------------------------------------------------------------

class TestEnvFallback:
    def test_falls_back_to_env(self, manager):
        """When DB is empty, should fall back to os.getenv."""
        with patch.dict(os.environ, {"MY_SPECIAL_KEY": "env_value"}):
            val = manager.get("agent", "ag1", "MY_SPECIAL_KEY", "u1", env_fallback=True)
            assert val == "env_value"

    def test_db_takes_precedence_over_env(self, manager):
        """DB credential should override .env."""
        manager.save("agent", "ag1", {"MY_SPECIAL_KEY": "db_value"}, "u1")
        with patch.dict(os.environ, {"MY_SPECIAL_KEY": "env_value"}):
            val = manager.get("agent", "ag1", "MY_SPECIAL_KEY", "u1", env_fallback=True)
            assert val == "db_value"

    def test_no_fallback_when_disabled(self, manager):
        """env_fallback=False should not check os.getenv."""
        with patch.dict(os.environ, {"MY_SPECIAL_KEY": "env_value"}):
            val = manager.get("agent", "ag1", "MY_SPECIAL_KEY", "u1", env_fallback=False)
            assert val is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_avoids_db_hit(self, manager):
        """Second get_all should hit cache, not DB."""
        manager.save("agent", "ag1", {"k": "v"}, "u1")
        _ = manager.get_all("agent", "ag1", "u1")  # populate cache

        # Spy on _load_from_generic_table
        manager._load_from_generic_table = MagicMock(return_value={})
        result = manager.get_all("agent", "ag1", "u1")  # should use cache
        manager._load_from_generic_table.assert_not_called()
        assert result == {"k": "v"}

    def test_cache_invalidated_on_save(self, manager):
        """Saving should invalidate the cache for that scope."""
        manager.save("agent", "ag1", {"k": "v1"}, "u1")
        _ = manager.get_all("agent", "ag1", "u1")  # populate cache
        manager.save("agent", "ag1", {"k": "v2"}, "u1")  # should invalidate
        result = manager.get_all("agent", "ag1", "u1")
        assert result["k"] == "v2"

    def test_cache_invalidated_on_delete(self, manager):
        """Deleting should invalidate the cache."""
        manager.save("agent", "ag1", {"k": "v"}, "u1")
        _ = manager.get_all("agent", "ag1", "u1")
        manager.delete("agent", "ag1", "u1")
        result = manager.get_all("agent", "ag1", "u1")
        assert result == {}


# ---------------------------------------------------------------------------
# Scope System
# ---------------------------------------------------------------------------

class TestScopes:
    def test_different_scopes_are_isolated(self, manager):
        """Credentials in different scopes don't leak."""
        manager.save("agent", "x", {"k": "agent_val"}, "u1")
        manager.save("tool", "x", {"k": "tool_val"}, "u1")
        manager.save("system", "x", {"k": "sys_val"}, "u1")

        assert manager.get("agent", "x", "k", "u1", env_fallback=False) == "agent_val"
        assert manager.get("tool", "x", "k", "u1", env_fallback=False) == "tool_val"
        assert manager.get("system", "x", "k", "u1", env_fallback=False) == "sys_val"

    def test_different_users_are_isolated(self, manager):
        """Same scope+scope_id but different users are separate."""
        manager.save("agent", "ag1", {"k": "user1_val"}, "user1")
        manager.save("agent", "ag1", {"k": "user2_val"}, "user2")

        assert manager.get("agent", "ag1", "k", "user1", env_fallback=False) == "user1_val"
        assert manager.get("agent", "ag1", "k", "user2", env_fallback=False) == "user2_val"


# ---------------------------------------------------------------------------
# Legacy Wrappers
# ---------------------------------------------------------------------------

class TestLegacyWrappers:
    def test_get_agent_credentials(self, manager):
        """Legacy wrapper should delegate to get_all."""
        manager.save("agent", "ag1", {"k": "v"}, "u1")
        result = manager.get_agent_credentials(None, "ag1", "u1")
        assert result == {"k": "v"}

    def test_save_agent_credentials(self, manager):
        """Legacy save wrapper."""
        assert manager.save_agent_credentials(None, "ag1", "u1", {"k": "v"}) is True
        assert manager.get_all("agent", "ag1", "u1") == {"k": "v"}

    def test_has_valid_credentials(self, manager):
        """Legacy has_valid wrapper."""
        assert manager.has_valid_credentials(None, "ag1", "u1") is False
        manager.save("agent", "ag1", {"k": "v"}, "u1")
        assert manager.has_valid_credentials(None, "ag1", "u1") is True

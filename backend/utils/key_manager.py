"""
API Key Manager - Unified key rotation for all LLM providers.

Handles rate limit-aware key rotation for any provider.
Supports both singular (API_KEY) and plural (API_KEYS) env vars.

Usage:
    # Get a key manager for any provider
    ollama_keys = KeyManager("OLLAMA_API_KEY")
    cerebras_keys = KeyManager("CEREBRAS_API_KEY")
    
    # Get best available key (handles rate limiting)
    key = ollama_keys.get_best_key_with_wait()
    
    # Report rate limit when you get 429
    ollama_keys.report_rate_limit(key)
"""

import os
import time
import logging
from typing import Dict, List, Optional
from itertools import cycle

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Manages API key rotation with smart cooldown handling.
    
    Tracks rate limit expiry for each key and prioritizes available keys.
    If all keys are limited, waits for the one with the shortest remaining cooldown.
    
    Usage:
        # For Ollama (supports OLLAMA_API_KEYS or OLLAMA_API_KEY)
        ollama = KeyManager("OLLAMA_API_KEY")
        key = ollama.get_best_key_with_wait()
        
        # For Cerebras (supports CEREBRAS_API_KEYS or CEREBRAS_API_KEY)
        cerebras = KeyManager("CEREBRAS_API_KEY")
        key = cerebras.get_best_key_with_wait()
    """
    
    def __init__(self, env_var_name: str):
        """
        Initialize key manager for a specific provider.
        
        Args:
            env_var_name: Environment variable name (e.g., "OLLAMA_API_KEY")
                         Will also check for plural form (e.g., "OLLAMA_API_KEYS")
        """
        self.env_var_name = env_var_name
        self.env_var_plural = env_var_name.replace("_KEY", "_KEYS")
        
        self._keys: List[str] = []
        self._key_cycle = None
        self._key_cooldowns: Dict[str, float] = {}
        self._current_key: Optional[str] = None
        self._initialized = False
        self._provider_name = env_var_name.replace("_API_KEY", "").lower()

        # Lazy initialization - load on first use
        self._try_init_from_env()

    def _try_init_from_env(self) -> bool:
        """Try to initialize from environment variables."""
        # Try plural form first (for key rotation)
        env_keys = os.getenv(self.env_var_plural)
        if env_keys:
            keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            if keys:
                logger.info(f"Loaded {len(keys)} {self._provider_name.upper()} API keys from {self.env_var_plural}")
                self._initialize(keys)
                return True
        
        # Fallback to singular form
        single_key = os.getenv(self.env_var_name)
        if single_key:
            logger.info(f"Loaded 1 {self._provider_name.upper()} API key from {self.env_var_name}")
            self._initialize([single_key.strip()])
            return True
        
        logger.debug(f"{self.env_var_name} not configured")
        return False

    def _initialize(self, keys: List[str]) -> None:
        """Initialize with the given keys."""
        self._keys = keys
        self._key_cycle = cycle(self._keys)
        self._key_cooldowns = {k: 0.0 for k in self._keys}
        self._current_key = next(self._key_cycle) if self._keys else None
        self._initialized = True

    def get_current_key(self) -> Optional[str]:
        """Get the currently active key if available, otherwise find best."""
        if self._is_available(self._current_key):
            return self._current_key
        return self.get_best_key_with_wait()

    def report_rate_limit(self, key: str, cooldown_seconds: int = 65):
        """Mark a key as rate limited with a cooldown (default 65s for safety)."""
        if key in self._key_cooldowns:
            expiry = time.time() + cooldown_seconds
            self._key_cooldowns[key] = expiry
            logger.warning(f"⏳ {self._provider_name.upper()} key ...{key[-4:]} rate limited. Cooling down for {cooldown_seconds}s")

    def _is_available(self, key: str) -> bool:
        """Check if a key is currently free of cooldowns."""
        if not key:
            return False
        return time.time() > self._key_cooldowns.get(key, 0)

    def get_best_key_with_wait(self) -> Optional[str]:
        """
        Finds the next available key.
        If ALL keys are rate limited, waits for the one with the shortest remaining time.
        """
        now = time.time()

        # Try to initialize from env if not done yet (lazy loading)
        if not self._initialized:
            self._try_init_from_env()

        # Handle case where no keys are configured
        if not self._keys:
            logger.warning(f"No {self._provider_name.upper()} API keys configured")
            return None

        # 1. Try to find an immediately available key
        for key in self._keys:
            if self._is_available(key):
                self._current_key = key
                logger.debug(f"✅ {self._provider_name.upper()} key available: ...{key[-4:]}")
                return key

        # 2. If we are here, ALL keys are limited.
        # Find the one with the minimum wait time.
        if not self._key_cooldowns:
            return None
        
        min_expiry = min(self._key_cooldowns.values())
        wait_time = max(0, min_expiry - now)

        if wait_time > 0:
            # Add a small buffer (1s) to be safe
            wait_time += 1.0
            logger.warning(f"🛑 All {self._provider_name.upper()} keys rate limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        # After waiting, find who is ready (should be at least one)
        for key in self._keys:
            if self._is_available(key):
                self._current_key = key
                logger.info(f"✅ {self._provider_name.upper()} key ready after wait: ...{key[-4:]}")
                return key

        # Fallback (shouldn't happen if logic is correct)
        return self._keys[0] if self._keys else None
    
    @property
    def keys(self) -> List[str]:
        """Get list of configured keys (for debugging/testing)."""
        return self._keys.copy()


# ============================================================================
# Global key managers for providers that need rotation
# Using single unified KeyManager class for ALL providers
# ============================================================================

# Single key manager instance - reused for all providers
_key_managers: Dict[str, KeyManager] = {}


def get_key_manager(provider: str) -> KeyManager:
    """
    Get or create a key manager for any provider.
    
    Usage:
        ollama_km = get_key_manager("OLLAMA_API_KEY")
        cerebras_km = get_key_manager("CEREBRAS_API_KEY")
    """
    if provider not in _key_managers:
        _key_managers[provider] = KeyManager(provider)
    return _key_managers[provider]


# ============================================================================
# Backward compatibility helpers (used by existing code)
# These are thin wrappers around get_key_manager()
# ============================================================================

def get_cerebras_key() -> Optional[str]:
    """Get best Cerebras key (might block if all rate limited)."""
    return get_key_manager("CEREBRAS_API_KEY").get_best_key_with_wait()


def report_rate_limit(key: str):
    """Report 429 for a Cerebras key."""
    get_key_manager("CEREBRAS_API_KEY").report_rate_limit(key)


def get_ollama_key() -> Optional[str]:
    """Get best Ollama key (might block if all rate limited)."""
    return get_key_manager("OLLAMA_API_KEY").get_best_key_with_wait()


def report_ollama_rate_limit(key: str):
    """Report 429 for an Ollama key."""
    get_key_manager("OLLAMA_API_KEY").report_rate_limit(key)


# ============================================================================
# Backward compatibility aliases (for inference_service)
# ============================================================================

# These are DEPRECATED but kept for backward compatibility
# New code should use get_key_manager() directly
cerebras_key_manager = get_key_manager("CEREBRAS_API_KEY")
ollama_key_manager = get_key_manager("OLLAMA_API_KEY")
key_manager = cerebras_key_manager  # Legacy alias


# ============================================================================
# Simple key getters for non-rotating providers
# These just load from .env with DB fallback (no rotation logic needed)
# ============================================================================

def get_simple_api_key(env_var: str, db_scope: str = "system", db_scope_id: str = "llm_providers") -> Optional[str]:
    """
    Get a simple API key (no rotation needed).
    
    Lookup order:
    1. Database (credential_manager)
    2. Environment variable
    3. None
    
    Usage:
        groq_key = get_simple_api_key("GROQ_API_KEY")
        nvidia_key = get_simple_api_key("NVIDIA_API_KEY")
    """
    try:
        from backend.services.credential_service import credential_manager
        db_key = credential_manager.get(db_scope, db_scope_id, env_var, env_fallback=False)
        if db_key:
            return db_key
    except Exception:
        pass
    
    # Fallback to .env
    return os.getenv(env_var)

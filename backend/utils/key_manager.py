import os
import time
import logging
import asyncio
from typing import List, Optional, Dict
from itertools import cycle

logger = logging.getLogger(__name__)


def load_keys_from_env() -> List[str]:
    """
    Load Cerebras API keys from CEREBRAS_API_KEYS env var (comma separated).
    Falls back to CEREBRAS_API_KEY (singular) if plural not set.
    Returns empty list if not configured - caller should handle appropriately.
    """
    env_keys = os.getenv("CEREBRAS_API_KEYS")
    if env_keys:
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
        if keys:
            logger.info(f"Loaded {len(keys)} Cerebras API keys from environment")
            return keys
    
    # Fallback to singular key name
    single_key = os.getenv("CEREBRAS_API_KEY")
    if single_key:
        logger.info("Loaded 1 Cerebras API key from environment (CEREBRAS_API_KEY)")
        return [single_key.strip()]
    
    logger.warning("CEREBRAS_API_KEY(S) not configured in environment - LLM calls may fail")
    return []

class KeyManager:
    """
    Manages API key rotation with smart cooldown handling.
    Tracks rate limit expiry for each key and prioritizes available keys.
    If all keys are limited, waits for the one with the shortest remaining cooldown.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self._keys: List[str] = []
        self._key_cycle = None
        self._key_cooldowns: Dict[str, float] = {}
        self._current_key: Optional[str] = None
        self._initialized = False
        
        # Initialize if keys provided or try to load from env
        if keys:
            self._initialize(keys)
        else:
            # Lazy initialization - will load on first use
            self._try_init_from_env()
    
    def _try_init_from_env(self) -> bool:
        """Try to initialize from environment variables."""
        keys = load_keys_from_env()
        if keys:
            self._initialize(keys)
            return True
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
            logger.warning(f"⏳ Key ...{key[-4:]} rate limited. Cooling down for {cooldown_seconds}s (until {time.strftime('%H:%M:%S', time.localtime(expiry))})")

    def _is_available(self, key: str) -> bool:
        """Check if a key is currently free of cooldowns."""
        if not key: return False
        return time.time() > self._key_cooldowns.get(key, 0)

    def get_best_key_with_wait(self) -> Optional[str]:
        """
        Finds the next available key.
        If ALL keys are rate limited, waits for the one with the shortest remaining time.
        """
        now = time.time()
        
        # Try to initialize from env if not done yet (lazy loading after .env is loaded)
        if not self._initialized:
            self._try_init_from_env()
        
        # Handle case where no keys are configured
        if not self._keys:
            logger.warning("No Cerebras API keys configured")
            return None
        
        # 1. Try to find an immediately available key
        # Check all keys to be fair (simple search)
        for key in self._keys:
            if self._is_available(key):
                self._current_key = key
                logger.info(f"✅ Found available key: ...{key[-4:]}")
                return key
                
        # 2. If we are here, ALL keys are limited.
        # Find the one with the minimum wait time.
        if not self._key_cooldowns:
            return None
        min_expiry = min(self._key_cooldowns.values())
        wait_time = min_expiry - now
        
        if wait_time > 0:
            # Add a small buffer (1s) to be safe
            wait_time += 1.0
            logger.warning(f"🛑 All keys rate limited. Waiting {wait_time:.1f}s for best key...")
            time.sleep(wait_time)
            
        # After waiting, find who is ready (should be at least one)
        for key in self._keys:
             if self._is_available(key):
                self._current_key = key
                logger.info(f"✅ Key ready after wait: ...{key[-4:]}")
                return key
                
        # Fallback (shouldn't happen if logic is correct)
        return self._keys[0]

# Global instance
key_manager = KeyManager()

def get_cerebras_key() -> str:
    """Helper to get best key (might block if all limited)"""
    return key_manager.get_best_key_with_wait()

def report_rate_limit(key: str):
    """Report 429 for a key"""
    key_manager.report_rate_limit(key)

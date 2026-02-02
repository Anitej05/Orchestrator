import os
import time
import logging
import asyncio
from typing import List, Optional, Dict
from itertools import cycle

logger = logging.getLogger(__name__)

# Keys extracted from browser_agent/llm.py
# User requested "any 3 api keys except 6th one".
CEREBRAS_KEYS = [
    "csk-52m4dv4chcpf9vy9jcmjrevnp5ft22y2vctd68wyr8dewndw",
    "csk-nnj93n833cr4c9rd2vttjeew3nwv494px62jfy45fmwjdch8",
    "csk-c2jjpt5k9kttxd44t9jwyn55vje4m2vmrvdjjkd6h2wphv6m",
]

class KeyManager:
    """
    Manages API key rotation with smart cooldown handling.
    Tracks rate limit expiry for each key and prioritizes available keys.
    If all keys are limited, waits for the one with the shortest remaining cooldown.
    """
    def __init__(self, keys: List[str] = CEREBRAS_KEYS):
        self._keys = keys
        self._key_cycle = cycle(keys)
        # Map key -> Unix timestamp when it becomes available
        self._key_cooldowns: Dict[str, float] = {k: 0.0 for k in keys}
        self._current_key = next(self._key_cycle) if keys else None
        
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
        
        # 1. Try to find an immediately available key
        # Check all keys to be fair (simple search)
        for key in self._keys:
            if self._is_available(key):
                self._current_key = key
                logger.info(f"✅ Found available key: ...{key[-4:]}")
                return key
                
        # 2. If we are here, ALL keys are limited.
        # Find the one with the minimum wait time.
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

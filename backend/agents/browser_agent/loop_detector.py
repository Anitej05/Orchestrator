import hashlib
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ActionLoopDetector:
    """Tracks action repetition and page stagnation for the browser agent."""
    
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.recent_action_hashes: List[str] = []
        self.recent_page_fingerprints: List[str] = []
        self.consecutive_stagnant = 0
        
    def _hash_action(self, action_name: str, params: Dict[str, Any]) -> str:
        """Hash an action and its meaningful parameters."""
        # Clean params depending on action type
        clean_params = {}
        if action_name == 'click':
            clean_params['index'] = params.get('index')
        elif action_name == 'type':
            clean_params['index'] = params.get('index')
            clean_params['text'] = params.get('text', '').lower().strip()
        elif action_name == 'scroll':
            clean_params['direction'] = params.get('direction')
        elif action_name == 'navigate':
            clean_params['url'] = params.get('url')
        else:
            # For other actions, just hash the name
            pass
            
        hash_str = f"{action_name}|{json.dumps(clean_params, sort_keys=True)}"
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def record_action(self, action_name: str, params: Dict[str, Any]) -> None:
        """Record an action and trim bounds to window size."""
        # Don't hash non-interactive actions (e.g. save_info, done)
        if action_name in ('save_info', 'done'):
            return
            
        action_hash = self._hash_action(action_name, params)
        self.recent_action_hashes.append(action_hash)
        
        if len(self.recent_action_hashes) > self.window_size:
            self.recent_action_hashes.pop(0)

    def record_page_state(self, url: str, element_count: int) -> None:
        """Track page fingerprint to detect if nothing is changing."""
        fingerprint = f"{url}|{element_count}"
        if self.recent_page_fingerprints and self.recent_page_fingerprints[-1] == fingerprint:
            self.consecutive_stagnant += 1
        else:
            self.consecutive_stagnant = 0
            
        self.recent_page_fingerprints.append(fingerprint)
        if len(self.recent_page_fingerprints) > self.window_size:
            self.recent_page_fingerprints.pop(0)

    def get_nudge(self) -> str | None:
        """Return an escalating nudge based on detected loops/stagnation."""
        # 1. Check for repeated actions (last N actions)
        if not self.recent_action_hashes:
            return None
            
        latest_hash = self.recent_action_hashes[-1]
        repetition_count = self.recent_action_hashes.count(latest_hash)
        
        if repetition_count >= 10:
            logger.warning("🔄 Loop Detector: Hard force-done (10 repetitions)")
            # This triggers the hard restriction in agent.py
            return None 
            
        elif repetition_count >= 7:
            logger.warning(f"🔄 Loop Detector: Action repeated {repetition_count} times")
            return (
                f"⚠️ YOU ARE STUCK IN A LOOP. You've repeated this action {repetition_count} times. "
                f"DO NOT do this again. Use `run_js` or `save_info` now, then call `done`."
            )
            
        elif repetition_count >= 4:
            logger.warning(f"🔄 Loop Detector: Action repeated {repetition_count} times")
            return (
                f"⚠️ You've repeated this action {repetition_count} times. "
                f"It is not working. Try a completely different approach. "
            )
            
        # 2. Check for page stagnation (doing things but nothing changes)
        if self.consecutive_stagnant >= 5:
            logger.warning(f"🧊 Loop Detector: Page stagnant for {self.consecutive_stagnant} steps")
            return (
                "⚠️ The page has not changed for 5 steps (same URL, same elements). "
                "Your actions are having no effect. Try `run_js` to extract data directly, "
                "or save what you currently know and call `done`."
            )
            
        return None

    def should_force_done(self) -> bool:
        """Return True if the agent is hopelessly stuck and must terminate."""
        if not self.recent_action_hashes:
            return False
            
        latest_hash = self.recent_action_hashes[-1]
        repetition_count = self.recent_action_hashes.count(latest_hash)
        
        if repetition_count >= 10:
            return True
        if self.consecutive_stagnant >= 8:
            return True
            
        return False

"""
Shared state module to avoid circular imports between main.py and orchestrator/graph.py
"""
from threading import Lock
from typing import Dict, Any

# Global conversation store for live updates
conversation_store: Dict[str, Dict[str, Any]] = {}
store_lock = Lock()

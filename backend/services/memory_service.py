"""
Memory Service — Persistent Knowledge Across Conversations

Wraps ArtifactStore with a clean, scoped interface for:
- Retrieving relevant memories (semantic search)
- Auto-capturing learnings from completed tasks
- Recording routing decisions (which agent worked for what)
- Managing user preferences
- System-level instructions (ORBIMESH.md)

Integrates with ContextPipeline to provide the right memories
at the right time without flooding the LLM context.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("MemoryService")

# Centralized storage paths
from backend.storage_config import SYSTEM_DIR
SYSTEM_MEMORY_PATH = SYSTEM_DIR / "ORBIMESH.md"


class MemoryService:
    """
    Persistent memory system across conversations.
    
    Wraps ArtifactStore with a clean API and adds scoped retrieval,
    auto-capture, and system-level memory support.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._artifact_store = None
        self._system_memory_cache: Optional[str] = None

    @property
    def artifact_store(self):
        """Lazy-load ArtifactStore to avoid circular imports."""
        if self._artifact_store is None:
            try:
                from backend.orchestrator.artifact_store import get_artifact_store
                self._artifact_store = get_artifact_store(self.user_id)
            except Exception as e:
                logger.warning(f"ArtifactStore unavailable: {e}")
        return self._artifact_store

    # ──────────────────────────────────────────────────────────────────────
    # RETRIEVAL — Pull relevant memories for context
    # ──────────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        scope: str = "all",
        top_k: int = 3,
        max_tokens: int = 500,
    ) -> str:
        """
        Retrieve relevant memories for a given query.
        
        Scopes:
        - "all": All memories (default)
        - "routing": Only routing decisions
        - "errors": Only error patterns
        - "tasks": Only task results
        - "preferences": Only user preferences
        """
        store = self.artifact_store
        if not store:
            return ""

        try:
            # Use hybrid semantic + keyword search
            results = store.retrieve_relevant(
                query=query,
                top_k=top_k,
                max_tokens=max_tokens * 3,  # char budget
            )

            if scope != "all" and results:
                # Filter by artifact type if scope is specific
                scope_type_map = {
                    "routing": "routing_knowledge",
                    "errors": "error_pattern",
                    "tasks": "task_result",
                    "preferences": "user_preference",
                }
                # For now, return all — scope filtering will be enhanced
                # when ArtifactStore supports type-based retrieval
                pass

            return results or ""

        except Exception as e:
            logger.debug(f"Memory recall failed: {e}")
            return ""

    def get_user_profile(self) -> str:
        """Get the user's profile/preferences summary."""
        store = self.artifact_store
        if not store:
            return "New user — no profile."
        try:
            return store.get_user_profile_prompt()
        except Exception as e:
            logger.debug(f"Profile retrieval failed: {e}")
            return "New user — no profile."

    # ──────────────────────────────────────────────────────────────────────
    # AUTO-CAPTURE — Learn from completed tasks
    # ──────────────────────────────────────────────────────────────────────

    def capture_from_result(
        self,
        task: str,
        agent_id: str,
        result: Dict[str, Any],
        success: bool = True,
    ) -> None:
        """
        Auto-capture learnings from a completed task.
        
        Called by Hands after successful agent execution.
        Delegates to ArtifactStore's capture_from_task().
        """
        store = self.artifact_store
        if not store:
            return

        try:
            # Build action entry compatible with ArtifactStore
            action_entry = {
                "action_type": "agent",
                "resource_id": agent_id,
                "instruction": task,
                "result_summary": result.get("task_summary", str(result)[:500]),
                "success": success,
                "execution_time": result.get("execution_time", 0),
            }

            state = {"original_prompt": task}
            store.capture_from_task(action_entry, state, objective=task)

            logger.debug(f"Captured memory from {agent_id} task")

        except Exception as e:
            logger.debug(f"Memory capture failed: {e}")

    def capture_routing_decision(
        self,
        prompt: str,
        agent_id: str,
        success: bool,
    ) -> None:
        """
        Remember which agent worked (or didn't) for a type of task.
        
        This builds up routing knowledge over time, so the Brain
        gets better at agent selection.
        """
        store = self.artifact_store
        if not store:
            return

        try:
            action_entry = {
                "action_type": "agent",
                "resource_id": agent_id,
                "instruction": prompt,
                "success": success,
            }

            state = {"original_prompt": prompt}
            store._capture_routing_knowledge(
                action_type="agent",
                resource_id=agent_id,
                instruction=prompt,
                objective=prompt,
            )

            logger.debug(
                f"Captured routing: {agent_id} {'✅' if success else '❌'} "
                f"for '{prompt[:50]}...'"
            )

        except Exception as e:
            logger.debug(f"Routing capture failed: {e}")

    def capture_error_pattern(
        self,
        task: str,
        agent_id: str,
        error_message: str,
    ) -> None:
        """
        Record error patterns to avoid repeating mistakes.
        """
        store = self.artifact_store
        if not store:
            return

        try:
            store._capture_error_pattern(
                action_type="agent",
                resource_id=agent_id,
                instruction=task,
                error_msg=error_message,
                objective=task,
            )

            logger.debug(f"Captured error pattern from {agent_id}")

        except Exception as e:
            logger.debug(f"Error capture failed: {e}")

    def capture_user_preference(self, key: str, value: str) -> None:
        """
        Store a discovered user preference.
        
        E.g., preferred email format, analysis style, timezone, etc.
        """
        store = self.artifact_store
        if not store:
            return

        try:
            store.update_user_profile(key, value)
            logger.debug(f"Captured preference: {key}={value[:50]}")
        except Exception as e:
            logger.debug(f"Preference capture failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # SYSTEM MEMORY — Equivalent to Claude Code's CLAUDE.md
    # ──────────────────────────────────────────────────────────────────────

    def get_system_context(self) -> str:
        """
        Load system-level instructions from ORBIMESH.md.
        
        This is the equivalent of Claude Code's CLAUDE.md — a user-editable
        file that provides persistent system context across all conversations.
        
        Location: storage/system/ORBIMESH.md
        """
        if self._system_memory_cache is not None:
            return self._system_memory_cache

        if SYSTEM_MEMORY_PATH.exists():
            try:
                self._system_memory_cache = SYSTEM_MEMORY_PATH.read_text(
                    encoding="utf-8"
                ).strip()
                logger.debug(
                    f"Loaded ORBIMESH.md ({len(self._system_memory_cache)} chars)"
                )
                return self._system_memory_cache
            except Exception as e:
                logger.warning(f"Failed to load ORBIMESH.md: {e}")

        self._system_memory_cache = ""
        return ""

    def save_system_context(self, content: str) -> bool:
        """Save system-level instructions to ORBIMESH.md."""
        try:
            SYSTEM_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            SYSTEM_MEMORY_PATH.write_text(content, encoding="utf-8")
            self._system_memory_cache = content
            logger.info(f"Saved ORBIMESH.md ({len(content)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to save ORBIMESH.md: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════

# Per-user memory service instances
_memory_services: Dict[str, MemoryService] = {}


def get_memory_service(user_id: str = "default") -> MemoryService:
    """Get or create a MemoryService instance for a user."""
    if user_id not in _memory_services:
        _memory_services[user_id] = MemoryService(user_id)
    return _memory_services[user_id]

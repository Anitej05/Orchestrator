"""
Canvas Registry Service

Thread-safe registry for managing multiple concurrent canvases across agents.
Replaces the single-slot canvas system with a full registry supporting:
- Multiple canvases per thread
- Priority-based auto-focus
- Versioning (updates increment version)
- Lifecycle management (active → archived → dismissed)
- Composite views (tabbed multi-canvas)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from backend.schemas import CanvasEntry, CanvasRegistryState

logger = logging.getLogger("CanvasRegistry")

# Priority constants
PRIORITY_CONFIRMATION = 100   # Confirmation dialogs auto-focus
PRIORITY_RESULT = 50          # Agent results
PRIORITY_PREVIEW = 10         # Previews and thumbnails
PRIORITY_DEFAULT = 0          # Default


class CanvasRegistry:
    """
    Thread-safe Canvas Registry for a single conversation thread.
    
    Manages multiple canvases from different agents with:
    - Priority-based focus (confirmation > result > preview)
    - Auto-versioning on re-registration
    - Status lifecycle (active, archived, dismissed)
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._canvases: Dict[str, CanvasEntry] = {}
        self._active_canvas_id: Optional[str] = None
        self._lock = asyncio.Lock()

    async def register(
        self,
        canvas_id: str,
        canvas_type: str,
        source_agent: str = "unknown",
        canvas_data: Optional[Dict[str, Any]] = None,
        canvas_content: Optional[str] = None,
        canvas_title: Optional[str] = None,
        priority: Optional[int] = None,
        requires_confirmation: bool = False,
        confirmation_message: Optional[str] = None,
        linked_file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> CanvasEntry:
        """
        Register or update a canvas in the registry.
        
        If canvas_id already exists, increments version and updates fields.
        Auto-focuses if this canvas has the highest priority.
        
        Returns the created/updated CanvasEntry.
        """
        async with self._lock:
            now = datetime.utcnow().isoformat()

            # Auto-assign priority if not provided
            if priority is None:
                if requires_confirmation:
                    priority = PRIORITY_CONFIRMATION
                elif canvas_type in ("spreadsheet", "email_preview", "document"):
                    priority = PRIORITY_RESULT
                else:
                    priority = PRIORITY_DEFAULT

            # Check if updating existing canvas
            existing = self._canvases.get(canvas_id)
            version = (existing.version + 1) if existing else 1
            created_at = existing.created_at if existing else now

            entry = CanvasEntry(
                canvas_id=canvas_id,
                canvas_type=canvas_type,
                canvas_data=canvas_data,
                canvas_content=canvas_content,
                canvas_title=canvas_title,
                source_agent=source_agent,
                priority=priority,
                version=version,
                created_at=created_at,
                updated_at=now,
                requires_confirmation=requires_confirmation,
                confirmation_message=confirmation_message,
                linked_file_id=linked_file_id,
                status="active",
                tags=tags or [],
            )

            self._canvases[canvas_id] = entry

            # Auto-focus: highest priority active canvas gets focus
            self._auto_focus()

            action = "updated" if existing else "registered"
            logger.info(
                f"🎨 Canvas {action}: {canvas_id} (type={canvas_type}, "
                f"agent={source_agent}, priority={priority}, v{version})"
            )

            return entry

    def register_sync(
        self,
        canvas_id: str,
        canvas_type: str,
        source_agent: str = "unknown",
        canvas_data: Optional[Dict[str, Any]] = None,
        canvas_content: Optional[str] = None,
        canvas_title: Optional[str] = None,
        priority: Optional[int] = None,
        requires_confirmation: bool = False,
        confirmation_message: Optional[str] = None,
        linked_file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> CanvasEntry:
        """
        Synchronous version of register() for use in non-async contexts
        (e.g., Hands._update_state_with_result which is sync).
        """
        now = datetime.utcnow().isoformat()

        if priority is None:
            if requires_confirmation:
                priority = PRIORITY_CONFIRMATION
            elif canvas_type in ("spreadsheet", "email_preview", "document"):
                priority = PRIORITY_RESULT
            else:
                priority = PRIORITY_DEFAULT

        existing = self._canvases.get(canvas_id)
        version = (existing.version + 1) if existing else 1
        created_at = existing.created_at if existing else now

        entry = CanvasEntry(
            canvas_id=canvas_id,
            canvas_type=canvas_type,
            canvas_data=canvas_data,
            canvas_content=canvas_content,
            canvas_title=canvas_title,
            source_agent=source_agent,
            priority=priority,
            version=version,
            created_at=created_at,
            updated_at=now,
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message,
            linked_file_id=linked_file_id,
            status="active",
            tags=tags or [],
        )

        self._canvases[canvas_id] = entry
        self._auto_focus()

        action = "updated" if existing else "registered"
        logger.info(
            f"🎨 Canvas {action}: {canvas_id} (type={canvas_type}, "
            f"agent={source_agent}, priority={priority}, v{version})"
        )

        return entry

    async def update(self, canvas_id: str, **updates) -> Optional[CanvasEntry]:
        """Partial update of an existing canvas entry."""
        async with self._lock:
            entry = self._canvases.get(canvas_id)
            if not entry:
                return None

            entry_dict = entry.model_dump()
            entry_dict.update(updates)
            entry_dict["version"] = entry.version + 1
            entry_dict["updated_at"] = datetime.utcnow().isoformat()

            updated_entry = CanvasEntry(**entry_dict)
            self._canvases[canvas_id] = updated_entry
            self._auto_focus()

            logger.info(f"🎨 Canvas updated: {canvas_id} v{updated_entry.version}")
            return updated_entry

    async def set_active(self, canvas_id: str) -> bool:
        """Manually set the active canvas."""
        async with self._lock:
            if canvas_id in self._canvases and self._canvases[canvas_id].status == "active":
                self._active_canvas_id = canvas_id
                logger.info(f"🎨 Active canvas set: {canvas_id}")
                return True
            return False

    async def dismiss(self, canvas_id: str) -> bool:
        """Dismiss a canvas (hidden but not deleted)."""
        async with self._lock:
            entry = self._canvases.get(canvas_id)
            if not entry:
                return False
            
            entry_dict = entry.model_dump()
            entry_dict["status"] = "dismissed"
            entry_dict["updated_at"] = datetime.utcnow().isoformat()
            self._canvases[canvas_id] = CanvasEntry(**entry_dict)

            # If dismissed canvas was active, auto-focus next
            if self._active_canvas_id == canvas_id:
                self._active_canvas_id = None
                self._auto_focus()

            logger.info(f"🎨 Canvas dismissed: {canvas_id}")
            return True

    async def archive(self, canvas_id: str) -> bool:
        """Archive a canvas (persisted but not displayed)."""
        async with self._lock:
            entry = self._canvases.get(canvas_id)
            if not entry:
                return False

            entry_dict = entry.model_dump()
            entry_dict["status"] = "archived"
            entry_dict["updated_at"] = datetime.utcnow().isoformat()
            self._canvases[canvas_id] = CanvasEntry(**entry_dict)

            if self._active_canvas_id == canvas_id:
                self._active_canvas_id = None
                self._auto_focus()

            logger.info(f"🎨 Canvas archived: {canvas_id}")
            return True

    def get_active(self) -> Optional[CanvasEntry]:
        """Get the currently active canvas."""
        if self._active_canvas_id:
            return self._canvases.get(self._active_canvas_id)
        return None

    def get_active_id(self) -> Optional[str]:
        """Get the active canvas ID."""
        return self._active_canvas_id

    def get_all(self, status: str = "active") -> List[CanvasEntry]:
        """Get all canvases with the given status."""
        return [
            entry for entry in self._canvases.values()
            if entry.status == status
        ]

    def get_by_agent(self, agent_id: str) -> List[CanvasEntry]:
        """Get all canvases from a specific agent."""
        return [
            entry for entry in self._canvases.values()
            if entry.source_agent == agent_id and entry.status == "active"
        ]

    def get_canvas(self, canvas_id: str) -> Optional[CanvasEntry]:
        """Get a specific canvas by ID."""
        return self._canvases.get(canvas_id)

    def get_registry_state(self) -> CanvasRegistryState:
        """Get the full registry state for API responses."""
        active_canvases = {
            cid: entry for cid, entry in self._canvases.items()
            if entry.status == "active"
        }

        # Order by priority (descending), then by created_at (descending)
        canvas_order = sorted(
            active_canvases.keys(),
            key=lambda cid: (
                active_canvases[cid].priority,
                active_canvases[cid].created_at,
            ),
            reverse=True,
        )

        return CanvasRegistryState(
            canvases=active_canvases,
            active_canvas_id=self._active_canvas_id,
            canvas_order=canvas_order,
        )

    def get_backward_compat_fields(self) -> Dict[str, Any]:
        """
        Generate backward-compatible fields from the active canvas.
        Populates: has_canvas, canvas_type, canvas_content, canvas_data, etc.
        """
        active = self.get_active()
        if not active:
            return {
                "has_canvas": False,
                "canvas_type": None,
                "canvas_content": None,
                "canvas_data": None,
                "canvas_title": None,
                "browser_view": None,
                "plan_view": None,
                "current_view": None,
            }

        # Determine current_view based on canvas type
        current_view = "browser"
        if active.canvas_type == "spreadsheet":
            current_view = "spreadsheet"
        elif active.canvas_type == "plan_graph":
            current_view = "plan"
        elif active.canvas_type in ("html", "email_preview"):
            current_view = "browser"

        return {
            "has_canvas": True,
            "canvas_type": active.canvas_type,
            "canvas_content": active.canvas_content,
            "canvas_data": active.canvas_data,
            "canvas_title": active.canvas_title,
            "browser_view": active.canvas_content if active.canvas_type == "html" else None,
            "plan_view": active.canvas_data if active.canvas_type == "plan_graph" else None,
            "current_view": current_view,
        }

    def clear(self):
        """Clear all canvases for this thread."""
        self._canvases.clear()
        self._active_canvas_id = None
        logger.info(f"🎨 Canvas registry cleared for thread {self.thread_id}")

    def _auto_focus(self):
        """Auto-focus the highest priority active canvas."""
        active_canvases = [
            (cid, entry) for cid, entry in self._canvases.items()
            if entry.status == "active"
        ]

        if not active_canvases:
            self._active_canvas_id = None
            return

        # Sort by priority (desc), then by updated_at (desc) for tie-breaking
        active_canvases.sort(
            key=lambda x: (x[1].priority, x[1].updated_at),
            reverse=True,
        )

        self._active_canvas_id = active_canvases[0][0]


# ============================================================================
# Thread-level Registry Manager
# ============================================================================

_registries: Dict[str, CanvasRegistry] = {}
_manager_lock = asyncio.Lock()


def get_canvas_registry(thread_id: str) -> CanvasRegistry:
    """
    Get or create a CanvasRegistry for a thread.
    Thread-safe singleton per thread_id.
    """
    if thread_id not in _registries:
        _registries[thread_id] = CanvasRegistry(thread_id)
    return _registries[thread_id]


def clear_canvas_registry(thread_id: str):
    """Clear and remove a thread's canvas registry."""
    if thread_id in _registries:
        _registries[thread_id].clear()
        del _registries[thread_id]


def get_all_registries() -> Dict[str, CanvasRegistry]:
    """Get all active registries (for debugging/admin)."""
    return _registries

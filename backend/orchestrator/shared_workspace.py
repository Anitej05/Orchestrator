"""
Shared Workspace Manager for Cross-Conversation Persistence

This extends the workspace manager to support a shared workspace
that persists across all conversations for a user.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from .workspace_manager import FileMetadata, WorkspaceManager, STORAGE_BASE
import logging

logger = logging.getLogger(__name__)

# Shared workspace location
SHARED_WORKSPACE = STORAGE_BASE / "shared"


class SharedWorkspaceManager(WorkspaceManager):
    """
    Manages a shared workspace that persists across all conversations.
    
    Use this for files that should be accessible from any conversation,
    such as:
    - User preferences/settings
    - Templates
    - Shared resources
    - Files explicitly marked as "shared"
    """
    
    def __init__(self, user_id: str = "default"):
        # Override to use shared workspace instead of thread-specific
        self.user_id = user_id
        # Use user_id as thread_id for the shared workspace context
        self.thread_id = f"shared_{user_id}"
        self.workspace_path = SHARED_WORKSPACE / user_id
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.workspace_path / ".file_index.json"
        self._created_files: List[FileMetadata] = []
        self._load_index()
    
    def share_file(self, file_path: str, description: str = "") -> FileMetadata:
        """
        Move a file from thread workspace to shared workspace.
        Returns the shared file metadata.
        """
        import shutil
        
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Copy to shared workspace
        dest = self.workspace_path / source.name
        shutil.copy2(file_path, dest)
        
        # Add to index
        metadata = self.add_file(
            file_path=str(dest),
            file_name=dest.name,
            file_type=self._detect_file_type(dest),
            created_by="user_shared",
            description=description
        )
        
        logger.info(f"File shared: {source.name} → shared workspace")
        return metadata


# Singleton registry
_shared_workspace_managers: Dict[str, SharedWorkspaceManager] = {}

def get_shared_workspace_manager(user_id: str = "default") -> SharedWorkspaceManager:
    """Get or create a shared workspace manager for a user."""
    if user_id not in _shared_workspace_managers:
        _shared_workspace_managers[user_id] = SharedWorkspaceManager(user_id)
    return _shared_workspace_managers[user_id]


def get_all_accessible_files(thread_id: str, user_id: str = "default") -> Dict[str, Any]:
    """
    Get all files accessible to the orchestrator:
    - Thread-specific files (current conversation)
    - Shared files (across all conversations)
    - Agent workspace files
    
    Returns structured data for the brain prompt.
    """
    from .workspace_manager import get_workspace_manager
    
    # Get thread workspace
    thread_wm = get_workspace_manager(thread_id)
    
    # Get shared workspace
    shared_wm = get_shared_workspace_manager(user_id)
    
    # Get agent workspaces
    agent_files = thread_wm.discover_agent_files()
    
    return {
        "thread_workspace": {
            "path": str(thread_wm.get_workspace_path()),
            "files": [f.to_dict() for f in thread_wm.list_files()]
        },
        "shared_workspace": {
            "path": str(shared_wm.get_workspace_path()),
            "files": [f.to_dict() for f in shared_wm.list_files()]
        },
        "agent_workspaces": {
            agent_id: [f.to_dict() for f in files]
            for agent_id, files in agent_files.items()
        }
    }


def to_prompt_context_with_shared(thread_id: str, user_id: str = "default") -> str:
    """
    Generate a prompt-friendly string describing all accessible files
    including shared workspace.
    """
    from .workspace_manager import get_workspace_manager
    
    thread_wm = get_workspace_manager(thread_id)
    shared_wm = get_shared_workspace_manager(user_id)
    
    lines = []
    
    # Thread-specific files
    thread_files = thread_wm.list_files()
    if thread_files:
        lines.append("## FILES IN THIS CONVERSATION")
        lines.append(f"(Location: {thread_wm.get_workspace_path()})")
        for f in thread_files:
            size_kb = f.size_bytes / 1024
            lines.append(f"- {f.file_name} ({f.file_type}, {size_kb:.1f} KB)")
        lines.append("")
    
    # Shared files
    shared_files = shared_wm.list_files()
    if shared_files:
        lines.append("## SHARED FILES (Available in all your conversations)")
        lines.append(f"(Location: {shared_wm.get_workspace_path()})")
        for f in shared_files:
            size_kb = f.size_bytes / 1024
            lines.append(f"- {f.file_name} ({f.file_type}, {size_kb:.1f} KB)")
            if f.description:
                lines.append(f"  Note: {f.description}")
        lines.append("")
    
    # Agent files
    agent_files = thread_wm.discover_agent_files()
    if agent_files:
        lines.append("## AGENT WORKSPACES")
        for agent_id, files in agent_files.items():
            lines.append(f"\n{agent_id}:")
            for f in files[:5]:
                lines.append(f"  - {f.file_name}")
            if len(files) > 5:
                lines.append(f"  ... and {len(files) - 5} more files")
    
    return "\n".join(lines) if lines else "No files available."

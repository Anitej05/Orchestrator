"""
Workspace Manager for Orchestrator

Manages file tracking across:
- Orchestrator's own workspace
- All agent workspaces
- Files created during conversation (persisted per thread)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Base storage path
STORAGE_BASE = Path(__file__).parent.parent / "storage"
ORCHESTRATOR_WORKSPACE = STORAGE_BASE / "orchestrator"

# Ensure orchestrator workspace exists
ORCHESTRATOR_WORKSPACE.mkdir(parents=True, exist_ok=True)


class FileMetadata:
    """Metadata for a tracked file."""
    def __init__(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        created_by: str,  # 'orchestrator', 'python', 'terminal', 'agent:{agent_id}'
        created_at: Optional[str] = None,
        description: str = "",
        size_bytes: int = 0,
        conversation_thread: str = "default"
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type
        self.created_by = created_by
        self.created_at = created_at or datetime.now().isoformat()
        self.description = description
        self.size_bytes = size_bytes
        self.conversation_thread = conversation_thread
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "description": self.description,
            "size_bytes": self.size_bytes,
            "conversation_thread": self.conversation_thread
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        return cls(**data)


class WorkspaceManager:
    """
    Manages file tracking and discovery across all workspaces.
    
    Responsibilities:
    1. Track files created by orchestrator during conversation
    2. Discover files in agent workspaces
    3. Provide file search across all workspaces
    4. Persist file index per conversation thread
    """
    
    def __init__(self, thread_id: str = "default"):
        self.thread_id = thread_id
        self.workspace_path = ORCHESTRATOR_WORKSPACE / thread_id
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.workspace_path / ".file_index.json"
        self._created_files: List[FileMetadata] = []
        self._load_index()
    
    def _load_index(self):
        """Load the file index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self._created_files = [
                        FileMetadata.from_dict(item) for item in data.get('files', [])
                    ]
                logger.info(f"Loaded {len(self._created_files)} files from index for thread {self.thread_id}")
            except Exception as e:
                logger.error(f"Failed to load file index: {e}")
                self._created_files = []
        else:
            self._created_files = []
    
    def _save_index(self):
        """Save the file index to disk."""
        try:
            data = {
                'thread_id': self.thread_id,
                'last_updated': datetime.now().isoformat(),
                'files': [f.to_dict() for f in self._created_files]
            }
            with open(self.index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save file index: {e}")
    
    def get_workspace_path(self) -> Path:
        """Get the orchestrator's workspace path for this thread."""
        return self.workspace_path
    
    def scan_for_new_files(self, created_by: str = "orchestrator") -> List[FileMetadata]:
        """
        Scan workspace for files not in index.
        Returns list of newly discovered files.
        """
        new_files = []
        existing_paths = {f.file_path for f in self._created_files}
        
        # Scan orchestrator workspace
        for file_path in self.workspace_path.iterdir():
            if file_path.is_file() and str(file_path) not in existing_paths:
                # Skip the index file itself
                if file_path.name == ".file_index.json":
                    continue
                
                metadata = FileMetadata(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    file_type=self._detect_file_type(file_path),
                    created_by=created_by,
                    size_bytes=file_path.stat().st_size,
                    conversation_thread=self.thread_id
                )
                new_files.append(metadata)
                self._created_files.append(metadata)
        
        if new_files:
            self._save_index()
            logger.info(f"Discovered {len(new_files)} new files in workspace")
        
        return new_files
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension."""
        ext = file_path.suffix.lower()
        type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.py': 'text/x-python',
            '.html': 'text/html',
            '.md': 'text/markdown'
        }
        return type_map.get(ext, 'application/octet-stream')
    
    def add_file(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        created_by: str,
        description: str = ""
    ) -> FileMetadata:
        """Add a file to the tracking index."""
        # If file is relative path, resolve to workspace
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_path / path
        
        metadata = FileMetadata(
            file_path=str(path),
            file_name=file_name,
            file_type=file_type,
            created_by=created_by,
            description=description,
            size_bytes=path.stat().st_size if path.exists() else 0,
            conversation_thread=self.thread_id
        )
        
        self._created_files.append(metadata)
        self._save_index()
        return metadata
    
    def list_files(
        self,
        file_type: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> List[FileMetadata]:
        """List files in orchestrator workspace, optionally filtered."""
        files = self._created_files
        
        if file_type:
            files = [f for f in files if f.file_type == file_type or f.file_type.startswith(file_type)]
        
        if created_by:
            files = [f for f in files if f.created_by == created_by]
        
        return files
    
    def search_files(self, query: str) -> List[FileMetadata]:
        """Search files by name or description."""
        query = query.lower()
        results = []
        
        for f in self._created_files:
            if query in f.file_name.lower() or query in f.description.lower():
                results.append(f)
        
        return results
    
    def get_file(self, file_name: str) -> Optional[FileMetadata]:
        """Get a specific file by name."""
        for f in self._created_files:
            if f.file_name == file_name:
                return f
        return None
    
    def discover_agent_files(self) -> Dict[str, List[FileMetadata]]:
        """
        Discover files in all agent workspaces.
        Returns dict: {agent_id: [FileMetadata, ...]}
        """
        agent_files = {}
        
        if not STORAGE_BASE.exists():
            return agent_files
        
        for agent_dir in STORAGE_BASE.iterdir():
            if agent_dir.is_dir() and agent_dir.name not in ['orchestrator', 'content', 'system', 'vector_store']:
                files = []
                for file_path in agent_dir.iterdir():
                    if file_path.is_file():
                        metadata = FileMetadata(
                            file_path=str(file_path),
                            file_name=file_path.name,
                            file_type=self._detect_file_type(file_path),
                            created_by=f"agent:{agent_dir.name}",
                            size_bytes=file_path.stat().st_size,
                            conversation_thread=self.thread_id
                        )
                        files.append(metadata)
                
                if files:
                    agent_files[agent_dir.name] = files
        
        return agent_files
    
    def get_all_accessible_files(self) -> Dict[str, Any]:
        """
        Get all files accessible to the orchestrator.
        Returns structured information about all workspaces.
        """
        return {
            "orchestrator_workspace": str(self.workspace_path),
            "orchestrator_files": [f.to_dict() for f in self._created_files],
            "agent_workspaces": {
                agent_id: [f.to_dict() for f in files]
                for agent_id, files in self.discover_agent_files().items()
            }
        }
    
    def to_prompt_context(self) -> str:
        """Generate a prompt-friendly string describing accessible files."""
        lines = []
        
        # Orchestrator files
        if self._created_files:
            lines.append("## FILES IN YOUR WORKSPACE")
            lines.append(f"(Location: {self.workspace_path})")
            for f in self._created_files:
                size_kb = f.size_bytes / 1024
                lines.append(f"- {f.file_name} ({f.file_type}, {size_kb:.1f} KB)")
                if f.description:
                    lines.append(f"  Description: {f.description}")
        else:
            lines.append("## YOUR WORKSPACE")
            lines.append(f"(Location: {self.workspace_path})")
            lines.append("No files created yet in this conversation.")
        
        # Agent files
        agent_files = self.discover_agent_files()
        if agent_files:
            lines.append("\n## AGENT WORKSPACES")
            for agent_id, files in agent_files.items():
                lines.append(f"\n{agent_id}:")
                for f in files[:5]:  # Limit to 5 files per agent
                    lines.append(f"  - {f.file_name}")
                if len(files) > 5:
                    lines.append(f"  ... and {len(files) - 5} more files")
        
        return "\n".join(lines)


# Singleton registry to manage workspace managers per thread
_workspace_managers: Dict[str, WorkspaceManager] = {}

def get_workspace_manager(thread_id: str = "default") -> WorkspaceManager:
    """Get or create a workspace manager for a thread."""
    if thread_id not in _workspace_managers:
        _workspace_managers[thread_id] = WorkspaceManager(thread_id)
    return _workspace_managers[thread_id]

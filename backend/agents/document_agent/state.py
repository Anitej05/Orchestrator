"""
Document Agent - State Management

Manages document sessions, version control, and editing history.
Optimized for cloud deployment with minimal persistent storage.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from threading import RLock
import shutil
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class EditAction:
    """Represents a single edit action."""
    timestamp: str
    action_type: str
    instruction: str
    parameters: Dict[str, Any]
    result: str
    success: bool


@dataclass
class DocumentSession:
    """Represents an editing session for a document."""
    document_path: str
    document_name: str
    session_id: str
    created_at: str
    last_accessed: str
    edit_history: List[EditAction]
    conversation_context: List[Dict[str, str]]
    metadata: Dict[str, Any]


class DocumentSessionManager:
    """Manages editing sessions with thread-safe concurrent access."""

    def __init__(self, sessions_dir: str = None):
        """Initialize session manager with optional custom directory."""
        if sessions_dir is None:
            # Get workspace root (4 levels up: state.py -> document_agent -> agents -> backend -> root)
            workspace_root = Path(__file__).parent.parent.parent.parent.resolve()
            sessions_dir = workspace_root / "storage" / "document_sessions"
        
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, DocumentSession] = {}
        self._session_lock = RLock()

    def _get_session_id(self, document_path: str, thread_id: str = None) -> str:
        """Generate unique session ID."""
        combined = f"{thread_id}:{document_path}" if thread_id else document_path
        return hashlib.md5(combined.encode()).hexdigest()

    def get_or_create_session(
        self,
        document_path: str,
        document_name: str,
        thread_id: str = None
    ) -> DocumentSession:
        """Get or create a session for a document."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)

            if session_id in self._active_sessions:
                return self._active_sessions[session_id]

            session_file = self.sessions_dir / f"{session_id}.json"

            if session_file.exists():
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    session = self._deserialize_session(data)
                    self._active_sessions[session_id] = session
                    return session

            # Create new session
            now = datetime.utcnow().isoformat()
            session = DocumentSession(
                document_path=document_path,
                document_name=document_name,
                session_id=session_id,
                created_at=now,
                last_accessed=now,
                edit_history=[],
                conversation_context=[],
                metadata={}
            )

            self._active_sessions[session_id] = session
            self._save_session(session)
            logger.info(f"Created session {session_id[:8]} for {document_name}")
            return session

    def add_edit_action(
        self,
        session_id: str,
        action: EditAction
    ) -> None:
        """Add an edit action to session history."""
        with self._session_lock:
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.edit_history.append(action)
                session.last_accessed = datetime.utcnow().isoformat()
                self._save_session(session)
                logger.info(f"Added edit action: {action.action_type}")

    def get_session_history(self, session_id: str) -> List[EditAction]:
        """Get edit history for a session."""
        with self._session_lock:
            if session_id in self._active_sessions:
                return self._active_sessions[session_id].edit_history
            return []

    def add_conversation_turn(
        self,
        document_path: str,
        user_message: str,
        agent_response: str,
        thread_id: str = None
    ) -> None:
        """Add a conversation turn to the session context."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            if session_id in self._active_sessions:
                session = self._active_sessions[session_id]
                session.conversation_context.append({
                    'role': 'user',
                    'content': user_message,
                    'timestamp': datetime.utcnow().isoformat()
                })
                session.conversation_context.append({
                    'role': 'assistant',
                    'content': agent_response,
                    'timestamp': datetime.utcnow().isoformat()
                })
                session.last_accessed = datetime.utcnow().isoformat()
                self._save_session(session)
                logger.info(f"Added conversation turn to session {session_id[:8]}")

    def get_session_context(self, document_path: str, thread_id: str = None) -> str:
        """Get formatted session context including edit history and conversation."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            if session_id not in self._active_sessions:
                return "No session context available."
            
            session = self._active_sessions[session_id]
            context_parts = []
            
            # Document info
            context_parts.append(f"Document: {session.document_name}")
            context_parts.append(f"Session started: {session.created_at}")
            context_parts.append(f"Last accessed: {session.last_accessed}")
            context_parts.append("")
            
            # Edit history
            if session.edit_history:
                context_parts.append("Edit History:")
                for i, action in enumerate(session.edit_history[-10:], 1):  # Last 10 actions
                    status = "✓" if action.success else "✗"
                    context_parts.append(
                        f"{i}. [{status}] {action.action_type}: {action.instruction[:100]}"
                    )
                context_parts.append("")
            
            # Conversation context
            if session.conversation_context:
                context_parts.append("Recent Conversation:")
                for turn in session.conversation_context[-6:]:  # Last 3 exchanges
                    role = turn.get('role', 'unknown').capitalize()
                    content = turn.get('content', '')[:150]
                    context_parts.append(f"{role}: {content}")
            
            return "\n".join(context_parts)

    def get_edit_history_summary(self, document_path: str, thread_id: str = None) -> List[Dict[str, Any]]:
        """Get a summary of edit history for display."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            if session_id not in self._active_sessions:
                return []
            
            session = self._active_sessions[session_id]
            summary = []
            
            for action in session.edit_history:
                summary.append({
                    'timestamp': action.timestamp,
                    'action_type': action.action_type,
                    'instruction': action.instruction,
                    'result': action.result,
                    'success': action.success
                })
            
            return summary

    def can_undo(self, document_path: str, thread_id: str = None) -> bool:
        """Check if undo is available for this document."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            if session_id not in self._active_sessions:
                return False
            
            session = self._active_sessions[session_id]
            return len(session.edit_history) > 0

    def get_last_action(self, document_path: str, thread_id: str = None) -> Optional[EditAction]:
        """Get the last edit action performed on this document."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            if session_id not in self._active_sessions:
                return None
            
            session = self._active_sessions[session_id]
            if session.edit_history:
                return session.edit_history[-1]
            return None

    def clear_session(self, document_path: str, thread_id: str = None) -> None:
        """Clear a session from memory and disk."""
        with self._session_lock:
            session_id = self._get_session_id(document_path, thread_id)
            
            # Remove from active sessions
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            # Remove session file
            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                try:
                    session_file.unlink()
                    logger.info(f"Cleared session {session_id[:8]}")
                except Exception as e:
                    logger.error(f"Failed to delete session file: {e}")

    def _save_session(self, session: DocumentSession) -> None:
        """Save session to disk."""
        session_file = self.sessions_dir / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self._serialize_session(session), f, indent=2)

    @staticmethod
    def _serialize_session(session: DocumentSession) -> Dict[str, Any]:
        """Serialize session to JSON-compatible format."""
        return {
            'document_path': session.document_path,
            'document_name': session.document_name,
            'session_id': session.session_id,
            'created_at': session.created_at,
            'last_accessed': session.last_accessed,
            'edit_history': [asdict(a) for a in session.edit_history],
            'conversation_context': session.conversation_context,
            'metadata': session.metadata
        }

    @staticmethod
    def _deserialize_session(data: Dict[str, Any]) -> DocumentSession:
        """Deserialize session from JSON."""
        return DocumentSession(
            document_path=data['document_path'],
            document_name=data['document_name'],
            session_id=data['session_id'],
            created_at=data['created_at'],
            last_accessed=data['last_accessed'],
            edit_history=[EditAction(**a) for a in data.get('edit_history', [])],
            conversation_context=data.get('conversation_context', []),
            metadata=data.get('metadata', {})
        )


class DocumentVersionManager:
    """Manages document versions with undo/redo capability."""

    def __init__(self, base_dir: str = None):
        """Initialize version manager."""
        if base_dir is None:
            # Get workspace root (4 levels up: state.py -> document_agent -> agents -> backend -> root)
            workspace_root = Path(__file__).parent.parent.parent.parent.resolve()
            base_dir = workspace_root / "storage" / "document_versions"
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.version_index_file = self.base_dir / "version_index.json"
        self.index: Dict[str, Any] = {}
        self._index_lock = RLock()
        self._load_index()

    def _load_index(self) -> None:
        """Load version index from disk."""
        with self._index_lock:
            if self.version_index_file.exists():
                with open(self.version_index_file, 'r') as f:
                    self.index = json.load(f)
            else:
                self.index = {}

    def _save_index(self) -> None:
        """Save version index to disk."""
        with self._index_lock:
            with open(self.version_index_file, 'w') as f:
                json.dump(self.index, f, indent=2)

    def _get_document_key(self, file_path: str) -> str:
        """Generate unique key for document."""
        return os.path.normpath(file_path).replace(os.sep, '_')

    def save_version(self, file_path: str, description: str = "Edit") -> str:
        """Save a document version."""
        doc_key = self._get_document_key(file_path)

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with self._index_lock:
            if doc_key not in self.index:
                self.index[doc_key] = {
                    "file_path": file_path,
                    "versions": [],
                    "current_version": -1
                }

            version_id = f"v{int(time.time() * 1000)}"
            version_dir = self.base_dir / doc_key / version_id
            version_dir.mkdir(parents=True, exist_ok=True)

            # Copy file
            file_name = Path(file_path).name
            version_file = version_dir / file_name
            shutil.copy2(file_path, version_file)

            # Save metadata
            metadata = {
                "version_id": version_id,
                "timestamp": time.time(),
                "description": description,
                "file_path": str(version_file),
                "original_path": file_path
            }

            with open(version_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            # Update index
            doc_history = self.index[doc_key]
            if doc_history["current_version"] < len(doc_history["versions"]) - 1:
                doc_history["versions"] = doc_history["versions"][:doc_history["current_version"] + 1]

            doc_history["versions"].append(metadata)
            doc_history["current_version"] = len(doc_history["versions"]) - 1

            self._save_index()
            logger.info(f"Saved version {version_id} for {file_path}")
            return version_id

    def get_versions(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all versions of a document."""
        doc_key = self._get_document_key(file_path)
        if doc_key in self.index:
            return self.index[doc_key].get("versions", [])
        return []

    def restore_version(self, file_path: str, version_id: str) -> bool:
        """Restore a specific version."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return False

        versions = self.index[doc_key].get("versions", [])
        for i, v in enumerate(versions):
            if v["version_id"] == version_id:
                version_file = v["file_path"]
                if Path(version_file).exists():
                    shutil.copy2(version_file, file_path)
                    self.index[doc_key]["current_version"] = i
                    self._save_index()
                    logger.info(f"Restored version {version_id} to {file_path}")
                    return True
        
        return False

    def cleanup_old_versions(self, file_path: str, keep_count: int = 10) -> int:
        """Remove old versions keeping only recent ones (for cloud optimization)."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return 0

        doc_history = self.index[doc_key]
        versions = doc_history["versions"]
        
        if len(versions) <= keep_count:
            return 0

        # Keep latest versions
        to_delete = versions[:-keep_count]
        deleted_count = 0

        for v in to_delete:
            try:
                version_dir = Path(v["file_path"]).parent
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete version: {e}")

        doc_history["versions"] = versions[-keep_count:]
        doc_history["current_version"] = len(doc_history["versions"]) - 1
        self._save_index()

        logger.info(f"Cleaned up {deleted_count} old versions")
        return deleted_count
    def undo(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Undo the last edit by restoring the previous version."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            logger.warning(f"No version history found for {file_path}")
            return None
        
        doc_history = self.index[doc_key]
        current_idx = doc_history["current_version"]
        
        if current_idx <= 0:
            logger.warning(f"Cannot undo - already at oldest version")
            return None
        
        # Move to previous version
        prev_idx = current_idx - 1
        prev_version = doc_history["versions"][prev_idx]
        
        # Restore the file
        version_file = prev_version["file_path"]
        if Path(version_file).exists():
            try:
                shutil.copy2(version_file, file_path)
                doc_history["current_version"] = prev_idx
                self._save_index()
                logger.info(f"Undid to version {prev_version['version_id']}")
                return prev_version
            except Exception as e:
                logger.error(f"Failed to undo: {e}")
                return None
        else:
            logger.error(f"Version file not found: {version_file}")
            return None

    def redo(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Redo the last undone edit."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            logger.warning(f"No version history found for {file_path}")
            return None
        
        doc_history = self.index[doc_key]
        current_idx = doc_history["current_version"]
        
        if current_idx >= len(doc_history["versions"]) - 1:
            logger.warning(f"Cannot redo - already at newest version")
            return None
        
        # Move to next version
        next_idx = current_idx + 1
        next_version = doc_history["versions"][next_idx]
        
        # Restore the file
        version_file = next_version["file_path"]
        if Path(version_file).exists():
            try:
                shutil.copy2(version_file, file_path)
                doc_history["current_version"] = next_idx
                self._save_index()
                logger.info(f"Redid to version {next_version['version_id']}")
                return next_version
            except Exception as e:
                logger.error(f"Failed to redo: {e}")
                return None
        else:
            logger.error(f"Version file not found: {version_file}")
            return None

    def can_undo(self, file_path: str) -> bool:
        """Check if undo is available."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return False
        
        return self.index[doc_key]["current_version"] > 0

    def can_redo(self, file_path: str) -> bool:
        """Check if redo is available."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return False
        
        doc_history = self.index[doc_key]
        return doc_history["current_version"] < len(doc_history["versions"]) - 1

    def get_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get the version history for a document."""
        return self.get_versions(file_path)
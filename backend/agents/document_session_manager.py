"""
Document Session Manager - Maintains stateful editing sessions for documents.

Each document gets a persistent session that tracks:
- Edit history (all actions performed)
- Document state snapshots
- Conversation context
- Current document structure
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class EditAction:
    """Represents a single edit action performed on a document."""
    timestamp: str
    action_type: str
    instruction: str  # Natural language instruction
    parameters: Dict[str, Any]
    result: str
    document_state_before: Dict[str, Any]  # Snapshot before edit
    document_state_after: Dict[str, Any]   # Snapshot after edit

@dataclass
class DocumentSession:
    """Represents an editing session for a document."""
    document_path: str
    document_name: str
    session_id: str
    created_at: str
    last_accessed: str
    edit_history: List[EditAction]
    conversation_context: List[Dict[str, str]]  # User messages and agent responses
    current_structure: Dict[str, Any]  # Current document structure
    metadata: Dict[str, Any]  # Additional metadata

class DocumentSessionManager:
    """
    Manages editing sessions for documents.
    Provides memory and context for document editing operations.
    """
    
    def __init__(self, sessions_dir: str = None):
        if sessions_dir is None:
            # Use path relative to this file's location
            base_dir = Path(__file__).parent.parent  # Go up to backend/
            sessions_dir = base_dir / "storage" / "document_sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, DocumentSession] = {}
    
    def _get_session_id(self, document_path: str, thread_id: str = None) -> str:
        """Generate a unique session ID for a document.
        
        Args:
            document_path: Path to the document
            thread_id: Optional conversation thread ID for isolation
                      If provided, prevents edit history mixing across conversations
        
        Returns:
            Unique session ID combining both document and thread context
        """
        # Combine document path with thread_id for better isolation
        # This ensures same doc uploaded to 2 different conversations get separate sessions
        if thread_id:
            combined = f"{thread_id}:{document_path}"
        else:
            combined = document_path
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.json"
    
    def get_or_create_session(self, document_path: str, document_name: str, thread_id: str = None) -> DocumentSession:
        """
        Get existing session or create a new one for a document.
        
        Args:
            document_path: Path to the document file
            document_name: Name of the document
            thread_id: Optional conversation thread ID for isolation across conversations
        
        Returns:
            DocumentSession object for this document in this conversation
        """
        session_id = self._get_session_id(document_path, thread_id)
        
        # Check if session is already loaded in memory
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.last_accessed = datetime.now().isoformat()
            return session
        
        # Try to load from disk
        session_file = self._get_session_file(session_id)
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Reconstruct EditAction objects
                edit_history = [
                    EditAction(**action) for action in data.get('edit_history', [])
                ]
                session = DocumentSession(
                    document_path=data['document_path'],
                    document_name=data['document_name'],
                    session_id=data['session_id'],
                    created_at=data['created_at'],
                    last_accessed=datetime.now().isoformat(),
                    edit_history=edit_history,
                    conversation_context=data.get('conversation_context', []),
                    current_structure=data.get('current_structure', {}),
                    metadata=data.get('metadata', {})
                )
                self._active_sessions[session_id] = session
                return session
        
        # Create new session
        session = DocumentSession(
            document_path=document_path,
            document_name=document_name,
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            edit_history=[],
            conversation_context=[],
            current_structure={},
            metadata={'thread_id': thread_id} if thread_id else {}
        )
        self._active_sessions[session_id] = session
        self._save_session(session)
        return session
    
    def add_edit_action(
        self,
        document_path: str,
        action_type: str,
        instruction: str,
        parameters: Dict[str, Any],
        result: str,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
        thread_id: str = None
    ):
        """
        Record an edit action in the session.
        
        Args:
            document_path: Path to the document
            action_type: Type of edit action
            instruction: Natural language instruction that was executed
            parameters: Parameters used for the action
            result: Result message from the action
            state_before: Document state before edit
            state_after: Document state after edit
            thread_id: Optional conversation thread ID for isolation
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        
        action = EditAction(
            timestamp=datetime.now().isoformat(),
            action_type=action_type,
            instruction=instruction,
            parameters=parameters,
            result=result,
            document_state_before=state_before,
            document_state_after=state_after
        )
        
        session.edit_history.append(action)
        session.current_structure = state_after
        session.last_accessed = datetime.now().isoformat()
        
        self._save_session(session)
    
    def add_conversation_turn(
        self,
        document_path: str,
        user_message: str,
        agent_response: str,
        thread_id: str = None
    ):
        """
        Add a conversation turn to the session.
        
        Args:
            document_path: Path to the document
            user_message: User's message
            agent_response: Agent's response
            thread_id: Optional conversation thread ID for isolation
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        
        session.conversation_context.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'agent': agent_response
        })
        
        session.last_accessed = datetime.now().isoformat()
        self._save_session(session)
    
    def get_session_context(self, document_path: str, thread_id: str = None) -> str:
        """
        Get formatted context for the document session.
        This includes edit history and conversation context.
        
        Args:
            document_path: Path to the document
            thread_id: Optional conversation thread ID for isolation
        
        Returns:
            Formatted string with session context
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        
        context = f"=== DOCUMENT EDITING SESSION ===\n"
        context += f"Document: {session.document_name}\n"
        context += f"Session started: {session.created_at}\n"
        context += f"Total edits: {len(session.edit_history)}\n\n"
        
        # Add recent conversation context (last 5 turns)
        if session.conversation_context:
            context += "=== RECENT CONVERSATION ===\n"
            for turn in session.conversation_context[-5:]:
                context += f"User: {turn['user']}\n"
                context += f"Agent: {turn['agent'][:100]}...\n\n"
        
        # Add recent edit history (last 5 edits)
        if session.edit_history:
            context += "=== RECENT EDITS ===\n"
            for i, action in enumerate(session.edit_history[-5:], 1):
                context += f"{i}. [{action.action_type}] {action.instruction}\n"
                context += f"   Result: {action.result[:100]}...\n"
                context += f"   Time: {action.timestamp}\n\n"
        
        # Add current document structure
        if session.current_structure:
            context += "=== CURRENT DOCUMENT STRUCTURE ===\n"
            context += json.dumps(session.current_structure, indent=2)
            context += "\n"
        
        return context
    
    def get_edit_history_summary(self, document_path: str, thread_id: str = None) -> List[Dict[str, Any]]:
        """
        Get a summary of all edits performed on the document.
        
        Args:
            document_path: Path to the document
            thread_id: Optional conversation thread ID for isolation
        
        Returns:
            List of edit action summaries
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        
        return [
            {
                'timestamp': action.timestamp,
                'action_type': action.action_type,
                'instruction': action.instruction,
                'result': action.result
            }
            for action in session.edit_history
        ]
    
    def can_undo(self, document_path: str, thread_id: str = None) -> bool:
        """
        Check if there are actions that can be undone.
        
        Args:
            document_path: Path to the document
            thread_id: Optional conversation thread ID for isolation
        
        Returns:
            True if undo is available
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        return len(session.edit_history) > 0
    
    def get_last_action(self, document_path: str, thread_id: str = None) -> Optional[EditAction]:
        """
        Get the last action performed on the document.
        
        Args:
            document_path: Path to the document
            thread_id: Optional conversation thread ID for isolation
        
        Returns:
            The last EditAction or None if no actions exist
        """
        session = self.get_or_create_session(document_path, Path(document_path).name, thread_id)
        return session.edit_history[-1] if session.edit_history else None
    
    def _save_session(self, session: DocumentSession):
        """Save session to disk."""
        session_file = self._get_session_file(session.session_id)
        
        # Convert to dict for JSON serialization
        data = {
            'document_path': session.document_path,
            'document_name': session.document_name,
            'session_id': session.session_id,
            'created_at': session.created_at,
            'last_accessed': session.last_accessed,
            'edit_history': [asdict(action) for action in session.edit_history],
            'conversation_context': session.conversation_context,
            'current_structure': session.current_structure,
            'metadata': session.metadata
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear_session(self, document_path: str):
        """Clear/reset a document session."""
        session_id = self._get_session_id(document_path)
        session_file = self._get_session_file(session_id)
        
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        if session_file.exists():
            session_file.unlink()

# Global session manager instance
session_manager = DocumentSessionManager()

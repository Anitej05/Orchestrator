"""
Spreadsheet Session Manager - Maintains stateful editing sessions for spreadsheets.

Similar to document_session_manager but for spreadsheet operations.
Tracks all pandas operations, queries, and transformations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class SpreadsheetOperation:
    """Represents a single operation performed on a spreadsheet."""
    timestamp: str
    operation_type: str  # query, transform, filter, aggregate, etc.
    instruction: str  # Natural language instruction
    pandas_code: str  # Actual pandas code executed
    result_summary: str  # Summary of the result
    rows_affected: int
    columns_affected: List[str]
    dataframe_state_before: Dict[str, Any]  # Shape, columns, dtypes
    dataframe_state_after: Dict[str, Any]

@dataclass
class SpreadsheetSession:
    """Represents an editing session for a spreadsheet."""
    file_id: str
    filename: str
    session_id: str
    created_at: str
    last_accessed: str
    operation_history: List[SpreadsheetOperation]
    conversation_context: List[Dict[str, str]]
    current_state: Dict[str, Any]  # Current dataframe state
    metadata: Dict[str, Any]

class SpreadsheetSessionManager:
    """
    Manages editing sessions for spreadsheets.
    Provides memory and context for spreadsheet operations.
    """
    
    def __init__(self, sessions_dir: str = None):
        if sessions_dir is None:
            base_dir = Path(__file__).parent.parent
            sessions_dir = base_dir / "storage" / "spreadsheet_sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, SpreadsheetSession] = {}
    
    def _get_session_id(self, file_id: str) -> str:
        """Generate a unique session ID for a spreadsheet."""
        return hashlib.md5(file_id.encode()).hexdigest()
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.json"
    
    def get_or_create_session(self, file_id: str, filename: str) -> SpreadsheetSession:
        """Get existing session or create a new one for a spreadsheet."""
        session_id = self._get_session_id(file_id)
        
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
                # Reconstruct SpreadsheetOperation objects
                operation_history = [
                    SpreadsheetOperation(**op) for op in data.get('operation_history', [])
                ]
                session = SpreadsheetSession(
                    file_id=data['file_id'],
                    filename=data['filename'],
                    session_id=data['session_id'],
                    created_at=data['created_at'],
                    last_accessed=datetime.now().isoformat(),
                    operation_history=operation_history,
                    conversation_context=data.get('conversation_context', []),
                    current_state=data.get('current_state', {}),
                    metadata=data.get('metadata', {})
                )
                self._active_sessions[session_id] = session
                return session
        
        # Create new session
        session = SpreadsheetSession(
            file_id=file_id,
            filename=filename,
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            operation_history=[],
            conversation_context=[],
            current_state={},
            metadata={}
        )
        self._active_sessions[session_id] = session
        self._save_session(session)
        return session
    
    def add_operation(
        self,
        file_id: str,
        operation_type: str,
        instruction: str,
        pandas_code: str,
        result_summary: str,
        rows_affected: int,
        columns_affected: List[str],
        state_before: Dict[str, Any],
        state_after: Dict[str, Any]
    ):
        """Record an operation in the session."""
        session = self.get_or_create_session(file_id, state_after.get('filename', 'unknown'))
        
        operation = SpreadsheetOperation(
            timestamp=datetime.now().isoformat(),
            operation_type=operation_type,
            instruction=instruction,
            pandas_code=pandas_code,
            result_summary=result_summary,
            rows_affected=rows_affected,
            columns_affected=columns_affected,
            dataframe_state_before=state_before,
            dataframe_state_after=state_after
        )
        
        session.operation_history.append(operation)
        session.current_state = state_after
        session.last_accessed = datetime.now().isoformat()
        
        self._save_session(session)
    
    def add_conversation_turn(
        self,
        file_id: str,
        user_message: str,
        agent_response: str
    ):
        """Add a conversation turn to the session."""
        session = self.get_or_create_session(file_id, "unknown")
        
        session.conversation_context.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'agent': agent_response
        })
        
        session.last_accessed = datetime.now().isoformat()
        self._save_session(session)
    
    def get_session_context(self, file_id: str) -> str:
        """
        Get formatted context for the spreadsheet session.
        Includes operation history and conversation context.
        """
        session = self.get_or_create_session(file_id, "unknown")
        
        context = f"=== SPREADSHEET EDITING SESSION ===\n"
        context += f"File: {session.filename}\n"
        context += f"Session started: {session.created_at}\n"
        context += f"Total operations: {len(session.operation_history)}\n\n"
        
        # Add recent conversation context (last 5 turns)
        if session.conversation_context:
            context += "=== RECENT CONVERSATION ===\n"
            for turn in session.conversation_context[-5:]:
                context += f"User: {turn['user']}\n"
                context += f"Agent: {turn['agent'][:100]}...\n\n"
        
        # Add recent operation history (last 5 operations)
        if session.operation_history:
            context += "=== RECENT OPERATIONS ===\n"
            for i, op in enumerate(session.operation_history[-5:], 1):
                context += f"{i}. [{op.operation_type}] {op.instruction}\n"
                context += f"   Code: {op.pandas_code[:80]}...\n"
                context += f"   Result: {op.result_summary[:100]}...\n"
                context += f"   Rows affected: {op.rows_affected}\n"
                context += f"   Time: {op.timestamp}\n\n"
        
        # Add current dataframe state
        if session.current_state:
            context += "=== CURRENT DATAFRAME STATE ===\n"
            context += json.dumps(session.current_state, indent=2)
            context += "\n"
        
        return context
    
    def get_operation_history(self, file_id: str) -> List[Dict[str, Any]]:
        """Get a summary of all operations performed on the spreadsheet."""
        session = self.get_or_create_session(file_id, "unknown")
        
        return [
            {
                'timestamp': op.timestamp,
                'operation_type': op.operation_type,
                'instruction': op.instruction,
                'pandas_code': op.pandas_code,
                'result_summary': op.result_summary,
                'rows_affected': op.rows_affected
            }
            for op in session.operation_history
        ]
    
    def _save_session(self, session: SpreadsheetSession):
        """Save session to disk."""
        session_file = self._get_session_file(session.session_id)
        
        # Convert to dict for JSON serialization
        data = {
            'file_id': session.file_id,
            'filename': session.filename,
            'session_id': session.session_id,
            'created_at': session.created_at,
            'last_accessed': session.last_accessed,
            'operation_history': [asdict(op) for op in session.operation_history],
            'conversation_context': session.conversation_context,
            'current_state': session.current_state,
            'metadata': session.metadata
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    def clear_session(self, file_id: str):
        """Clear/reset a spreadsheet session."""
        session_id = self._get_session_id(file_id)
        session_file = self._get_session_file(session_id)
        
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        if session_file.exists():
            session_file.unlink()

# Global session manager instance
spreadsheet_session_manager = SpreadsheetSessionManager()

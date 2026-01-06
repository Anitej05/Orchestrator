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
from threading import Lock

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
    result_summary: Any  # Summary of the result
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
            # Get workspace root (3 levels up: spreadsheet_session_manager.py -> agents -> backend -> root)
            workspace_root = Path(__file__).parent.parent.parent.resolve()
            sessions_dir = workspace_root / "storage" / "spreadsheet_sessions"
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions: Dict[str, SpreadsheetSession] = {}
        self._session_lock = Lock()  # Thread-safe concurrent access
    
    def _get_session_id(self, file_id: str, thread_id: str = None) -> str:
        """
        Generate a unique session ID for a spreadsheet.
        
        Args:
            file_id: The file identifier
            thread_id: Optional conversation thread ID for isolation across conversations
        
        Returns:
            Hash-based session ID that uniquely identifies this spreadsheet in this conversation
        """
        if thread_id:
            combined = f"{thread_id}:{file_id}"
        else:
            combined = file_id
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.json"
    
    def get_or_create_session(self, file_id: str, filename: str, thread_id: str = None) -> SpreadsheetSession:
        """
        Get existing session or create a new one for a spreadsheet.
        
        Args:
            file_id: The file identifier
            filename: The filename
            thread_id: Optional conversation thread ID for isolation across conversations
        
        Returns:
            SpreadsheetSession object for this spreadsheet in this conversation
        """
        with self._session_lock:  # LOCK: Protect session access
            session_id = self._get_session_id(file_id, thread_id)
            
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
                    # Reconstruct SpreadsheetOperation objects with defaults for missing fields
                    operation_history = [
                        self._deserialize_operation(op)
                        for op in data.get('operation_history', [])
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
                metadata={'thread_id': thread_id} if thread_id else {}
            )
            self._active_sessions[session_id] = session
            self._save_session(session)
            return session

    def _deserialize_operation(self, op_data: Dict[str, Any]) -> SpreadsheetOperation:
        """Safely deserialize an operation with sensible defaults."""
        defaults = {
            'timestamp': op_data.get('timestamp', datetime.now().isoformat()),
            'operation_type': op_data.get('operation_type', op_data.get('operation', 'unknown')),
            'instruction': op_data.get('instruction', op_data.get('description', '')),
            'pandas_code': op_data.get('pandas_code', ''),
            'result_summary': op_data.get('result_summary', ''),
            'rows_affected': op_data.get('rows_affected', 0),
            'columns_affected': op_data.get('columns_affected', []),
            'dataframe_state_before': op_data.get('dataframe_state_before', {}),
            'dataframe_state_after': op_data.get('dataframe_state_after', {}),
        }
        return SpreadsheetOperation(**defaults)
    
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
        state_after: Dict[str, Any],
        thread_id: str = None
    ):
        """
        Record an operation in the session.
        
        Args:
            file_id: The file identifier
            operation_type: Type of operation (query, transform, filter, etc.)
            instruction: Natural language instruction
            pandas_code: Actual pandas code executed
            result_summary: Summary of the result
            rows_affected: Number of rows affected
            columns_affected: List of columns affected
            state_before: Dataframe state before operation
            state_after: Dataframe state after operation
            thread_id: Optional conversation thread ID
        """
        session = self.get_or_create_session(file_id, state_after.get('filename', 'unknown'), thread_id)
        
        with self._session_lock:  # LOCK: Protect session modifications
            operation = SpreadsheetOperation(
                timestamp=datetime.now().isoformat(),
                operation_type=operation_type,
                instruction=instruction,
                pandas_code=pandas_code,
                result_summary=self._format_result_summary(result_summary),
                rows_affected=rows_affected,
                columns_affected=columns_affected,
                dataframe_state_before=state_before,
                dataframe_state_after=state_after
            )
            
            session.operation_history.append(operation)
            self._trim_operation_history(session)
            session.current_state = state_after
            session.last_accessed = datetime.now().isoformat()
            
            self._save_session(session)

    def track_operation(
        self,
        file_id: str,
        thread_id: Optional[str] = None,
        operation: str = "unknown",
        description: str = "",
        result_summary: Any = None
    ) -> None:
        """Track a lightweight operation (used by /nl_query and /transform)."""
        session = self.get_or_create_session(file_id, filename="unknown", thread_id=thread_id)

        with self._session_lock:
            operation_entry = SpreadsheetOperation(
                timestamp=datetime.now().isoformat(),
                operation_type=operation,
                instruction=description,
                pandas_code="",
                result_summary=self._format_result_summary(result_summary),
                rows_affected=0,
                columns_affected=[],
                dataframe_state_before=dict(session.current_state) if session.current_state else {},
                dataframe_state_after=dict(session.current_state) if session.current_state else {}
            )

            session.operation_history.append(operation_entry)
            self._trim_operation_history(session)
            session.last_accessed = datetime.now().isoformat()
            self._save_session(session)

    def get_session_history(self, file_id: str, thread_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent operations for a spreadsheet/thread combination."""
        session = self.get_or_create_session(file_id, filename="unknown", thread_id=thread_id)

        with self._session_lock:
            history = session.operation_history[-limit:]
            return [
                {
                    'timestamp': op.timestamp,
                    'operation': op.operation_type,
                    'description': op.instruction,
                    'result_summary': op.result_summary
                }
                for op in history
            ]
    
    def add_conversation_turn(
        self,
        file_id: str,
        user_message: str,
        agent_response: str,
        thread_id: str = None
    ):
        """
        Add a conversation turn to the session.
        
        Args:
            file_id: The file identifier
            user_message: User's message
            agent_response: Agent's response
            thread_id: Optional conversation thread ID
        """
        session = self.get_or_create_session(file_id, "unknown", thread_id)
        
        session.conversation_context.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'agent': agent_response
        })
        
        session.last_accessed = datetime.now().isoformat()
        self._save_session(session)
    
    def get_session_context(self, file_id: str, thread_id: str = None) -> str:
        """
        Get formatted context for the spreadsheet session.
        Includes operation history and conversation context.
        
        Args:
            file_id: The file identifier
            thread_id: Optional conversation thread ID
        
        Returns:
            Formatted string with session context
        """
        session = self.get_or_create_session(file_id, "unknown", thread_id)
        
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
    
    def get_operation_history(self, file_id: str, thread_id: str = None) -> List[Dict[str, Any]]:
        """
        Get a summary of all operations performed on the spreadsheet.
        
        Args:
            file_id: The file identifier
            thread_id: Optional conversation thread ID
        
        Returns:
            List of operation summaries
        """
        session = self.get_or_create_session(file_id, "unknown", thread_id)
        
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
    
    def get_relevant_context(self, file_id: str, current_instruction: str,
                            thread_id: str = None, max_tokens: int = 2000) -> str:
        """
        Get context most relevant to the current instruction.
        Uses smart truncation to stay within token limits.
        
        Args:
            file_id: The file identifier
            current_instruction: The current user instruction
            thread_id: Optional conversation thread ID
            max_tokens: Maximum estimated tokens for context
        
        Returns:
            Formatted context string optimized for LLM consumption
        """
        session = self.get_or_create_session(file_id, "unknown", thread_id)
        
        context_parts = []
        estimated_tokens = 0
        
        # Header
        header = f"=== AGENT MEMORY: {session.filename} ===\n"
        header += f"Session: {len(session.operation_history)} operations performed\n\n"
        context_parts.append(header)
        estimated_tokens += self._estimate_tokens(header)
        
        # Recent operations (last 5 with full details)
        if session.operation_history:
            recent_section = "RECENT OPERATIONS:\n"
            for i, op in enumerate(session.operation_history[-5:], 1):
                recent_section += f"{i}. [{op.operation_type}] {op.instruction}\n"
                recent_section += f"   Code: {op.pandas_code[:60]}...\n"
                recent_section += f"   → {op.result_summary[:80]}{'...' if len(op.result_summary) > 80 else ''}\n"
            recent_section += "\n"
            
            if estimated_tokens + self._estimate_tokens(recent_section) < max_tokens:
                context_parts.append(recent_section)
                estimated_tokens += self._estimate_tokens(recent_section)
        
        # Older operations (summarized, if space allows)
        if len(session.operation_history) > 5 and estimated_tokens < max_tokens * 0.7:
            older_section = "EARLIER OPERATIONS (summarized):\n"
            for op in session.operation_history[-20:-5]:
                summary = f"- {op.operation_type}: {op.instruction[:50]}...\n"
                older_section += summary
            older_section += "\n"
            
            if estimated_tokens + self._estimate_tokens(older_section) < max_tokens:
                context_parts.append(older_section)
                estimated_tokens += self._estimate_tokens(older_section)
        
        # Current dataframe state (if space allows)
        if session.current_state and estimated_tokens < max_tokens * 0.85:
            state_section = "CURRENT DATAFRAME STATE:\n"
            state = session.current_state
            state_section += f"Shape: {state.get('shape', 'unknown')}\n"
            state_section += f"Columns: {state.get('columns', [])}\n"
            
            if estimated_tokens + self._estimate_tokens(state_section) < max_tokens:
                context_parts.append(state_section)
        
        return "".join(context_parts)

    def _trim_operation_history(self, session: SpreadsheetSession, max_entries: int = 200) -> None:
        """Keep operation history bounded to avoid unbounded growth."""
        if len(session.operation_history) > max_entries:
            session.operation_history = session.operation_history[-max_entries:]

    def _format_result_summary(self, summary: Any) -> str:
        """Normalize result summaries to safe, readable strings for logging and display."""
        if summary is None:
            return ""
        if isinstance(summary, str):
            return summary
        try:
            return json.dumps(summary, cls=NumpyEncoder)
        except Exception:
            return str(summary)
    
    def get_all_files_summary(self, thread_id: str) -> str:
        """
        Get a summary of all spreadsheets worked on in this thread.
        Useful for cross-file awareness.
        
        Args:
            thread_id: The conversation thread ID
        
        Returns:
            Summary of all spreadsheets and what was done to them
        """
        summaries = []
        
        # Find all sessions for this thread
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Check if this session belongs to this thread
                if data.get('metadata', {}).get('thread_id') == thread_id:
                    filename = data.get('filename', 'Unknown')
                    op_count = len(data.get('operation_history', []))
                    
                    if op_count > 0:
                        last_op = data['operation_history'][-1]
                        summary = f"- {filename}: {op_count} ops, last: {last_op.get('operation_type', 'unknown')}"
                        summaries.append(summary)
            except Exception:
                continue
        
        if summaries:
            return "SPREADSHEETS IN THIS CONVERSATION:\n" + "\n".join(summaries)
        return ""
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return len(text) // 4
    
    def clear_session(self, file_id: str, thread_id: str = None):
        """
        Clear/reset a spreadsheet session.
        
        Args:
            file_id: The file identifier
            thread_id: Optional conversation thread ID
        """
        session_id = self._get_session_id(file_id, thread_id)
        session_file = self._get_session_file(session_id)
        
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        
        if session_file.exists():
            session_file.unlink()


# Global session manager instance
spreadsheet_session_manager = SpreadsheetSessionManager()

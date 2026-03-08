"""
Conversations Router - Handles conversation management endpoints.

Extracted from main.py to improve code organization and maintainability.
Includes: status, history, list all, clear, debug.
"""

import os
import json
import logging
from threading import Lock
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database import get_db
from models import UserThread, ConversationSearch

router = APIRouter(tags=["Conversations"])
logger = logging.getLogger("uvicorn.error")

# Use absolute path so this works regardless of the server's CWD
_ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))      # backend/routers/
_BACKEND_DIR = os.path.dirname(_ROUTER_DIR)                    # backend/
CONVERSATION_HISTORY_DIR = os.path.join(_BACKEND_DIR, "conversation_history")

conversation_store: Dict[str, Dict[str, Any]] = {}
store_lock = Lock()


def set_shared_state(conv_store: Dict, lock: Lock):
    global conversation_store, store_lock
    conversation_store = conv_store
    store_lock = lock


# --- Models ---
class ConversationStatus(BaseModel):
    """Model for conversation status responses"""
    thread_id: str
    status: str
    question_for_user: Optional[str] = None
    final_response: Optional[str] = None
    task_agent_pairs: Optional[List[Dict]] = None


@router.get("/api/chat/status/{thread_id}", response_model=ConversationStatus)
async def get_conversation_status(thread_id: str):
    """Get the current status of a conversation thread."""
    try:
        with store_lock:
            state_data = conversation_store.get(thread_id)

        if not state_data:
            raise HTTPException(status_code=404, detail="Conversation thread not found")

        if state_data.get("pending_user_input"):
            status = "pending_user_input"
        elif state_data.get("final_response"):
            status = "completed"
        else:
            status = "processing"

        return ConversationStatus(
            thread_id=thread_id,
            status=status,
            question_for_user=state_data.get("question_for_user"),
            final_response=state_data.get("final_response"),
            task_agent_pairs=state_data.get("task_agent_pairs", [])
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error getting conversation status for thread_id {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@router.get("/api/chat/history/{thread_id}")
async def get_conversation_history_simple(thread_id: str):
    """
    Load the full conversation history from the saved JSON file.
    Returns all messages, metadata, plan, and uploaded files.
    Fallback to in-memory store if JSON not yet persisted.
    """
    try:
        from main import conversation_store
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_dir = os.path.join(backend_dir, "conversation_history")
        history_path = os.path.join(history_dir, f"{thread_id}.json")
        
        logger.info(f"Looking for conversation history at: {history_path}")
        
        # Try to load from JSON file first
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            logger.info(f"✅ Successfully loaded conversation {thread_id} from JSON file")
            return conversation_data
        
        # Fallback: Check in-memory store
        if thread_id in conversation_store:
            logger.info(f"⚡ Loaded conversation {thread_id} from in-memory store (not persisted to JSON yet)")
            return conversation_store[thread_id]
        
        logger.warning(f"❌ Conversation history not found for thread_id: {thread_id}")
        logger.warning(f"   → Checked JSON: {history_path} - NOT FOUND")
        logger.warning(f"   → Checked in-memory store - NOT FOUND")
        raise HTTPException(status_code=404, detail=f"Conversation history not found for thread_id: {thread_id}")
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error loading conversation history for thread_id {thread_id}: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=f"Failed to load conversation history: {str(e)}")


@router.delete("/api/chat/{thread_id}")
async def clear_conversation(thread_id: str):
    """Clear a conversation thread from memory."""
    try:
        with store_lock:
            if thread_id in conversation_store:
                del conversation_store[thread_id]
                logger.info(f"Cleared conversation for thread_id: {thread_id}")
                return {"message": f"Conversation {thread_id} cleared successfully"}
            else:
                raise HTTPException(status_code=404, detail="Conversation thread not found")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error clearing conversation for thread_id {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@router.get("/api/chat/debug/conversations")
async def debug_conversations():
    """Debug endpoint to see all active conversations (remove in production)."""
    try:
        with store_lock:
            conversations = {}
            for thread_id, state in conversation_store.items():
                conversations[thread_id] = {
                    "pending_user_input": state.get("pending_user_input", False),
                    "question_for_user": state.get("question_for_user"),
                    "has_final_response": bool(state.get("final_response")),
                    "parsed_tasks_count": len(state.get("parsed_tasks", [])),
                    "original_prompt": state.get("original_prompt", "")[:100] + "..." if state.get("original_prompt", "") else ""
                }
        return {"active_conversations": conversations}

    except Exception as e:
        logger.error(f"Error getting debug conversations: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


def _extract_last_message(history_path: str) -> Optional[str]:
    """Read the last non-empty message content from a conversation JSON file."""
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        messages = history_data.get("messages", [])
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content") or (msg.get("data") or {}).get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()[:100]
    except Exception:
        pass
    return None


def _build_conversation_entry_from_file(thread_id: str, history_path: str) -> dict:
    """Build a conversation list entry from a JSON history file."""
    title = "Untitled Conversation"
    created_at = None
    updated_at = None
    last_message = None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_title = data.get("original_prompt") or data.get("title") or ""
        if raw_title.strip():
            title = raw_title.strip()[:100]
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        # Use file mtime if no timestamp in file
        if not updated_at:
            mtime = os.path.getmtime(history_path)
            import datetime
            updated_at = datetime.datetime.utcfromtimestamp(mtime).isoformat() + "Z"
        messages = data.get("messages", [])
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content") or (msg.get("data") or {}).get("content", "")
            if isinstance(content, str) and content.strip():
                last_message = content.strip()[:100]
                break
    except Exception as e:
        logger.warning(f"Could not parse history file for {thread_id}: {e}")
    return {
        "id": thread_id,
        "thread_id": thread_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "last_message": last_message,
    }


@router.get("/api/conversations")
async def get_all_conversations(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves a list of conversations for the authenticated user.
    Falls back to scanning conversation_history/*.json files when the DB has no
    records (e.g. dev mode without Clerk, or first-run after migrating).
    """
    user_id = None
    try:
        from auth import get_user_from_request
        user = get_user_from_request(request)
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        is_dev_mode = user.get("dev_mode", False)
    except HTTPException as auth_err:
        # Auth failed (missing/invalid token, Clerk misconfigured, etc.)
        # Fall through to JSON file scan so history is always visible in dev.
        logger.warning(f"Auth failed for /api/conversations ({auth_err.status_code}): {auth_err.detail}. Falling back to JSON scan.")
        is_dev_mode = True

    try:
        conversations = []

        if user_id:
            logger.info(f"Fetching conversations for user: {user_id} (dev_mode={is_dev_mode})")
            user_threads = db.query(UserThread).filter_by(user_id=user_id).order_by(
                UserThread.updated_at.desc()
            ).all()
            logger.info(f"Found {len(user_threads)} DB conversations for user {user_id}")

            if user_threads:
                for ut in user_threads:
                    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{ut.thread_id}.json")
                    last_message = _extract_last_message(history_path) if os.path.exists(history_path) else None
                    title = ut.title if (ut.title and ut.title.strip() and ut.title != "None") else "Untitled Conversation"
                    conversations.append({
                        "id": ut.thread_id,
                        "thread_id": ut.thread_id,
                        "title": title,
                        "created_at": ut.created_at.isoformat() + "Z" if ut.created_at else None,
                        "updated_at": ut.updated_at.isoformat() + "Z" if ut.updated_at else None,
                        "last_message": last_message,
                    })
                return conversations

        # No user_id (auth failed) OR DB returned no records — scan JSON files
        logger.info(f"Scanning {CONVERSATION_HISTORY_DIR} for conversation JSON files")
        if os.path.isdir(CONVERSATION_HISTORY_DIR):
            import glob as _glob
            json_files = sorted(
                _glob.glob(os.path.join(CONVERSATION_HISTORY_DIR, "*.json")),
                key=os.path.getmtime,
                reverse=True,
            )
            for fpath in json_files:
                tid = os.path.splitext(os.path.basename(fpath))[0]
                conversations.append(_build_conversation_entry_from_file(tid, fpath))

        return conversations

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.get("/api/conversations/{thread_id}")
async def get_conversation_history_auth(thread_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Retrieves the full, standardized conversation state from its JSON file.
    This is the single source of truth for a conversation's history.
    Ensures user can only access their own conversations.
    
    Fallback to in-memory store if JSON file doesn't exist yet.
    """
    try:
        from auth import get_user_from_request
        from main import conversation_store
        user_id = None
        try:
            user = get_user_from_request(request)
            user_id = user.get("sub") or user.get("user_id") or user.get("id")
            is_dev_mode = user.get("dev_mode", False)
        except HTTPException as auth_err:
            logger.warning(f"Auth failed for /api/conversations/{thread_id} ({auth_err.status_code}). Serving from file if available.")
            is_dev_mode = True

        # Ownership check — skip in dev/auth-failed mode
        if user_id and not is_dev_mode:
            logger.info(f"Checking ownership: thread_id={thread_id}, user_id={user_id}")
            user_thread = db.query(UserThread).filter_by(
                thread_id=thread_id,
                user_id=user_id
            ).first()
            if not user_thread:
                logger.warning(f"User {user_id} attempted to access thread {thread_id} they don't own")
                raise HTTPException(status_code=403, detail="You don't have permission to access this conversation")
        
        history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        
        # Try to load from JSON file first (preferred)
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            logger.info(f"✅ User {user_id} loaded conversation {thread_id} from JSON file (persistent storage)")
            return history_data
        
        # Fallback: Check if conversation is in memory (for recently created conversations)
        if thread_id in conversation_store:
            memory_state = conversation_store[thread_id]
            logger.info(f"⚡ Loaded conversation {thread_id} from in-memory store (not persisted to JSON yet)")

            # Auto-persist memory state so future loads come from durable storage
            try:
                from backend.orchestrator.graph import save_conversation_history

                if isinstance(memory_state, dict):
                    state_to_save = dict(memory_state)
                    state_to_save["thread_id"] = thread_id
                    state_to_save["owner"] = {"user_id": user_id}
                    save_conversation_history(
                        state_to_save,
                        {"configurable": {"thread_id": thread_id, "owner": {"user_id": user_id}}},
                    )
                    logger.info(f"✅ Auto-persisted in-memory conversation {thread_id} to JSON during load")
                else:
                    logger.warning(f"Could not auto-persist thread {thread_id}: state is not a dict")
            except Exception as persist_err:
                logger.warning(f"Auto-persist skipped for {thread_id}: {persist_err}")

            return memory_state
        
        # If neither JSON nor in-memory, try to recover from database
        logger.warning(f"⚠️  Conversation {thread_id} not found in JSON or memory")
        logger.warning(f"   → JSON file missing: {history_path}")
        
        # Try to recover messages from ConversationSearch table
        search_records = db.query(ConversationSearch).filter_by(
            thread_id=thread_id,
            user_id=user_id
        ).order_by(ConversationSearch.message_index).all()
        
        if search_records:
            logger.info(f"✅ Found {len(search_records)} messages in ConversationSearch table for {thread_id}")
            # Reconstruct messages from database
            recovered_messages = []
            for record in search_records:
                recovered_messages.append({
                    "role": record.message_role or "assistant",
                    "content": record.message_content,
                    "timestamp": record.message_timestamp.isoformat() if record.message_timestamp else None
                })
            return {
                "thread_id": thread_id,
                "title": user_thread.title or "Untitled Conversation",
                "status": "recovered_from_database",
                "messages": recovered_messages,
                "created_at": user_thread.created_at.isoformat() if user_thread.created_at else None,
                "updated_at": user_thread.updated_at.isoformat() if user_thread.updated_at else None,
                "note": "Conversation history recovered from database. It may be incomplete."
            }
        
        # No recovery possible - return minimal state
        logger.warning(f"   → No messages found in database either")
        logger.warning(f"   → Returning minimal state with title: '{user_thread.title}'")
        logger.warning(f"   → This conversation was likely created before save bugs were fixed")
        return {
            "thread_id": thread_id,
            "title": user_thread.title or "Untitled Conversation",
            "status": "empty",
            "messages": [],
            "created_at": user_thread.created_at.isoformat() if user_thread.created_at else None,
            "updated_at": user_thread.updated_at.isoformat() if user_thread.updated_at else None,
            "note": "This conversation was created before recent fixes. The detailed history was never saved and cannot be recovered. Start a new conversation to use the fixed save system."
        }
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conversation history for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse conversation history file.")
    except Exception as e:
        logger.error(f"Error loading conversation history for {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while loading the conversation: {str(e)}")

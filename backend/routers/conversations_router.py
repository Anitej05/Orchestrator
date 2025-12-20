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

from database import SessionLocal
from models import UserThread

router = APIRouter(tags=["Conversations"])
logger = logging.getLogger("uvicorn.error")

CONVERSATION_HISTORY_DIR = "conversation_history"

# --- Shared State (will be passed from main.py) ---
# These will be injected via dependency injection or module-level assignment
conversation_store: Dict[str, Dict[str, Any]] = {}
store_lock = Lock()


def set_shared_state(conv_store: Dict, lock: Lock):
    """Called by main.py to inject shared state."""
    global conversation_store, store_lock
    conversation_store = conv_store
    store_lock = lock


# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    """
    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        history_dir = os.path.join(backend_dir, "conversation_history")
        history_path = os.path.join(history_dir, f"{thread_id}.json")
        
        logger.info(f"Looking for conversation history at: {history_path}")
        
        if not os.path.exists(history_path):
            logger.warning(f"Conversation history not found at: {history_path}")
            raise HTTPException(status_code=404, detail=f"Conversation history not found for thread_id: {thread_id}")
        
        with open(history_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        logger.info(f"Successfully loaded conversation history for thread_id: {thread_id}")
        return conversation_data
        
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


@router.get("/api/conversations")
async def get_all_conversations(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves a list of conversations for the authenticated user.
    Returns conversation objects with metadata (id, title, created_at, last_message).
    """
    try:
        from auth import get_user_from_request
        user = get_user_from_request(request)
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Unable to determine user identity")
        
        logger.info(f"Fetching conversations for user: {user_id}")
        
        user_threads = db.query(UserThread).filter_by(user_id=user_id).order_by(
            UserThread.updated_at.desc()
        ).all()
        
        logger.info(f"Found {len(user_threads)} conversations for user {user_id}")
        
        conversations = []
        for ut in user_threads:
            history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{ut.thread_id}.json")
            last_message = None
            
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                        messages = history_data.get("messages", [])
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict):
                                last_message = last_msg.get("content", "")[:100]
                except Exception as e:
                    logger.warning(f"Failed to read history for {ut.thread_id}: {e}")
            
            title = ut.title
            if not title or title == "None" or title.strip() == "":
                title = "Untitled Conversation"
            
            conversations.append({
                "id": ut.thread_id,
                "thread_id": ut.thread_id,
                "title": title,
                "created_at": ut.created_at.isoformat() if ut.created_at else None,
                "updated_at": ut.updated_at.isoformat() if ut.updated_at else None,
                "last_message": last_message
            })
        
        return conversations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.get("/api/conversations/{thread_id}")
async def get_conversation_history_auth(thread_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Retrieves the full, standardized conversation state from its JSON file.
    This is the single source of truth for a conversation's history.
    Ensures user can only access their own conversations.
    """
    try:
        from auth import get_user_from_request
        user = get_user_from_request(request)
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Unable to determine user identity")
        
        logger.info(f"Checking ownership: looking for thread_id={thread_id}, user_id={user_id}")
        user_thread = db.query(UserThread).filter_by(
            thread_id=thread_id,
            user_id=user_id
        ).first()
        
        if not user_thread:
            logger.warning(f"User {user_id} attempted to access thread {thread_id} they don't own")
            all_user_threads = db.query(UserThread).filter_by(user_id=user_id).all()
            logger.debug(f"User {user_id} owns {len(all_user_threads)} threads: {[t.thread_id for t in all_user_threads]}")
            thread_for_any_user = db.query(UserThread).filter_by(thread_id=thread_id).first()
            if thread_for_any_user:
                logger.debug(f"Thread {thread_id} exists and belongs to user: {thread_for_any_user.user_id}")
            raise HTTPException(status_code=403, detail="You don't have permission to access this conversation")
        
        history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        
        if not os.path.exists(history_path):
            raise HTTPException(status_code=404, detail="Conversation history not found.")
            
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        
        logger.info(f"User {user_id} successfully accessed conversation {thread_id}")
        return history_data
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conversation history for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse conversation history file.")
    except Exception as e:
        logger.error(f"Error loading conversation history for {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while loading the conversation: {str(e)}")

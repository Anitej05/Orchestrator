"""
Orchestrator Utilities

Shared utility functions used across multiple node modules.
"""

import os
import re
import json
import logging
from typing import Any, Dict, List
from pydantic.networks import HttpUrl
from langchain_core.messages import messages_to_dict

logger = logging.getLogger("AgentOrchestrator")

# Directory paths
ORCHESTRATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(ORCHESTRATOR_DIR)

# Global embeddings cache (lazy-loaded)
_hf_embeddings = None
_embedding_model = None

from datetime import datetime

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for HttpUrl, datetime, and other special types."""
    def default(self, o):
        if isinstance(o, HttpUrl):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def extract_json_from_response(text: str) -> str | None:
    """
    A robust function to extract a JSON object from a string that may contain
    markdown, <think> blocks, and other conversational text.
    """
    if not isinstance(text, str):
        return None

    # 1. Try to find a JSON object embedded in a markdown code block.
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. Strip any <think> blocks and try to find valid JSON.
    text_no_thinking = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 3. Find the first '{' and the last '}'.
    start = text_no_thinking.find('{')
    end = text_no_thinking.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text_no_thinking[start:end+1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            pass

    return None


def serialize_complex_object(obj):
    """Helper function to serialize complex objects consistently."""
    # First, handle common complex types directly
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if type(obj).__name__ == 'HttpUrl' or (hasattr(obj, '__class__') and 'HttpUrl' in str(type(obj))):
        return str(obj)

    try:
        # For Pydantic models, use model_dump or dict
        if hasattr(obj, 'model_dump'):
            data = obj.model_dump(mode='json')
            return serialize_complex_object(data) # Recurse to normalize contents
        elif hasattr(obj, 'dict'):
            data = obj.dict()
            return serialize_complex_object(data)
            
        # For LangChain messages
        if hasattr(obj, 'type') or hasattr(obj, '_type'):
            try:
                serialized_list = messages_to_dict([obj])
                if serialized_list:
                    d = serialized_list[0]
                    # NORMALIZE: Map 'ai' -> 'assistant' and 'human' -> 'user' for frontend
                    if d.get('type') == 'ai':
                        d['type'] = 'assistant'
                    elif d.get('type') == 'human':
                        d['type'] = 'user'
                    # FLATTEN: Unwrap 'data' if present
                    if 'data' in d and isinstance(d['data'], dict):
                        data_content = d.pop('data')
                        d.update(data_content)
                    return d
            except:
                pass

        # For collections, recurse
        if isinstance(obj, (list, tuple)):
            # Check for message list
            is_message_list = obj and all(
                (hasattr(item, 'type') or hasattr(item, '_type')) 
                for item in obj if item is not None
            )
            if is_message_list:
                msgs = messages_to_dict(obj)
                for m in msgs:
                    if m.get('type') == 'ai':
                        m['type'] = 'assistant'
                    elif m.get('type') == 'human':
                        m['type'] = 'user'
                return msgs
            return [serialize_complex_object(item) for item in obj]
            
        if isinstance(obj, dict):
            return {str(k): serialize_complex_object(v) for k, v in obj.items()}

        # Final fallback: try standard JSON path via encoder
        return json.loads(json.dumps(obj, cls=CustomJSONEncoder))
    except Exception as e:
        logger.warning(f"Complex serialization fallback for {type(obj)}: {e}")
        return str(obj)


def transform_payload_types(payload: Dict[str, Any], parameters: List[Any]) -> Dict[str, Any]:
    """
    Transform payload parameter types to match the endpoint schema.
    Fixes common issues like:
    - String values that should be arrays
    - Missing optional parameters with defaults
    """
    transformed = payload.copy()
    
    for param in parameters:
        param_name = param.name
        param_type = param.param_type
        
        if param_name not in transformed:
            continue
        
        value = transformed[param_name]
        
        if param_type == "array" and isinstance(value, str):
            logger.info(f"Transforming parameter '{param_name}' from string to array")
            transformed[param_name] = [value]
        elif param_type == "array" and not isinstance(value, list):
            logger.info(f"Transforming parameter '{param_name}' to array")
            transformed[param_name] = [value]
        elif param_type == "integer" and isinstance(value, str):
            try:
                transformed[param_name] = int(value)
            except ValueError:
                logger.warning(f"Could not convert '{param_name}' value '{value}' to integer")
    
    return transformed


def get_hf_embeddings():
    """Lazily load HuggingFace embeddings to avoid import-time issues."""
    global _hf_embeddings, _embedding_model
    if _hf_embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
        _hf_embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    return _hf_embeddings


def save_conversation_history(state: dict, config=None, *args, **kwargs):
    """Saves the conversation history to a JSON file. Accepts extra args for compatibility."""
    thread_id = state.get("thread_id")

    # Fallback: extract thread_id from LangGraph-style config payload
    if not thread_id:
        config_payload = None
        if args and isinstance(args[0], dict):
            config_payload = args[0]
        elif isinstance(kwargs.get("config"), dict):
            config_payload = kwargs.get("config")
        elif isinstance(kwargs.get("configurable"), dict):
            config_payload = {"configurable": kwargs.get("configurable")}

        if isinstance(config_payload, dict):
            thread_id = (
                config_payload.get("thread_id")
                or config_payload.get("configurable", {}).get("thread_id")
            )

        # Keep state consistent for downstream serialization
        if thread_id and isinstance(state, dict):
            state["thread_id"] = thread_id

    if not thread_id:
        thread_id = state.get("thread_id")
    if not thread_id:
        logger.warning("No thread_id found in state or config, skipping history save")
        return

    # Validate thread_id format
    import re
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if not re.match(uuid_pattern, str(thread_id), re.IGNORECASE):
        logger.error(f"Invalid thread_id format: '{thread_id}' (length: {len(str(thread_id))})")
        return

    # Use BACKEND_DIR/conversation_history — the canonical location read by all API endpoints
    history_dir = os.path.join(BACKEND_DIR, "conversation_history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"{thread_id}.json")
    
    logger.info(f"Attempting to save conversation history: thread_id={thread_id}, path={history_path}")
    
    try:
        # Extract messages using LangChain utility
        messages = state.get("messages", [])
        # Handle case where messages are already serialized or not a list
        if not isinstance(messages, list):
             messages = []
        
        # Serialize messages if they are objects
        try:
            serialized_messages = messages_to_dict(messages)
        except:
            # Fallback for already serialized or mixed content
            serialized_messages = [serialize_complex_object(m) for m in messages]
        
        # Determine overall status
        status = "completed"
        if state.get("pending_user_input"):
            status = "pending_user_input"
        elif state.get("pending_action_approval") or state.get("pending_approval"):
            status = "pending_approval"

        data = {
            "thread_id": thread_id,
            "status": status,
            "original_prompt": state.get("original_prompt"),
            "messages": serialized_messages,
            # Final / response fields
            "final_response": state.get("final_response"),
            "task_agent_pairs": serialize_complex_object(state.get("task_agent_pairs", [])),
            "uploaded_files": serialize_complex_object(state.get("uploaded_files", [])),
            # Plan / task fields
            "task_plan": serialize_complex_object(state.get("task_plan", [])),
            "plan": serialize_complex_object(state.get("task_plan", [])),  # alias for frontend
            "todo_list": serialize_complex_object(state.get("todo_list", [])),
            "execution_plan": serialize_complex_object(state.get("execution_plan")),
            "task_statuses": serialize_complex_object(state.get("task_statuses", {})),
            # Canvas fields
            "has_canvas": state.get("has_canvas", False),
            "canvas_type": state.get("canvas_type"),
            "canvas_content": state.get("canvas_content"),
            "canvas_data": serialize_complex_object(state.get("canvas_data")),
            "canvas_title": state.get("canvas_title"),
            "canvas_registry": serialize_complex_object(state.get("canvas_registry")),
            "active_canvas_id": state.get("active_canvas_id"),
            # Pending input / approval fields
            "pending_user_input": state.get("pending_user_input", False),
            "question_for_user": state.get("question_for_user"),
            "pending_action_approval": state.get("pending_action_approval", False),
            "pending_action": serialize_complex_object(state.get("pending_action")),
            # Internal/memory fields
            "action_history": serialize_complex_object(state.get("action_history", [])),
            "insights": serialize_complex_object(state.get("insights", {})),
            "memory": serialize_complex_object(state.get("memory", {})),
            # For backward compatibility 
            "metadata": {},
        }
        
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)
            
        logger.info(f"✅ Conversation history successfully saved: {history_path} (size: {os.path.getsize(history_path)} bytes)")
        return history_path
    except Exception as e:
        logger.error(f"❌ Failed to save conversation history for thread_id={thread_id}: {type(e).__name__}: {e}", exc_info=True)
        return None


def get_serializable_state(state: dict, *args, **kwargs) -> dict:
    """Returns a JSON-serializable version of the state. Accepts extra args for compatibility."""
    return serialize_complex_object(state)

"""
Orchestrator Utilities

Shared utility functions used across multiple node modules.
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from pydantic.networks import HttpUrl
from langchain_core.messages import messages_to_dict
from langchain_cerebras import ChatCerebras
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq

logger = logging.getLogger("AgentOrchestrator")

# Directory paths
ORCHESTRATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(ORCHESTRATOR_DIR)
PLAN_DIR = os.path.join(BACKEND_DIR, "agent_plans")
os.makedirs(PLAN_DIR, exist_ok=True)


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


def save_plan_to_file(state: dict, *args, **kwargs):
    """Saves the current plan and completed tasks to a Markdown file. Accepts extra args for compatibility."""
    thread_id = state.get("thread_id")
    if not thread_id:
        logger.warning("No thread_id found in state, skipping plan save")
        return {}

    plan_path = os.path.join(PLAN_DIR, f"{thread_id}-plan.md")

    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(f"# Execution Plan for Thread: {thread_id}\n\n")
        f.write(f"**Original Prompt:** {state.get('original_prompt', 'N/A')}\n\n")

        f.write("## Attachments\n")
        if uploaded_files := state.get("uploaded_files"):
            for file_obj in uploaded_files:
                if isinstance(file_obj, dict):
                    file_name = file_obj.get('file_name', 'N/A')
                    file_type = file_obj.get('file_type', 'N/A')
                else:
                    file_name = getattr(file_obj, 'file_name', 'N/A')
                    file_type = getattr(file_obj, 'file_type', 'N/A')
                f.write(f"- `{file_name}` ({file_type})\n")
        else:
            f.write("- No attachments.\n")
        
        f.write("\n## Pending Tasks\n")
        if state.get("task_plan"):
            for i, batch in enumerate(state["task_plan"]):
                f.write(f"### Batch {i+1}\n")
                for task in batch:
                    if isinstance(task, dict):
                        task_name = task.get('task_name', 'N/A')
                        task_description = task.get('task_description', 'N/A')
                        primary_id = task.get('primary', {}).get('id', 'N/A') if task.get('primary') else 'N/A'
                    else:
                        task_name = getattr(task, 'task_name', 'N/A')
                        task_description = getattr(task, 'task_description', 'N/A')
                        primary_id = getattr(getattr(task, 'primary', None), 'id', 'N/A')

                    f.write(f"- **Task**: `{task_name}`\n")
                    f.write(f"  - **Description**: {task_description}\n")
                    f.write(f"  - **Agent**: {primary_id}\n")
        else:
            f.write("- No pending tasks.\n")

        f.write("\n## Completed Tasks\n")
        if state.get("completed_tasks"):
            for task in state["completed_tasks"]:
                if isinstance(task, dict):
                    task_name = task.get('task_name', 'N/A')
                    result_obj = task.get('result', {})
                else:
                    task_name = getattr(task, 'task_name', 'N/A')
                    result_obj = getattr(task, 'result', {})
                    
                result_str = json.dumps(result_obj, indent=2, cls=CustomJSONEncoder)
                indented_result_str = "\n".join("      " + line for line in result_str.splitlines())

                f.write(f"- **Task**: `{task_name}`\n")
                f.write("  - **Result**:\n")
                f.write("    ```json\n")
                f.write(f"{indented_result_str}\n")
                f.write("    ```\n")
        else:
            f.write("- No completed tasks.\n")

    logger.info(f"Plan for thread {thread_id} saved to {plan_path}")
    return {}


# Lazy-loaded embeddings
_hf_embeddings = None
_embedding_model = None


def get_hf_embeddings():
    """Lazily load HuggingFace embeddings to avoid import-time issues."""
    global _hf_embeddings, _embedding_model
    if _hf_embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
        _hf_embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    return _hf_embeddings




def save_conversation_history(state: dict, *args, **kwargs):
    """Saves the conversation history to a JSON file. Accepts extra args for compatibility."""
    thread_id = state.get("thread_id")
    if not thread_id:
        logger.warning("No thread_id found in state, skipping history save")
        return

    history_dir = os.path.join(BACKEND_DIR, "agent_conversations")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"{thread_id}.json")
    
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
        
        data = {
            "thread_id": thread_id,
            "original_prompt": state.get("original_prompt"),
            "messages": serialized_messages,
            "todo_list": serialize_complex_object(state.get("todo_list", [])),
            "execution_plan": serialize_complex_object(state.get("execution_plan")),
            "action_history": serialize_complex_object(state.get("action_history", [])),
            "insights": serialize_complex_object(state.get("insights", {})),
            "memory": serialize_complex_object(state.get("memory", {}))
        }
        
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)
            
        logger.info(f"Conversation history saved to {history_path}")
    except Exception as e:
        logger.error(f"Failed to save conversation history: {e}")


def get_serializable_state(state: dict, *args, **kwargs) -> dict:
    """Returns a JSON-serializable version of the state. Accepts extra args for compatibility."""
    return serialize_complex_object(state)

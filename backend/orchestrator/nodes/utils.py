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


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for HttpUrl and other special types."""
    def default(self, o):
        if isinstance(o, HttpUrl):
            return str(o)
        return json.JSONEncoder.default(self, o)



class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for HttpUrl and other special types."""
    def default(self, o):
        if isinstance(o, HttpUrl):
            return str(o)
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
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if type(obj).__name__ == 'HttpUrl' or (hasattr(obj, '__class__') and 'HttpUrl' in str(type(obj))):
            return str(obj)
        elif hasattr(obj, 'model_dump'):
            try:
                return obj.model_dump(mode='json')
            except:
                pass
        elif hasattr(obj, 'dict'):
            try:
                return obj.dict()
            except:
                pass
        elif hasattr(obj, '__dict__'):
            try:
                return obj.__dict__
            except:
                pass
        elif isinstance(obj, (list, tuple)):
            try:
                if obj and all(hasattr(item, '_type') for item in obj if item is not None):
                    return messages_to_dict(obj)
                else:
                    result = []
                    for item in obj:
                        try:
                            serialized = serialize_complex_object(item)
                            result.append(serialized)
                        except Exception as e:
                            logger.warning(f"Failed to serialize list item: {e}")
                            result.append(str(item))
                    return result
            except Exception as e:
                logger.warning(f"Failed to serialize list: {e}")
                return [str(item) for item in obj]
        elif hasattr(obj, '_type'):
            try:
                return messages_to_dict([obj])[0] if messages_to_dict([obj]) else str(obj)
            except:
                pass
        
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


def save_plan_to_file(state: dict):
    """Saves the current plan and completed tasks to a Markdown file."""
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


def invoke_llm_with_fallback(primary_llm, fallback_llm, prompt: str, pydantic_schema: Any, max_retries: int = 2):
    """
    Invoke an LLM with fallback to Groq and NVIDIA when Cerebras fails with external API issues.
    Cycles through providers in order: Cerebras -> Groq -> NVIDIA -> Cerebras (2 times each).
    """
    cerebras_llm = primary_llm
    groq_llm = ChatGroq(model="openai/gpt-oss-120b") if os.getenv("GROQ_API_KEY") else None
    nvidia_llm = fallback_llm
    
    available_llms = []
    llm_names = []
    
    if cerebras_llm:
        available_llms.append(cerebras_llm)
        llm_names.append("Cerebras")
    
    if groq_llm:
        available_llms.append(groq_llm)
        llm_names.append("Groq")
    
    if nvidia_llm:
        available_llms.append(nvidia_llm)
        llm_names.append("NVIDIA")
    
    if not available_llms:
        raise ValueError("No LLMs available to process the request.")
    
    logger.info(f"Available LLMs: {', '.join(llm_names)}")
    
    errors = {}
    
    if pydantic_schema is not None:
        json_prompt = f"""
        {prompt}

        Please provide your response in a valid JSON format that adheres to the following Pydantic schema:
        
        ```json
        {json.dumps(pydantic_schema.model_json_schema(), indent=2)}
        ```

        IMPORTANT: Only output the JSON object itself, without any extra text, explanations, or markdown formatting.
        """
    else:
        json_prompt = prompt

    total_attempts = max_retries * len(available_llms)
    for attempt in range(total_attempts):
        llm_index = attempt % len(available_llms)
        current_llm = available_llms[llm_index]
        current_llm_name = llm_names[llm_index]
        
        llm_attempt = (attempt // len(available_llms)) + 1
        is_last_attempt = attempt == (total_attempts - 1)
        
        try:
            logger.info(f"Trying {current_llm_name} LLM, attempt {llm_attempt}/{max_retries}")
            response_content = current_llm.invoke(json_prompt).content
            logger.info(f"{current_llm_name.upper()} LLM RAW RESPONSE:\n---START---\n{response_content}\n---END---")
            
            if pydantic_schema is None:
                return response_content
            
            json_str = extract_json_from_response(response_content)
            if json_str:
                parsed_json = json.loads(json_str)
                validated_obj = pydantic_schema.model_validate(parsed_json)
                logger.info(f"Successfully got response from {current_llm_name} LLM.")
                return validated_obj
            else:
                error = ValueError(f"No valid JSON object could be extracted from the {current_llm_name} LLM response.")
                errors[current_llm_name] = error
                logger.warning(f"{current_llm_name} LLM attempt {llm_attempt} failed: {error}")
                if is_last_attempt:
                    raise error
        except Exception as e:
            errors[current_llm_name] = e
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["401", "429", "413", "rate", "too_many_requests", "high traffic", "queue_exceeded", "request too large", "tokens per", "wrong api key"]):
                logger.warning(f"{current_llm_name} LLM failed with API issue: {e}. Will try next LLM provider.")
                if not is_last_attempt:
                    continue
            else:
                logger.error(f"{current_llm_name} LLM failed with non-API error: {e}")
                raise

    logger.error("All LLM attempts exhausted. Returning None to trigger fallback plan creation.")
    logger.error(f"Errors encountered: {errors}")
    return None

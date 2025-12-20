# In Project_Agent_Directory/orchestrator/graph.py

from orchestrator.state import State, CompletedTask
from schemas import (
    ParsedRequest,
    TaskAgentPair,
    ExecutionPlan,
    AgentCard,
    PlannedTask,
    FileObject,
    AnalysisResult,
    EndpointDetail,
    EndpointParameterDetail
)
# Lazy import to avoid jaxlib issues
# from sentence_transformers import SentenceTransformer
from models import AgentCapability
import httpx
import asyncio
import json
import time
import os
import re
import base64
import numpy as np
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.networks import HttpUrl
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, ChatMessage
from langchain_cerebras import ChatCerebras
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from typing import Protocol, Any

# Define a protocol for the invoke_json method to help with type checking
class JsonInvoker(Protocol):
    def invoke_json(self, prompt: str, pydantic_schema: Any, max_retries: int = 3) -> Any:
        ...

# Extend ChatCerebras with the protocol to help type checkers
class ExtendedChatCerebras(ChatCerebras):
    pass
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Literal
from database import SessionLocal
from models import AgentCapability, AgentEndpoint
from sqlalchemy import select
import logging
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import messages_from_dict, messages_to_dict
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import json

class ForceJsonSerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "json", self.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.loads(data[1])

# Use absolute path relative to backend directory
BACKEND_DIR_FOR_HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONVERSATION_HISTORY_DIR = os.path.join(BACKEND_DIR_FOR_HISTORY, "conversation_history")
os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)

# --- Unified Content Management Integration ---
# Import content orchestrator for standardized file/artifact handling with agents
try:
    from orchestrator.content_orchestrator import (
        prepare_content_for_agent,
        inject_content_id_into_payload,
        capture_agent_outputs,
        agent_requires_file_upload,
        hooks as content_hooks,
        config as content_config,
        compress_state_for_saving,
        expand_state_from_saved,
        get_optimized_llm_context
    )
    CONTENT_INTEGRATION_ENABLED = True
    # Backward compatibility - use content_hooks for both
    artifact_hooks = content_hooks
    ARTIFACT_INTEGRATION_ENABLED = True
except ImportError as e:
    CONTENT_INTEGRATION_ENABLED = False
    ARTIFACT_INTEGRATION_ENABLED = False
    content_hooks = None
    artifact_hooks = None
    logging.warning(f"Content integration not available: {e}")

# Backward compatibility aliases
FILE_INTEGRATION_ENABLED = CONTENT_INTEGRATION_ENABLED
prepare_files_for_task = prepare_content_for_agent if CONTENT_INTEGRATION_ENABLED else None
inject_file_id_into_payload = inject_content_id_into_payload if CONTENT_INTEGRATION_ENABLED else None
capture_agent_output_files = capture_agent_outputs if CONTENT_INTEGRATION_ENABLED else None

# --- Imports for Document Processing ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, HttpUrl):
            return str(o)
        return json.JSONEncoder.default(self, o)

# --- Utility Function ---
def extract_json_from_response(text: str) -> str | None:
    '''
    A robust function to extract a JSON object from a string that may contain 
    , markdown, and other conversational text.

    Args:
        text: The raw string output from the language model.

    Returns:
        A clean string of the JSON object if found, otherwise None.
    '''
    if not isinstance(text, str):
        return None

    # 1. First, try to find a JSON object embedded in a markdown code block.
    # This is the most reliable method. The regex is non-greedy.
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # 2. If no markdown block is found, strip any <think> blocks and then
    # try to find the first valid JSON object in the remaining text.
    text_no_thinking = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 3. Find the first '{' and the last '}' in the cleaned text. This is a
    # common pattern for LLM responses that are just the JSON object.
    start = text_no_thinking.find('{')
    end = text_no_thinking.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text_no_thinking[start:end+1]
        try:
            # Validate if the extracted string is actually valid JSON
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # The substring was not valid JSON, so we'll pass and let the next method try
            pass

    # 4. As a last resort, if the above methods fail, return None.
    return None


def _summarize_completed_tasks_for_context(completed_tasks: List[Dict]) -> List[Dict]:
    """
    Summarize completed tasks to reduce context size for LLM calls.
    Keeps essential information while removing large data payloads.
    Specifically handles browser agent responses to extract useful content.
    For structured data results (dicts with numeric/simple values), preserves the full result.
    
    IMPORTANT: Deduplicates tasks by name, keeping only the LAST occurrence.
    This handles the case where a task is executed twice (preview + actual edit).
    """
    if not completed_tasks:
        return []
    
    # CRITICAL FIX: Deduplicate tasks by name, keeping only the LAST occurrence
    # This ensures we use the actual edit result, not the preview
    task_map = {}
    for task in completed_tasks:
        task_name = task.get("task_name", "unknown")
        task_map[task_name] = task  # Later occurrences overwrite earlier ones
    
    # Process deduplicated tasks
    summarized = []
    for task in task_map.values():
        summary = {
            "task_name": task.get("task_name", "unknown"),
            "status": "completed"
        }
        
        result = task.get("result", {})
        raw_response = task.get("raw_response", {})
        
        # Check if this is a document task - preserve important information
        task_name = task.get("task_name", "")
        is_document_task = any(keyword in task_name for keyword in [
            "document", "summarize", "analyze", "answer_question", "edit", "format", "create"
        ])
        
        # Check if this is a document edit/format task - preserve success/failure status
        is_document_edit_task = any(keyword in task_name for keyword in [
            "edit", "format", "create_document", "modify"
        ])
        
        logger.info(f"📝 Summarizing task '{task_name}': is_document_task={is_document_task}, is_document_edit_task={is_document_edit_task}, result_type={type(result).__name__}")
        
        if isinstance(result, str):
            # For document edit tasks, preserve success messages
            if is_document_edit_task and any(keyword in result.lower() for keyword in ["success", "edited", "formatted", "created", "saved"]):
                summary["result"] = result  # Keep full success message
                summary["edit_status"] = "success"
                logger.info(f"📝 Document edit success preserved: {result}")
            # For document analysis tasks, keep more content (up to 5000 chars)
            elif is_document_task:
                max_length = 5000
                summary["result"] = result[:max_length] + ("..." if len(result) > max_length else "")
                logger.info(f"📝 Document task result: length={len(result)}, truncated_to={len(summary['result'])}")
            else:
                # For other tasks, truncate to 500 chars
                max_length = 500
                summary["result"] = result[:max_length] + ("..." if len(result) > max_length else "")
                logger.info(f"📝 String result: length={len(result)}, truncated_to={len(summary['result'])}")
        elif isinstance(result, dict):
            # Special handling for document edit results with "message" key
            if is_document_edit_task and "message" in result:
                # Extract the success message
                summary["result"] = result["message"]
                summary["edit_status"] = "success"
                if "file_path" in result:
                    summary["file_path"] = result["file_path"]
                logger.info(f"📝 Document edit message extracted: {result['message']}")
            # Special handling for document edit results with canvas_display
            elif is_document_edit_task and "canvas_display" in result:
                # Extract key information from canvas_display
                canvas_display = result.get("canvas_display")
                # CRITICAL FIX: Handle None canvas_display (e.g., from multi-document analysis)
                if canvas_display is None:
                    logger.info("📝 canvas_display is None - skipping extraction")
                    summary["result"] = result
                else:
                    canvas_data = canvas_display.get("canvas_data", {})
                    
                    # Check if this is a successful edit
                    if canvas_data.get("status") == "edited":
                        summary["result"] = f"Document edited successfully: {canvas_data.get('title', 'document')}"
                        summary["edit_status"] = "success"
                        summary["file_path"] = canvas_data.get("file_path", "")
                        logger.info(f"📝 Document edit success extracted from canvas_display")
                    elif canvas_data.get("status") == "preview":
                        summary["result"] = f"Preview generated for: {canvas_data.get('title', 'document')}"
                        summary["edit_status"] = "preview"
                        summary["requires_confirmation"] = True
                        logger.info(f"📝 Document preview status extracted from canvas_display")
                    else:
                        summary["result"] = result
            # Special handling for document analysis results with "answer" key
            elif is_document_task and "answer" in result:
                # Preserve the full answer for document analysis
                answer_text = result["answer"]
                max_length = 5000
                summary["result"] = answer_text[:max_length] + ("..." if len(answer_text) > max_length else "")
                logger.info(f"📝 Document answer extracted: length={len(answer_text)}, truncated_to={len(summary['result'])}")
            else:
                # Check if result is structured data (small dict with simple values like numbers, bools, short strings)
                # This preserves API responses, stock data, weather data, etc.
                result_str = json.dumps(result, default=str)
                if len(result_str) < 2000:  # Small structured data - keep it all
                    summary["result"] = result
                else:
                    # Large result - extract key information
                    if "status" in result:
                        summary["result_status"] = result["status"]
                    if "summary" in result:
                        summary["result_summary"] = str(result["summary"])[:500]
                    elif "task_summary" in result:
                        summary["result_summary"] = str(result["task_summary"])[:500]
        
        # Handle browser agent raw_response specially - extract the useful content
        if raw_response and isinstance(raw_response, dict):
            # Get task summary
            if "task_summary" in raw_response:
                summary["task_summary"] = raw_response["task_summary"]
            
            # Extract the actual useful data from browser results
            if "extracted_data" in raw_response:
                extracted = raw_response["extracted_data"]
                
                # Handle entries array (browser agent format)
                if isinstance(extracted, dict) and "entries" in extracted:
                    entries = extracted["entries"]
                    if entries and len(entries) > 0:
                        # Get the text_content which has the actual useful data
                        first_entry = entries[0]
                        if "text_content" in first_entry:
                            # This is the key data - the actual page content
                            text_content = first_entry["text_content"]
                            # Truncate but keep enough for useful context
                            summary["page_content"] = text_content[:2000] if len(text_content) > 2000 else text_content
                        if "title" in first_entry:
                            summary["page_title"] = first_entry["title"]
                        if "url" in first_entry:
                            summary["page_url"] = first_entry["url"]
                        # Skip links, images, headings - they bloat the context
                
                # Handle direct extracted_data dict
                elif isinstance(extracted, dict):
                    if "text_content" in extracted:
                        summary["page_content"] = extracted["text_content"][:2000]
                    if "title" in extracted:
                        summary["page_title"] = extracted["title"]
            
            # Include actions summary
            if "actions_taken" in raw_response:
                actions = raw_response["actions_taken"]
                summary["actions_count"] = len(actions)
                # Just include action types, not full details
                summary["actions"] = [a.get("action", "unknown") for a in actions[:5]]
        
        # Include error info if present
        if isinstance(result, dict) and "error" in result:
            summary["error"] = str(result["error"])[:200]
        
        summarized.append(summary)
    
    return summarized


def transform_payload_types(payload: Dict[str, Any], parameters: List[Any]) -> Dict[str, Any]:
    '''
    Transform payload parameter types to match the endpoint schema.
    Fixes common issues like:
    - String values that should be arrays (e.g., "email@example.com" -> ["email@example.com"])
    - Missing optional parameters with defaults
    '''
    transformed = payload.copy()
    
    for param in parameters:
        param_name = param.name
        param_type = param.param_type
        
        # Skip if parameter not in payload
        if param_name not in transformed:
            continue
        
        value = transformed[param_name]
        
        # Transform string to array if parameter expects array
        if param_type == "array" and isinstance(value, str):
            logger.info(f"Transforming parameter '{param_name}' from string to array: '{value}' -> ['{value}']")
            transformed[param_name] = [value]
        
        # Transform single item to array if parameter expects array
        elif param_type == "array" and not isinstance(value, list):
            logger.info(f"Transforming parameter '{param_name}' to array: {value} -> [{value}]")
            transformed[param_name] = [value]
        
        # Ensure integer type
        elif param_type == "integer" and isinstance(value, str):
            try:
                transformed[param_name] = int(value)
                logger.info(f"Transformed parameter '{param_name}' from string to integer: '{value}' -> {transformed[param_name]}")
            except ValueError:
                logger.warning(f"Could not convert '{param_name}' value '{value}' to integer")
    
    return transformed

def serialize_complex_object(obj):
    '''Helper function to serialize complex objects consistently'''
    try:
        # First try direct JSON serialization
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Handle different object types
        # Check for HttpUrl by type name instead of isinstance (avoids subscripted generics error)
        if type(obj).__name__ == 'HttpUrl' or (hasattr(obj, '__class__') and 'HttpUrl' in str(type(obj))):
            return str(obj)  # Convert HttpUrl to string
        elif hasattr(obj, 'model_dump'):
            # Pydantic v2 models
            try:
                return obj.model_dump(mode='json')
            except:
                pass
        elif hasattr(obj, 'dict'):
            # Pydantic models
            try:
                return obj.dict()
            except:
                pass
        elif hasattr(obj, '__dict__'):
            # Regular Python objects
            try:
                return obj.__dict__
            except:
                pass
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples of complex objects
            try:
                # Check if this is a list of LangChain message objects
                if obj and all(hasattr(item, '_type') for item in obj if item is not None):
                    # Use LangChain's messages_to_dict for message objects
                    return messages_to_dict(obj)
                else:
                    # Recursively serialize each item in the list
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
        elif hasattr(obj, '_type'):  # Check for LangChain message objects
            # Handle LangChain message objects specifically
            try:
                # Use LangChain's messages_to_dict for individual message objects
                return messages_to_dict([obj])[0] if messages_to_dict([obj]) else str(obj)
            except:
                pass
        elif hasattr(obj, 'model_dump'):
            # Newer Pydantic v2 models
            try:
                return obj.model_dump(mode='json') # Use mode='json' here too for safety
            except:
                pass
        
        # Fallback to string representation
        return str(obj)

# --- New Pydantic Schemas for New Nodes ---
class PlanValidation(BaseModel):
    '''Schema for the pre-flight plan validation node.'''
    status: str = Field(..., description="Either 'ready' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The question to ask the user if parameters are missing.")

class AgentResponseEvaluation(BaseModel):
    '''Schema for evaluating an agent's response post-flight.'''
    status: str = Field(..., description="Either 'complete' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The clarifying question to ask the user if the result is vague.")

class PlanValidationResult(BaseModel):
    '''Schema for the advanced validation node's output.'''
    status: Literal["ready", "replan_needed", "user_input_required"] = Field(..., description="The status of the plan validation.")
    reasoning: Optional[str] = Field(None, description="Required explanation if status is 'replan_needed' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The direct question for the user if input is absolutely required.")


def strip_think_tags(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Monkey-patch the ChatCerebras class to add the invoke_json method and strip_think_tags
original_generate = ChatCerebras._generate

def patched_generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
    chat_result = original_generate(self, messages, stop, run_manager, **kwargs)
    for generation in chat_result.generations:
        if isinstance(generation, ChatGeneration) and hasattr(generation.message, 'content'):
            original_content = generation.message.content
            generation.message.content = strip_think_tags(original_content)
    return chat_result

def invoke_json_method(self, prompt: str, pydantic_schema: Any, max_retries: int = 3):
    '''
    A more robust version of invoke_json that uses the enhanced parser.
    '''
    original_prompt = prompt  
    
    # Try ChatCerebras first
    for attempt in range(max_retries):
        failed_object_str = ""
        try:
            # The initial prompt to the LLM remains the same
            if pydantic_schema is not None:
                json_prompt = f'''
                {prompt}

                Please provide your response in a valid JSON format that adheres to the following Pydantic schema:
                
                ```json
                {json.dumps(pydantic_schema.model_json_schema(), indent=2)}
                ```

                IMPORTANT: Only output the JSON object itself, without any extra text, explanations, or markdown formatting.
                '''
            else:
                json_prompt = prompt
                
            response_content = self.invoke(json_prompt).content
            logger.info(f"LLM RAW RESPONSE (Attempt {attempt + 1}):\n---START---\n{response_content}\n---END---")

            # --- USE THE NEW PARSER HERE ---
            json_str = extract_json_from_response(response_content)
            
            if json_str and pydantic_schema is not None:
                parsed_json = json.loads(json_str)
                validated_obj = pydantic_schema.model_validate(parsed_json)
                
                # Use model_dump_json for safe serialization
                failed_object_str = validated_obj.model_dump_json(indent=2)
                
                return validated_obj
            elif json_str and pydantic_schema is None:
                # If schema is None, just return the parsed JSON
                parsed_json = json.loads(json_str)
                return parsed_json
            else:
                # If the new parser returns None, no valid JSON was found
                raise ValueError("No valid JSON object could be extracted from the response.")

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # The retry logic remains the same
                retry_context = f"<your_previous_invalid_output>\n{failed_object_str}\n</your_previous_invalid_output>" if failed_object_str else ""
                prompt = f'''
                Your previous attempt failed because the output was not valid JSON or could not be extracted. Please re-evaluate the original request and provide a valid, clean JSON response.
                <error>{e}</error>
                {retry_context}
                Original prompt was:\n{original_prompt}
                Please correct your response and try again.
                '''
            else:
                logging.error(f"Failed to get a valid JSON response after {max_retries} attempts.")
                raise

# Create a fallback wrapper for LLM calls
def invoke_llm_with_fallback(primary_llm, fallback_llm, prompt: str, pydantic_schema: Any, max_retries: int = 2):
    '''
    Invoke an LLM with fallback to Groq and NVIDIA when Cerebras fails with external API issues.
    Cycles through providers in order: Cerebras -> Groq -> NVIDIA -> Cerebras (2 times each).
    '''
    # Initialize all LLMs
    cerebras_llm = primary_llm
    groq_llm = ChatGroq(model="openai/gpt-oss-120b") if os.getenv("GROQ_API_KEY") else None
    nvidia_llm = fallback_llm  # ChatNVIDIA
    
    # Create list of available LLMs
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
    
    # Track errors for each provider
    errors = {}
    
    # Prepare prompts based on schema
    if pydantic_schema is not None:
        json_prompt = f'''
        {prompt}

        Please provide your response in a valid JSON format that adheres to the following Pydantic schema:
        
        ```json
        {json.dumps(pydantic_schema.model_json_schema(), indent=2)}
        ```

        IMPORTANT: Only output the JSON object itself, without any extra text, explanations, or markdown formatting.
        '''
    else:
        json_prompt = prompt

    # Cycle through providers: Cerebras -> Groq -> NVIDIA -> Cerebras (2 times each)
    total_attempts = max_retries * len(available_llms)  # 2 cycles * 3 providers = 6 total attempts
    for attempt in range(total_attempts):
        # Determine which LLM to use (cycle through available LLMs)
        llm_index = attempt % len(available_llms)
        current_llm = available_llms[llm_index]
        current_llm_name = llm_names[llm_index]
        
        # Calculate attempt number for this specific LLM (1 or 2)
        llm_attempt = (attempt // len(available_llms)) + 1
        is_last_attempt = attempt == (total_attempts - 1)
        
        try:
            logger.info(f"Trying {current_llm_name} LLM, attempt {llm_attempt}/{max_retries}")
            response_content = current_llm.invoke(json_prompt).content
            logger.info(f"{current_llm_name.upper()} LLM RAW RESPONSE:\n---START---\n{response_content}\n---END---")
            
            # For non-Pydantic responses, return the content directly
            if pydantic_schema is None:
                return response_content
            
            # Parse the response using the same logic for Pydantic schemas
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
            # Check if this is an external API issue that warrants trying the next provider
            # Include 413 (request too large) as some providers have different token limits
            # Include 401 (wrong API key) to allow fallback to other providers
            if any(keyword in error_msg for keyword in ["401", "429", "413", "rate", "too_many_requests", "high traffic", "queue_exceeded", "request too large", "tokens per", "wrong api key"]):
                logger.warning(f"{current_llm_name} LLM failed with API issue: {e}. Will try next LLM provider.")
                if is_last_attempt:
                    # If this is the last attempt and it failed, we've exhausted all options
                    pass
                else:
                    continue  # Continue to try next LLM provider
            else:
                # For non-API errors, re-raise immediately
                logger.error(f"{current_llm_name} LLM failed with non-API error: {e}")
                raise

    # If we've exhausted all attempts, return None to trigger fallback logic
    logger.error("All LLM attempts exhausted. Returning None to trigger fallback plan creation.")
    logger.error(f"Errors encountered: {errors}")
    
    # Don't try to create minimal objects - let the caller handle the fallback
    # This ensures we use the simple_plan creation logic which properly fills parameters
    return None

ChatCerebras._generate = patched_generate

# Add the invoke_json method as a bound method to the class
def bind_invoke_json_method(self, prompt: str, pydantic_schema: Any, max_retries: int = 3):
    return invoke_json_method(self, prompt, pydantic_schema, max_retries)

# Add the method to the class using setattr
setattr(ChatCerebras, 'invoke_json', bind_invoke_json_method)

logging.info("ChatCerebras has been monkey-patched to strip  and handle JSON manually.")

load_dotenv()

logger = logging.getLogger("AgentOrchestrator")
# Use absolute path relative to this file's directory
ORCHESTRATOR_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(ORCHESTRATOR_DIR)
PLAN_DIR = os.path.join(BACKEND_DIR, "agent_plans")
os.makedirs(PLAN_DIR, exist_ok=True)
os.makedirs("storage/vector_store", exist_ok=True)

# Create a function to load embeddings lazily
def get_hf_embeddings():
    """Lazily load HuggingFace embeddings to avoid import-time issues."""
    global hf_embeddings
    if 'hf_embeddings' not in globals():
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        global embedding_model
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        hf_embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
    return hf_embeddings


cached_capabilities = {
    "texts": [],
    "embeddings": None,
    "timestamp": 0
}
CACHE_DURATION_SECONDS = 30  # 30 seconds for faster development

# OPTIMIZATION: GET request cache for agent responses
get_request_cache = {}
GET_CACHE_DURATION_SECONDS = 300  # 5 minutes

# --- New File-Based Memory Functions ---
def save_plan_to_file(state: dict):
    '''Saves the current plan and completed tasks to a Markdown file.'''
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

                    # Write each line separately to ensure correct newlines
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
                
                # Indent every line of the JSON string for proper markdown rendering
                indented_result_str = "\n".join("      " + line for line in result_str.splitlines())

                # Write each part of the completed task entry separately
                f.write(f"- **Task**: `{task_name}`\n")
                f.write(" - **Result**:\n")
                f.write("    ```json\n")
                f.write(f"{indented_result_str}\n")
                f.write("    ```\n")
        else:
            f.write("- No completed tasks.\n")

    logger.info(f"Plan for thread {thread_id} saved to {plan_path}")
    return {}

# --- NEW NODE: Preprocess Files ---
async def preprocess_files(state: State):
    '''
    Processes uploaded files asynchronously with caching.
    - Images: Validates path only
    - Documents: Creates vector store + displays (async + cached)
    - Spreadsheets: Uploads to agent + displays (async)
    '''
    logger.info("⚡ Starting ASYNC file preprocessing...")
    logger.info(f"🔍 PREPROCESS_FILES ENTRY: State keys = {list(state.keys())}")
    logger.info(f"🔍 PREPROCESS_FILES ENTRY: uploaded_files in state = {'uploaded_files' in state}")
    uploaded_files = state.get("uploaded_files", [])
    logger.info(f"📁 Found {len(uploaded_files)} files to preprocess")
    logger.info(f"🔍 PREPROCESS_FILES: uploaded_files = {uploaded_files}")
    if not uploaded_files:
        logger.info("No files to preprocess.")
        return state

    # Import the async file processor
    from services.file_processor import file_processor
    
    # Separate files by type for batch processing
    image_files = []
    document_files = []
    spreadsheet_files = []
    canvas_displays = []  # Collect canvas displays
    
    # Convert dictionaries from state back to Pydantic objects
    for idx, file_obj_dict in enumerate(uploaded_files):
        try:
            file_obj = FileObject.model_validate(file_obj_dict)
            logger.info(f"📄 File {idx+1}/{len(uploaded_files)}: {file_obj.file_name}, type={file_obj.file_type}")
            
            if file_obj.file_type == 'image':
                # Images: just validate path
                if os.path.exists(file_obj.file_path):
                    image_files.append(file_obj)
                else:
                    logger.warning(f"Image not found: {file_obj.file_path}")
            elif file_obj.file_type == 'document':
                document_files.append(file_obj)
            elif file_obj.file_type == 'spreadsheet':
                spreadsheet_files.append(file_obj)
                
        except Exception as e:
            logger.error(f"Failed to validate file {file_obj_dict.get('file_name')}: {e}")
            continue
    
    # Process all files concurrently (with limit of 3 concurrent)
    all_files_to_process = document_files + spreadsheet_files
    logger.info(f"⚡ Processing {len(all_files_to_process)} files async ({len(document_files)} docs, {len(spreadsheet_files)} sheets)")
    
    if all_files_to_process:
        processing_results = await file_processor.process_files_batch(
            all_files_to_process,
            max_concurrent=3
        )
        
        # Update file objects with processing results
        for file_obj, result in zip(all_files_to_process, processing_results):
            if 'error' in result:
                logger.error(f"❌ Processing failed for {file_obj.file_name}: {result['error']}")
                continue
            
            # Update file object with results
            if file_obj.file_type == 'document':
                file_obj.vector_store_path = result.get('vector_store_path')
                logger.info(f"✅ Document processed: {file_obj.file_name} "
                          f"({result.get('chunks_count')} chunks, "
                          f"{result.get('processing_time', 0):.2f}s, "
                          f"cached={result.get('cached', False)})")
            elif file_obj.file_type == 'spreadsheet':
                file_id = result.get('file_id')
                if file_id:
                    file_obj.file_id = file_id
                    file_obj.content_id = file_id
                    logger.info(f"✅ Spreadsheet uploaded: {file_obj.file_name} "
                              f"(file_id={file_id}, cached={result.get('cached', False)})")
            
            # Collect canvas displays
            canvas_display = result.get('canvas_display')
            if canvas_display:
                canvas_displays.append(canvas_display)
    
    # Combine all processed files
    processed_files = image_files + all_files_to_process
    
    # Prepare result with canvas displays
    result = {"uploaded_files": [pf.model_dump(mode='json', exclude_none=True) for pf in processed_files]}
    logger.info(f"🔍 PREPROCESS_FILES: Returning {len(processed_files)} uploaded files to state")
    logger.info(f"🔍 PREPROCESS_FILES: uploaded_files={result['uploaded_files']}")
    
    if canvas_displays:
        # Use the first canvas display (or combine multiple if needed)
        if len(canvas_displays) == 1:
            canvas_display = canvas_displays[0]
        else:
            # Determine the combined type based on what's uploaded
            file_types = set(pf.file_type for pf in processed_files if pf.file_type in ['document', 'spreadsheet'])
            combined_type = list(file_types)[0] if len(file_types) == 1 else 'document'
            canvas_display = {
                "canvas_type": combined_type,
                "canvas_data": {
                    "title": f"{len(canvas_displays)} Files Uploaded",
                    "content": f"Uploaded {len(canvas_displays)} file(s). They are ready for analysis.",
                    "items": canvas_displays
                }
            }
        
        # Set canvas state fields properly
        result["canvas_content"] = canvas_display.get('canvas_content')
        result["canvas_data"] = canvas_display.get('canvas_data')
        result["canvas_type"] = canvas_display.get('canvas_type')
        result["canvas_title"] = canvas_display.get('canvas_title')
        result["has_canvas"] = True
        
        # Store canvas_displays in state for later use
        result["canvas_displays"] = canvas_displays
        
        # Add a system message to inform user
        from orchestrator.message_manager import MessageManager
        existing_messages = state.get("messages", [])
        
        doc_names = [pf.file_name for pf in processed_files if pf.file_type == 'document']
        spreadsheet_names = [pf.file_name for pf in processed_files if pf.file_type == 'spreadsheet']
        
        if spreadsheet_names:
            message_content = f"📊 Displaying uploaded spreadsheet{'s' if len(spreadsheet_names) > 1 else ''}: {', '.join(spreadsheet_names)}"
        elif doc_names:
            message_content = f"📄 Displaying uploaded document{'s' if len(doc_names) > 1 else ''}: {', '.join(doc_names)}"
        else:
            message_content = f"📁 Uploaded {len(canvas_displays)} file(s)"
        
        system_message = AIMessage(content=message_content)
        updated_messages = MessageManager.add_message(existing_messages, system_message)
        result["messages"] = updated_messages
        
        logger.info(f"📊 Canvas display set for {len(canvas_displays)} uploaded file(s)")
    
    return result


# --- Existing and Modified Graph Nodes ---

def get_all_capabilities():
    global cached_capabilities
    now = time.time()

    if now - cached_capabilities["timestamp"] < CACHE_DURATION_SECONDS and cached_capabilities["texts"]:
        logger.info("Using cached capabilities and embeddings.")
        return cached_capabilities["texts"], cached_capabilities["embeddings"]

    logger.info("Fetching and embedding capabilities from database...")
    db = SessionLocal()
    try:
        results = db.query(AgentCapability.capability_text).distinct().all()
        capability_texts = [res[0] for res in results]
        
        if capability_texts:
            cached_capabilities["texts"] = capability_texts
            # Embeddings are no longer used in the graph but kept for potential future use
            cached_capabilities["timestamp"] = now
        else:
            cached_capabilities["texts"] = []
            
        return cached_capabilities["texts"], None # Return None for embeddings
    finally:
        db.close()

async def fetch_agents_for_task(client: httpx.AsyncClient, task_name: str, url: str):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
            validated_agents_as_dicts = [AgentCard.model_validate(agent).model_dump(mode='json') for agent in response.json()]
            return {"task_name": task_name, "agents": validated_agents_as_dicts}
        except (httpx.RequestError, httpx.HTTPStatusError, ValidationError) as e:
            # Check if this is a rate limit error
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + (0.1 * attempt)
                    logger.warning(f"Rate limit hit for task '{task_name}'. Waiting {wait_time:.2f} seconds before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached for task '{task_name}' due to rate limiting: {e}")
            else:
                logger.error(f"API call or validation failed for task '{task_name}': {e}")
            
            # Return empty agents if all retries failed or if it's not a retryable error
            if attempt == max_retries - 1:
                return {"task_name": task_name, "agents": []}

def parse_prompt(state: State):
    """
    SUPER-PARSER: 3-in-1 optimization
    1. Chitchat detection (direct response)
    2. Task parsing with parameter extraction
    3. Title generation (first turn only)
    """
    # --- Create a formatted history of the conversation ---
    history = ""
    if messages := state.get('messages'):
        # Limit to the last few messages to keep the prompt concise
        for msg in messages[-20:]:  # Using the last 20 messages as context
            if hasattr(msg, 'type') and msg.type == "human":
                history += f"Human: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history += f"AI: {msg.content}\n"

    # Check if this is the first turn (for title generation)
    is_first_turn = not state.get("messages") or len(state.get("messages")) == 0
    
    capability_texts, _ = get_all_capabilities()
    capabilities_list_str = ", ".join(f"'{c}'" for c in capability_texts)
    
    # Add file context if files are uploaded
    file_context = ""
    uploaded_files = state.get("uploaded_files", [])
    if uploaded_files:
        # Handle both dict and FileObject instances - include file type information
        file_details = []
        for f in uploaded_files:
            if isinstance(f, dict):
                file_name = f.get('file_name', 'unknown')
                file_type = f.get('file_type', 'document')
            else:
                file_name = f.file_name
                file_type = f.file_type
            file_details.append(f"{file_name} (type: {file_type})")
        
        file_context = f'''
        **UPLOADED FILES DETECTED:**
        The user has uploaded {len(file_details)} file(s): {", ".join(file_details)}
        
        **CRITICAL INSTRUCTIONS BASED ON FILE TYPE:**
        - **For spreadsheet files (.csv, .xlsx):** Create tasks like "display spreadsheet", "analyze spreadsheet data", "query spreadsheet" - NOT document tasks!
        - **For document files (.pdf, .docx, .txt):** Create tasks like "summarize_document", "answer_question_about_document"
        - **For image files:** Create tasks like "analyze image", "describe image"
        
        **IMPORTANT:** Match the task type to the file type! CSV files need spreadsheet tasks, not document tasks.
        - If the user asks ANY question about the file(s) (e.g., "What is this?", "Display this", "Show me the data"), create the appropriate task based on file type.
        - The task_description MUST include the file name and match the file type.
        - DO NOT treat this as chitchat - files uploaded = task required!
        '''
        logger.info(f"[PARSE_PROMPT_DEBUG] File context built for {len(file_details)} files: {file_details}")
    else:
        logger.info("[PARSE_PROMPT_DEBUG] No uploaded_files in state")
    
    logger.info(f"[PARSE_PROMPT_DEBUG] Capabilities count: {len(capability_texts)}, first 5: {capability_texts[:5]}")
    logger.info(f"[PARSE_PROMPT_DEBUG] Original prompt: '{state['original_prompt']}'")


    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None

    error_feedback = state.get("parsing_error_feedback")
    retry_prompt_injection = ""
    if error_feedback:
        retry_prompt_injection = f'''
        **IMPORTANT - PREVIOUS ATTEMPT FAILED:**
        You are being asked to try again because your previous attempt failed to produce a useful result.
        **Failure Feedback:** {error_feedback}
        Please analyze this feedback and the original user prompt carefully and generate a new set of tasks with much more detailed and specific `task_description` fields.
        '''

    prompt = f'''
        You are the Orbimesh Orchestrator, an intelligent AI system that coordinates multiple specialized agents to complete user requests.
        
        **YOUR IDENTITY AND CAPABILITIES:**
        - You are Orbimesh, a multi-agent orchestration system
        - You can delegate tasks to specialized agents in your agent directory
        - You have a built-in Canvas feature that can render interactive HTML/CSS/JavaScript and Markdown content
        - The Canvas is NOT an agent - it's your own built-in capability for creating visualizations, games, interactive demos, and rich content
        
        **SUPER-PARSER MODE: You have THREE jobs in this single call:**
        
        **JOB 1: CHITCHAT CHECK (Priority)**
        If the user input is a simple greeting ("Hi", "Hello"), acknowledgment ("Thanks", "OK"), or general question requiring NO external tools/agents, DO NOT generate tasks. Instead:
        - Set `direct_response` with a helpful, friendly reply
        - Set `tasks` to an empty list
        - This will skip the entire orchestration graph for efficiency
        
        **CRITICAL: If the user has uploaded files, ANY question about those files (even vague ones like "What is this?", "Tell me about this", "Display this", "Show me") MUST be treated as a file-related task (spreadsheet task for CSV/Excel, document task for PDF/DOCX), NOT chitchat!**
        
        **JOB 2: TASK PARSING (If not chitchat)**
        If the user needs external agents (search, analysis, data fetching), break it down into tasks.
        - **Extract Parameters:** If the user provides specific values (e.g., "Search for Apple stock"), extract them into the `parameters` dictionary (e.g., `parameters: {{"query": "Apple", "ticker": "AAPL"}}`)
        - This optimization allows us to skip LLM calls during execution if all required parameters are already extracted
        
        **JOB 3: TITLE GENERATION (First turn only)**
        {"Since this is the first message in the conversation, generate a short 3-5 word title in `suggested_title` that summarizes the user's request." if is_first_turn else "This is a continuation, so leave `suggested_title` as null."}
        
        **IMPORTANT: When users ask for interactive content, games, visualizations, or web-based demos:**
        - DO NOT create a task to search for a "canvas agent" or "visualization agent"
        - These will be handled by your built-in Canvas feature in the final response generation
        - Focus on breaking down only the data-fetching or computation tasks that need external agents
        
        You are an expert at breaking down any user request—no matter how short, vague, or poorly written—into a clear list of distinct tasks that can each be handled by a single agent.
        {retry_prompt_injection}

        Here is the recent conversation history for context:
        ---
        {history}
        ---
        
        {file_context}

        Here is a list of agent capabilities that already exist in the system:
        ---
        AVAILABLE CAPABILITIES: [{capabilities_list_str}]
        ---

        Follow these rules:
        1.  **Group Related Information:** If the user asks for multiple pieces of information that are likely to be returned by a single tool or API call (e.g., "get news headlines, publishers, and links" or "get a stock's open, high, low, and close price"), you **MUST** treat this as a single, unified task. Do not split these into separate tasks. For example, a request for "news headlines, publishers, and links" should become a single task like "get company news with details".
        2.  **One Task, One Agent:** A "task" must represent ONE coherent, self-contained action that can be given to a single agent.
        3.  **No Unnecessary Splitting:** Do NOT split a task into smaller parts unless they are truly independent and could be completed by different agents without losing context.
        4.  **Simple Language:** Keep language simple and avoid technical jargon unless the user explicitly uses it.
        5.  **Infer Intent:** If the prompt is unclear, infer the most reasonable interpretation based on common intent.
        6.  **Strict Schema:** Always output tasks in the required schema.
        7.  **Decompose Analytical Requests:** If the user's request requires multiple distinct capabilities or asks for analysis (e.g., 'find X and then analyze its effect on Y', 'compare X and Y'), you **MUST** break it down into a sequence of discrete tasks. For example, a request to 'see which news affected stocks' should be decomposed into two separate tasks: one for `get company stock history` and another for `get company news headlines`. The final analysis will be handled by a later step.
        8.  **Prioritize Specificity:** When creating a `task_description`, be as specific and detailed as possible. The description should be a clear, self-contained instruction for another agent. For example, instead of "find company news," write "Find the three most recent news articles about the company 'TechCorp' and extract their headlines, publication dates, and a brief summary of each." This level of detail is crucial for the next agent to perform its job accurately.

        For each task you identify, provide:
        1. `task_name`: A short, descriptive name (e.g., "get_company_news", "summarize_document").
            - **Check Existing Capabilities First:** When choosing a `task_name`, you **MUST** check the AVAILABLE CAPABILITIES list. If a capability in the list is a good match for the grouped task, use that exact capability as the `task_name`.
            - **Create New if Needed:** If no single existing capability is a good fit for the grouped task, create a new, concise, 2-4 word `task_name` that accurately describes the entire action (e.g., "get_news_details", "get_ohlc_prices").
            - **Prefer Existing:** Always prefer using an existing capability if it covers the user's request to ensure a higher chance of finding an agent.
        2. `task_description`: A detailed explanation of what the task is and what needs to be done, including all the details from the user's prompt. For example, for "get AAPL news headlines with publishers and links", the description should be "Get the latest news headlines for AAPL, including the publisher and a link to the article for each headline."
        3. `parameters`: A dictionary of pre-extracted parameters from the user's prompt (e.g., {{"ticker": "AAPL", "query": "Apple stock news"}}). This is CRITICAL for optimization - extract as many specific values as possible.

        Also extract any general user expectations (tone, urgency, budget, quality rating, etc.) from the prompt, if present. If not present, set them to null.

        The user's prompt will be provided like this:
        ---
        {state['original_prompt']}
        ---

        **EXAMPLES:**
        
        Example 1 (Chitchat): If the user prompt is "Hi there!", your output should be:
        ```json
        {{
            "tasks": [],
            "user_expectations": {{}},
            "direct_response": "Hello! I'm Orbimesh, your AI orchestration assistant. I can help you with data analysis, web searches, document processing, and much more. What would you like to do today?",
            "suggested_title": "Greeting"
        }}
        ```
        
        Example 2 (Complex with parameters): If the user prompt is "Get the latest 10 news headlines for AAPL with publishers and article links.", your output should be:
        ```json
        {{
            "tasks": [
                {{
                    "task_name": "get company news headlines",
                    "task_description": "Get the latest 10 news headlines for AAPL, including the publisher and a link to the article for each headline.",
                    "parameters": {{
                        "ticker": "AAPL",
                        "limit": 10,
                        "include_publisher": true,
                        "include_link": true
                    }}
                }}
            ],
            "user_expectations": {{}},
            "direct_response": null,
            "suggested_title": "AAPL News Headlines"
        }}
        ```

        Your output must follow the schema exactly, and all number fields must be numeric or null (never strings).
        When extracting `user_expectations`, follow this strictly:
        - Only include fields that the user explicitly mentioned (e.g., price, budget, tone, urgency, quality rating).
        - Do NOT include any field with a null value.
        - If, after removing nulls, no fields remain, set `user_expectations` to an empty object `{{}}`.
    '''

    try:
        # Use the fallback wrapper for LLM calls
        response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, ParsedRequest)
        logger.info(f"LLM parsed prompt into: {response}")

        # OPTIMIZATION: Check for direct response (chitchat short-circuit)
        if response and response.direct_response:
            logger.info("Direct response detected (chitchat). Short-circuiting orchestration graph.")
            return {
                "parsed_tasks": [],
                "user_expectations": {},
                "final_response": response.direct_response,
                "suggested_title": response.suggested_title,
                "needs_complex_processing": False,  # Flag to skip orchestration
                "parsing_error_feedback": None,
                "parse_retry_count": 0
            }

        if not response or not response.tasks:
            logger.warning("LLM returned a valid JSON but with an empty list of tasks. This may be a misclassified simple request.")
            parsed_tasks = []
            user_expectations = {}
        else:
            parsed_tasks = getattr(response, 'tasks', [])
            user_expectations = getattr(response, 'user_expectations', {})

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to parse prompt after all retries: {e}")
        
        # Check if this is an external API issue (like rate limiting)
        if "429" in error_msg or "rate" in error_msg.lower() or "too_many_requests" in error_msg.lower() or "high traffic" in error_msg.lower():
            # This is an external issue, not a user input issue
            logger.warning("External API issue detected - routing to final response instead of asking user")
            return {
                "parsed_tasks": [],
                "user_expectations": {},
                "parsing_error_feedback": None,
                "parse_retry_count": state.get('parse_retry_count', 0) + 1,
                "final_response": f"Sorry, I'm currently experiencing high traffic or technical issues with the underlying services. Please try again later. Error: {str(e)}"
            }
        else:
            # This is likely a user input issue
            parsed_tasks = []
            user_expectations = {}
            response = None  # Set response to None when exception occurs

    current_retry_count = state.get('parse_retry_count', 0)

    return {
        "parsed_tasks": parsed_tasks,
        "user_expectations": user_expectations or {},
        "parsing_error_feedback": None,
        "parse_retry_count": current_retry_count + 1,
        "suggested_title": getattr(response, 'suggested_title', None) if response else None,
        "needs_complex_processing": True  # Complex processing needed if we got here
    }

async def agent_directory_search(state: State):
    """
    LLM-BASED SEMANTIC AGENT SELECTION (Primary) with vector similarity as fallback.
    
    This approach:
    1. Fetches ALL active agents with their names, descriptions, and capabilities
    2. Uses LLM to semantically match agents to tasks based on the full context
    3. Falls back to vector similarity search if LLM fails
    
    Benefits:
    - Respects explicit user preferences (e.g., "use browser automation agent")
    - Better semantic understanding of task requirements
    - More accurate agent selection for complex/ambiguous tasks
    """
    parsed_tasks = state.get('parsed_tasks', [])
    original_prompt = state.get('original_prompt', '')
    logger.info(f"Searching for agents for tasks: {[t.task_name for t in parsed_tasks]}")
    
    if not parsed_tasks:
        logger.warning("No valid tasks to process in agent_directory_search")
        return {"candidate_agents": {}}
    
    user_expectations = state.get('user_expectations') or {}
    candidate_agents_map = {}
    
    db = SessionLocal()
    try:
        from models import Agent, AgentCapability
        from sqlalchemy.orm import joinedload
        
        # Fetch ALL active agents with their full details
        query = db.query(Agent).options(
            joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
        ).filter(Agent.status == 'active')
        
        # Apply user expectations filters
        if 'price' in user_expectations:
            max_price = user_expectations['price']
            query = query.filter(Agent.price_per_call_usd <= max_price)
        
        if 'rating' in user_expectations:
            min_rating = user_expectations['rating']
            query = query.filter(Agent.rating >= min_rating)
        
        all_agents = query.all()
        logger.info(f"Fetched {len(all_agents)} active agents for LLM-based selection")
        
        if not all_agents:
            logger.warning("No active agents found in database")
            return {"candidate_agents": {}, "parsing_error_feedback": "No active agents available in the system."}
        
        # Build agent catalog for LLM (lightweight: id, name, description, capabilities only)
        agent_catalog = []
        agent_lookup = {}  # For quick lookup after LLM selection
        
        for agent in all_agents:
            agent_info = {
                "id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "rating": agent.rating,
                "price_per_call_usd": agent.price_per_call_usd
            }
            agent_catalog.append(agent_info)
            agent_lookup[agent.id] = agent
        
        # Use LLM to select agents for each task
        primary_llm = ChatGroq(model="llama-3.3-70b-versatile") if os.getenv("GROQ_API_KEY") else ChatCerebras(model="llama-3.3-70b")
        fallback_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct") if os.getenv("NVIDIA_API_KEY") else None
        
        # Build task list for the prompt
        tasks_info = []
        for task in parsed_tasks:
            tasks_info.append({
                "task_name": task.task_name,
                "task_description": task.task_description
            })
        
        # LLM prompt for semantic agent selection
        prompt = f'''
You are an expert at matching tasks to the most appropriate agents based on their capabilities.

**ORIGINAL USER REQUEST:**
"{original_prompt}"

**TASKS TO ASSIGN:**
{json.dumps(tasks_info, indent=2)}

**AVAILABLE AGENTS:**
{json.dumps(agent_catalog, indent=2)}

**INSTRUCTIONS:**
For each task, select ALL agents that could potentially handle it, ranked from best to worst match.

**CRITICAL RULES:**
1. **Respect Explicit User Preferences**: If the user explicitly mentions wanting to use a specific type of agent (e.g., "use browser automation", "use the browser agent", "automate with browser"), you MUST prioritize agents that match that preference.
2. **Semantic Matching**: Match based on the meaning and intent, not just keyword overlap. For example:
   - "browse a website" → browser automation agent (NOT just web search)
   - "search for information" → web search agent
   - "automate clicking on a page" → browser automation agent
3. **Capability Depth**: Prefer agents with more specific/relevant capabilities over generic ones.
4. **Include Multiple Candidates**: Include 2-4 candidate agents per task when possible, so the ranking step can make the final choice.

**OUTPUT FORMAT:**
Return a JSON object where keys are task names and values are arrays of agent IDs (ordered best to worst):
{{
    "task_name_1": ["best_agent_id", "second_best_agent_id", ...],
    "task_name_2": ["best_agent_id", ...]
}}

Only include agents that are genuinely capable of handling the task. If no agent can handle a task, use an empty array.
'''
        
        # Schema for LLM response
        class AgentSelectionResult(BaseModel):
            selections: Dict[str, List[str]] = Field(
                description="Map of task names to ordered list of agent IDs"
            )
        
        try:
            logger.info("Using LLM for semantic agent selection...")
            response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, AgentSelectionResult)
            
            # Process LLM selections
            for task in parsed_tasks:
                task_name = task.task_name
                selected_agent_ids = response.selections.get(task_name, [])
                
                if not selected_agent_ids:
                    logger.warning(f"LLM selected no agents for task '{task_name}'")
                    candidate_agents_map[task_name] = []
                    continue
                
                # Convert selected agent IDs to full AgentCard objects
                validated_agents = []
                for agent_id in selected_agent_ids:
                    agent = agent_lookup.get(agent_id)
                    if not agent:
                        logger.warning(f"Agent ID '{agent_id}' not found in lookup")
                        continue
                    
                    try:
                        agent_card = AgentCard(
                            id=agent.id,
                            owner_id=agent.owner_id,
                            name=agent.name,
                            description=agent.description,
                            capabilities=agent.capabilities,
                            price_per_call_usd=agent.price_per_call_usd,
                            status=agent.status.value if hasattr(agent.status, 'value') else agent.status,
                            rating=agent.rating,
                            public_key_pem=agent.public_key_pem,
                            agent_type=agent.agent_type if hasattr(agent, 'agent_type') else 'http_rest',
                            connection_config=agent.connection_config if hasattr(agent, 'connection_config') else None,
                            endpoints=[
                                EndpointDetail(
                                    endpoint=ep.endpoint,
                                    http_method=ep.http_method,
                                    description=ep.description,
                                    parameters=[
                                        EndpointParameterDetail(
                                            name=p.name,
                                            description=p.description,
                                            param_type=p.param_type,
                                            required=p.required,
                                            default_value=p.default_value
                                        ) for p in ep.parameters
                                    ]
                                ) for ep in agent.endpoints
                            ]
                        )
                        validated_agents.append(agent_card.model_dump(mode='json'))
                    except Exception as e:
                        logger.error(f"Failed to convert agent {agent.id} to AgentCard: {e}")
                        continue
                
                candidate_agents_map[task_name] = validated_agents
                logger.info(f"LLM selected {len(validated_agents)} agents for task '{task_name}': {[a['name'] for a in validated_agents]}")
            
            logger.info("LLM-based agent selection complete.")
            
        except Exception as e:
            logger.error(f"LLM-based agent selection failed: {e}. Falling back to text-based search.")
            # Fallback to simple text-based matching
            candidate_agents_map = await _fallback_text_search(parsed_tasks, all_agents, agent_lookup)
        
        # Check for tasks with no agents found
        for task in parsed_tasks:
            if not candidate_agents_map.get(task.task_name):
                error_feedback = (
                    f"The previous attempt to parse the prompt resulted in the task description "
                    f"'{task.task_description}', which was matched to the capability '{task.task_name}'. "
                    f"However, no agents were found that could perform this task. Please generate a new, "
                    f"more detailed and specific task description that better captures the user's intent."
                )
                logger.warning(f"No agents found for task '{task.task_name}'. Looping back to re-parse.")
                return {"candidate_agents": {}, "parsing_error_feedback": error_feedback}
        
        return {"candidate_agents": candidate_agents_map, "parsing_error_feedback": None}
        
    except Exception as e:
        logger.error(f"Agent directory search failed: {e}")
        return {"candidate_agents": {}, "parsing_error_feedback": f"Agent search failed: {str(e)}"}
        
    finally:
        db.close()


async def _fallback_text_search(parsed_tasks, all_agents, agent_lookup):
    """
    Fallback text-based search when LLM selection fails.
    Uses simple keyword matching on capabilities.
    """
    from models import AgentEndpoint
    candidate_agents_map = {}
    
    for task in parsed_tasks:
        task_name = task.task_name.lower()
        task_desc = task.task_description.lower()
        
        matched_agents = []
        for agent in all_agents:
            # Check if any capability matches the task
            capabilities_str = " ".join(agent.capabilities).lower() if agent.capabilities else ""
            description_str = (agent.description or "").lower()
            
            # Simple keyword matching
            if (task_name in capabilities_str or 
                task_name in description_str or
                any(cap.lower() in task_desc for cap in (agent.capabilities or []))):
                
                try:
                    agent_card = AgentCard(
                        id=agent.id,
                        owner_id=agent.owner_id,
                        name=agent.name,
                        description=agent.description,
                        capabilities=agent.capabilities,
                        price_per_call_usd=agent.price_per_call_usd,
                        status=agent.status.value if hasattr(agent.status, 'value') else agent.status,
                        rating=agent.rating,
                        public_key_pem=agent.public_key_pem,
                        agent_type=agent.agent_type if hasattr(agent, 'agent_type') else 'http_rest',
                        connection_config=agent.connection_config if hasattr(agent, 'connection_config') else None,
                        endpoints=[
                            EndpointDetail(
                                endpoint=ep.endpoint,
                                http_method=ep.http_method,
                                description=ep.description,
                                parameters=[
                                    EndpointParameterDetail(
                                        name=p.name,
                                        description=p.description,
                                        param_type=p.param_type,
                                        required=p.required,
                                        default_value=p.default_value
                                    ) for p in ep.parameters
                                ]
                            ) for ep in agent.endpoints
                        ]
                    )
                    matched_agents.append(agent_card.model_dump(mode='json'))
                except Exception as e:
                    logger.error(f"Failed to convert agent {agent.id} in fallback: {e}")
                    continue
        
        candidate_agents_map[task.task_name] = matched_agents
        logger.info(f"Fallback search found {len(matched_agents)} agents for task '{task.task_name}'")
    
    return candidate_agents_map


# Removed - now defined inline in rank_agents for enhanced reasoning

def rank_agents(state: State):
    """
    ENHANCED LLM RANKING: Uses LLM for intelligent agent selection with rich metadata.
    Optimized with Groq for speed/cost while maintaining quality.
    """
    parsed_tasks = state.get('parsed_tasks', [])
    logger.info(f"Ranking agents for tasks: {[t.task_name for t in parsed_tasks]}")
    
    if not parsed_tasks:
        logger.warning("No tasks to rank in rank_agents")
        return {"task_agent_pairs": []}
    
    # OPTIMIZATION: Use Groq for ranking (faster and cheaper than Cerebras for this task)
    ranking_llm = ChatGroq(model="openai/gpt-oss-120b") if os.getenv("GROQ_API_KEY") else ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    # Get user expectations for context
    user_expectations = state.get('user_expectations', {})
    
    # Build conversation history for context
    conversation_history = ""
    if messages := state.get('messages'):
        recent_messages = messages[-10:]  # Last 10 messages for context
        for msg in recent_messages:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    conversation_history += f"User: {msg.content}\n"
                elif msg.type == "ai":
                    conversation_history += f"Assistant: {msg.content}\n"
    
    final_selections = []
    for task in parsed_tasks:
        task_name = task.task_name
        
        # Rehydrate here. Convert dicts from state back into AgentCard objects.
        candidate_agent_dicts = state.get('candidate_agents', {}).get(task_name, [])
        candidate_agents = [AgentCard.model_validate(d) for d in candidate_agent_dicts]
        
        if not candidate_agents:
            continue

        if len(candidate_agents) == 1:
            primary_agent = candidate_agents[0]
            fallback_agents = []
        else:
            # ENHANCED: Provide rich metadata for intelligent ranking
            agents_metadata = []
            for agent in candidate_agents:
                agents_metadata.append({
                    'id': agent.id,
                    'name': agent.name,
                    'description': agent.description,
                    'capabilities': agent.capabilities,
                    'rating': agent.rating,
                    'price_per_call_usd': agent.price_per_call_usd,
                    'status': agent.status,
                    'endpoints_count': len(agent.endpoints)
                })
            
            conversation_context = f"\n**Conversation Context:**\n{conversation_history}\n" if conversation_history else ""
            
            prompt = f'''
            You are an expert at selecting the best agent for a given task based on multiple factors.
            {conversation_context}
            **Task Details:**
            - Task Name: "{task.task_name}"
            - Task Description: "{task.task_description}"
            - User Expectations: {json.dumps(user_expectations, indent=2) if user_expectations else "None specified"}

            **Available Agents:**
            {json.dumps(agents_metadata, indent=2)}

            **Ranking Criteria (in order of importance):**
            1. **Capability Match** (Most Important): How well does the agent's description and capabilities match the task?
            2. **Quality/Rating**: Higher rated agents are generally more reliable (scale 0-5)
            3. **Price**: Consider user's budget if specified, otherwise prefer reasonable pricing
            4. **Task Complexity**: Match agent sophistication to task complexity
            5. **Status**: Prefer active agents

            **Instructions:**
            - Rank ALL agents from best to worst
            - Consider trade-offs (e.g., "Agent A is slightly more expensive but has much better ratings")
            - If user specified budget/rating preferences, prioritize those
            - Provide brief reasoning for your top choice

            **Output Format:**
            {{
                "ranked_agent_ids": ["agent_id_1", "agent_id_2", ...],
                "reasoning": "Brief explanation of why the top agent was chosen"
            }}
            '''
            # Enhanced schema with reasoning
            class RankedAgentsWithReasoning(BaseModel):
                ranked_agent_ids: List[str]
                reasoning: str
            
            try:
                response = invoke_llm_with_fallback(ranking_llm, fallback_llm, prompt, RankedAgentsWithReasoning)
                ranked_agent_ids = response.ranked_agent_ids
                
                logger.info(f"LLM ranking reasoning for '{task_name}': {response.reasoning}")
                
                sorted_agents = sorted(candidate_agents, key=lambda agent: ranked_agent_ids.index(agent.id) if agent.id in ranked_agent_ids else float('inf'))
                
                primary_agent = sorted_agents[0]
                fallback_agents = sorted_agents[1:4]
                
            except Exception as e:
                logger.error(f"LLM agent ranking failed: {e}. Falling back to default ranking.")
                scored_agents = []
                prices = [agent.price_per_call_usd for agent in candidate_agents if agent.price_per_call_usd is not None]
                min_price, max_price = (min(prices), max(prices)) if prices else (0, 0)
                price_range = (max_price - min_price) if (max_price > min_price) else 1.0
                
                for agent in candidate_agents:
                    norm_rating = (agent.rating - 1) / 4.0 if agent.rating is not None else 0
                    norm_price = 1 - ((agent.price_per_call_usd - min_price) / price_range) if price_range > 0 and agent.price_per_call_usd is not None else 1.0
                    score = (0.6 * norm_rating) + (0.4 * norm_price)
                    scored_agents.append({"agent": agent, "score": score})
                
                sorted_agents_by_score = sorted(scored_agents, key=lambda x: x['score'], reverse=True)
                primary_agent = sorted_agents_by_score[0]['agent']
                fallback_agents = [item['agent'] for item in sorted_agents_by_score[1:4]]

        pair = TaskAgentPair(
            task_name=task_name,
            task_description=task.task_description,
            primary=primary_agent,
            fallbacks=fallback_agents
        )
        final_selections.append(pair)
    
    # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
    serializable_pairs = [p.model_dump(mode='json') for p in final_selections]

    logger.info("Agent ranking complete.")
    logger.debug(f"Final agent selections: {[p for p in serializable_pairs]}")
    return {"task_agent_pairs": serializable_pairs}

def plan_execution(state: State, config: RunnableConfig):
    '''
    Creates an initial execution plan or modifies an existing one if a replan is needed,
    and saves the result to a file.
    '''
    logger.info(f"🔍 PLAN_EXECUTION START: uploaded_files in state = {len(state.get('uploaded_files', []))} files")
    logger.info(f"🔍 PLAN_EXECUTION START: uploaded_files content = {state.get('uploaded_files', [])}")
    replan_reason = state.get("replan_reason")
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    output_state = {}

    if replan_reason:
        # --- REPLANNING MODE ---
        logger.info(f"Replanning initiated. Reason: {replan_reason}")
        
        # Get available agents with their details for intelligent replanning
        task_agent_pair_dicts = state.get('task_agent_pairs', [])
        available_agents_info = []
        if task_agent_pair_dicts:
            for pair_dict in task_agent_pair_dicts:
                pair = TaskAgentPair.model_validate(pair_dict)
                agent_info = {
                    "agent_id": pair.primary.id,
                    "agent_name": pair.primary.name,
                    "capabilities": pair.primary.capabilities[:5],  # Top 5 capabilities
                    "endpoints": [{"endpoint": ep.endpoint, "description": ep.description} for ep in pair.primary.endpoints[:3]]
                }
                available_agents_info.append(agent_info)
        
        all_capabilities, _ = get_all_capabilities()
        capabilities_str = ", ".join(all_capabilities)

        # Get recently completed tasks for context
        completed_tasks = state.get('completed_tasks', [])
        recent_tasks_summary = []
        for task in completed_tasks[-3:]:  # Last 3 tasks
            task_summary = {
                "task_name": task.get('task_name', 'Unknown'),
                "agent_used": task.get('agent_name', 'Unknown'),
                "endpoint_used": task.get('endpoint', 'Unknown'),
                "status": "success" if task.get('result') else "unknown"
            }
            recent_tasks_summary.append(task_summary)
        
        # Build conversation history for full context
        conversation_history = ""
        if messages := state.get('messages'):
            recent_messages = messages[-10:]  # Last 10 messages for context
            for msg in recent_messages:
                if hasattr(msg, 'type'):
                    if msg.type == "human":
                        conversation_history += f"User: {msg.content}\n"
                    elif msg.type == "ai":
                        conversation_history += f"Assistant: {msg.content}\n"
        
        prompt = f'''
        You are an expert autonomous planner. The current execution plan is stalled. Your task is to surgically insert a new task into the plan to resolve the issue.

        **Conversation History:**
        {conversation_history}
        
        **Reason for Replan:** "{replan_reason}"
        **Current Stalled Plan:** {json.dumps([task for batch in state.get('task_plan', []) for task in batch], indent=2)}
        **Recently Completed Tasks:** {json.dumps(recent_tasks_summary, indent=2)}
        **Full List of Available System Capabilities:** [{capabilities_str}]
        
        **Available Agents:**
        {json.dumps(available_agents_info, indent=2)}
        
        **Instructions:**
        1.  Read the `Conversation History` to understand the full context of what the user wants
        2.  Analyze the `Reason for Replan` to understand what's missing or what failed
        3.  Look at `Recently Completed Tasks` to see what was just done successfully
        4.  For continuation tasks (same document, same type of work), reuse the SAME agent AND endpoint
        5.  Create a new `PlannedTask` with proper agent assignment (primary field with agent_id, http_method, endpoint)
        6.  Return the entire modified plan conforming to the `ExecutionPlan` schema
        
        **Key Principles:**
        - Conversation context is your primary source of truth for understanding user intent
        - Recently completed tasks show you what's working - reuse successful patterns
        - For document editing continuations, the endpoint should match the previous successful edit
        - Think logically: if a document was just edited, continuing to edit it uses the same endpoint
        '''
        try:
            response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, ExecutionPlan)
            # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
            if response and hasattr(response, 'plan'):
                # CRITICAL FIX: Correct HTTP methods based on agent configuration
                # The LLM sometimes guesses wrong HTTP methods (e.g., GET instead of POST)
                # We must respect the agent's actual endpoint configuration
                for batch in response.plan:
                    for task in batch:
                        if task.primary:
                            # Find the agent configuration
                            agent_id = task.primary.id
                            matching_pair = next((p for p in task_agent_pair_dicts if TaskAgentPair.model_validate(p).primary.id == agent_id), None)
                            if matching_pair:
                                agent_card = TaskAgentPair.model_validate(matching_pair).primary
                                # Find the endpoint in the agent's configuration
                                endpoint_path = task.primary.endpoint
                                matching_endpoint = next((ep for ep in agent_card.endpoints if str(ep.endpoint) == str(endpoint_path)), None)
                                if matching_endpoint:
                                    # Override the HTTP method with the correct one from configuration
                                    correct_http_method = matching_endpoint.http_method
                                    if task.primary.http_method != correct_http_method:
                                        logger.warning(f"🔧 CORRECTING HTTP METHOD: LLM suggested {task.primary.http_method} for {endpoint_path}, but agent config requires {correct_http_method}")
                                        task.primary.http_method = correct_http_method
                
                serializable_plan = [[task.model_dump(mode='json') for task in batch] for batch in (response.plan or [])]
                
                # CRITICAL FIX: Update task_agent_pairs with new task names from replan
                # This ensures execute_batch can find matching agents for replanned tasks
                new_task_agent_pairs = []
                for batch in response.plan:
                    for task in batch:
                        if task.primary:
                            # Find the original agent card for this agent_id
                            original_pair = next(
                                (TaskAgentPair.model_validate(p) for p in task_agent_pair_dicts 
                                 if TaskAgentPair.model_validate(p).primary.id == task.primary.id), 
                                None
                            )
                            if original_pair:
                                # Create new pair with the replanned task name
                                new_pair = TaskAgentPair(
                                    task_name=task.task_name,
                                    task_description=task.task_description,
                                    primary=original_pair.primary,
                                    fallbacks=original_pair.fallbacks
                                )
                                new_task_agent_pairs.append(new_pair.model_dump(mode='json'))
                                logger.info(f"✅ Created new task_agent_pair for replanned task: '{task.task_name}' -> agent '{task.primary.id}'")
                
                # Combine existing pairs with new ones
                updated_task_agent_pairs = task_agent_pair_dicts + new_task_agent_pairs
                
                output_state = {
                    "task_plan": serializable_plan, 
                    "task_agent_pairs": updated_task_agent_pairs,
                    "replan_reason": None  # Clear the reason after replanning
                }
            else:
                # If response is None or doesn't have plan attribute, create a simple plan
                logger.warning("Replanning LLM response was invalid. Creating simple plan.")
                task_plan_dicts = state.get("task_plan", [])
                output_state = {"task_plan": task_plan_dicts, "replan_reason": None}
        except Exception as e:
            logger.error(f"Replanning failed: {e}. Falling back to asking user.")
            output_state = {
                "pending_user_input": True,
                "question_for_user": f"I tried to solve the issue of '{replan_reason}' but failed. Could you please provide the missing information directly?"
            }

    else:
        # --- INITIAL PLANNING MODE ---
        logger.info("Creating initial execution plan.")
        
        # Rehydrate here
        task_agent_pair_dicts = state.get('task_agent_pairs', [])
        if not task_agent_pair_dicts:
            return {"task_plan": []}
        task_agent_pairs = [TaskAgentPair.model_validate(d) for d in task_agent_pair_dicts]

        # Simplify the prompt for better compatibility with fallback LLM
        prompt = f'''
        You are an expert project planner. Convert tasks and their assigned agents into an executable plan.
        
        **CRITICAL RULE: Create EXACTLY ONE PlannedTask for EACH input task. Do NOT duplicate tasks!**
        
        Instructions:
        1. For each task in the input list, create EXACTLY ONE PlannedTask (no more, no less)
        2. Select the most appropriate endpoint from the primary agent's list
        3. Create an ExecutionStep with the agent id, http_method, and endpoint
        4. Do not generate a payload
        5. Group tasks that can run in parallel into the same batch
        6. Return a valid JSON object that conforms to the ExecutionPlan schema
        
        **EXAMPLE:**
        If input has 1 task, output should have 1 PlannedTask.
        If input has 3 tasks, output should have 3 PlannedTasks.
        
        **DO NOT CREATE DUPLICATE TASKS!**

        Tasks to Plan: {json.dumps([p.model_dump(mode='json') for p in task_agent_pairs], indent=2)}
        
        Remember: Create EXACTLY {len(task_agent_pairs)} PlannedTask(s) - one for each input task.
        '''
        try:
            response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, ExecutionPlan)
            logger.info(f"🔍 LLM RESPONSE DEBUG: response={response}, type={type(response)}, has_plan={hasattr(response, 'plan') if response else False}")
            if response and hasattr(response, 'plan'):
                logger.info(f"🔍 LLM RESPONSE PLAN: plan={response.plan}, plan_length={len(response.plan) if response.plan else 0}")
            else:
                logger.warning(f"⚠️ LLM RESPONSE INVALID: response is None or missing 'plan' attribute")
            # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
            if response and hasattr(response, 'plan') and response.plan:
                # CRITICAL FIX: Correct HTTP methods based on agent configuration (same as replanning)
                for batch in response.plan:
                    for task in batch:
                        if task.primary:
                            # Find the agent configuration
                            agent_id = task.primary.id
                            matching_pair = next((p for p in task_agent_pairs if p.primary.id == agent_id), None)
                            if matching_pair:
                                agent_card = matching_pair.primary
                                # Find the endpoint in the agent's configuration
                                endpoint_path = task.primary.endpoint
                                matching_endpoint = next((ep for ep in agent_card.endpoints if str(ep.endpoint) == str(endpoint_path)), None)
                                if matching_endpoint:
                                    # Override the HTTP method with the correct one from configuration
                                    correct_http_method = matching_endpoint.http_method
                                    if task.primary.http_method != correct_http_method:
                                        logger.warning(f"🔧 CORRECTING HTTP METHOD: LLM suggested {task.primary.http_method} for {endpoint_path}, but agent config requires {correct_http_method}")
                                        task.primary.http_method = correct_http_method
                
                # AUTO-INJECT PARAMETERS: Fill in parameters from uploaded files BEFORE serialization
                uploaded_files = state.get("uploaded_files", [])
                logger.info(f"🔍 DEBUG: uploaded_files count={len(uploaded_files)}, response.plan exists={response.plan is not None}, plan length={len(response.plan) if response.plan else 0}")
                
                if uploaded_files and response.plan:
                    logger.info(f"AUTO-INJECT: Processing {len(uploaded_files)} uploaded files for parameter injection")
                    for batch in response.plan:
                        for task in batch:
                            # Get the parsed task to access pre-extracted parameters
                            parsed_task = next((t for t in state.get('parsed_tasks', []) if t.task_name == task.task_name), None)
                            if parsed_task and parsed_task.parameters:
                                # Start with pre-extracted parameters
                                if not task.primary.payload:
                                    task.primary.payload = {}
                                task.primary.payload.update(parsed_task.parameters)
                                logger.info(f"Using pre-extracted parameters for '{task.task_name}': {task.primary.payload}")
                            elif not task.primary.payload:
                                task.primary.payload = {}
                            
                            # Auto-inject vector_store_path for document tasks
                            logger.info(f"🔍 Checking {len(uploaded_files)} files for vector_store_path injection")
                            
                            # Collect ALL vector store paths for documents
                            vector_store_paths = []
                            for file_obj in uploaded_files:
                                file_dict = file_obj if isinstance(file_obj, dict) else (file_obj.__dict__ if hasattr(file_obj, '__dict__') else {})
                                logger.info(f"🔍 File: {file_dict.get('file_name')}, type={file_dict.get('file_type')}, has_vector_store={bool(file_dict.get('vector_store_path'))}")
                                if file_dict.get('file_type') == 'document' and file_dict.get('vector_store_path'):
                                    vector_store_paths.append(file_dict['vector_store_path'])
                            
                            # Inject ALL vector store paths (not just first one)
                            if vector_store_paths:
                                if len(vector_store_paths) == 1:
                                    # Single document: use vector_store_path (backward compatible)
                                    if 'vector_store_path' not in task.primary.payload:
                                        task.primary.payload['vector_store_path'] = vector_store_paths[0]
                                        logger.info(f"✅ AUTO-INJECTED vector_store_path for '{task.task_name}': {vector_store_paths[0]}")
                                else:
                                    # Multiple documents: use vector_store_paths array
                                    if 'vector_store_paths' not in task.primary.payload:
                                        task.primary.payload['vector_store_paths'] = vector_store_paths
                                        logger.info(f"✅ AUTO-INJECTED {len(vector_store_paths)} vector_store_paths for '{task.task_name}'")
                                        logger.info(f"   Files: {[os.path.basename(p) for p in vector_store_paths]}")
                                    else:
                                        logger.info(f"⏭️ vector_store_paths already in payload, skipping injection")
                            else:
                                logger.info(f"⏭️ No document files with vector stores found")
                            
                            # Auto-inject query if not provided
                            if 'query' not in task.primary.payload:
                                original_prompt = state.get('original_prompt', '')
                                if any(word in original_prompt.lower() for word in ['what', 'summarize', 'summary', 'about', 'describe', 'analyze']):
                                    task.primary.payload['query'] = original_prompt
                                    logger.info(f"✅ AUTO-INJECTED query for '{task.task_name}': {original_prompt}")
                
                serializable_plan = [[task.model_dump(mode='json') for task in batch] for batch in (response.plan or [])]
                output_state = {
                    "task_plan": serializable_plan, 
                    "user_response": None
                }
                print(f"!!! PLAN_EXECUTION: LLM created plan with {len(serializable_plan)} batches, total tasks: {sum(len(batch) for batch in serializable_plan)} !!!")
                logger.info(f"LLM created plan with {len(serializable_plan)} batches")
            else:
                # If response is None or doesn't have plan attribute, create a simple plan
                print(f"!!! PLAN_EXECUTION: LLM response invalid, creating simple plan !!!")
                logger.warning("Planning LLM response was invalid. Creating simple plan.")
                # Create a simple plan with one batch containing all tasks
                from schemas import ExecutionStep
                import uuid
                simple_plan = []
                for pair in task_agent_pairs:
                    if pair.primary and pair.primary.endpoints:
                        # Take the first endpoint as default
                        endpoint = pair.primary.endpoints[0]
                        planned_task = PlannedTask(
                            task_name=pair.task_name,
                            task_description=pair.task_description,
                            primary=ExecutionStep(
                                id=str(uuid.uuid4()),
                                http_method=endpoint.http_method,
                                endpoint=endpoint.endpoint,
                                payload={}  # Empty payload for now, will be filled below
                            )
                        )
                        simple_plan.append(planned_task)
                
                # AUTO-INJECT PARAMETERS: Fill in parameters from uploaded files BEFORE serialization
                uploaded_files = state.get("uploaded_files", [])
                if uploaded_files and simple_plan:
                    logger.info(f"AUTO-INJECT (simple_plan path 1): Processing {len(uploaded_files)} uploaded files")
                    for task in simple_plan:
                        # Get the parsed task to access pre-extracted parameters
                        parsed_task = next((t for t in state.get('parsed_tasks', []) if t.task_name == task.task_name), None)
                        if parsed_task and parsed_task.parameters:
                            # Start with pre-extracted parameters
                            if not task.primary.payload:
                                task.primary.payload = {}
                            task.primary.payload.update(parsed_task.parameters)
                            logger.info(f"Using pre-extracted parameters for '{task.task_name}': {task.primary.payload}")
                        elif not task.primary.payload:
                            task.primary.payload = {}
                        
                        # Auto-inject vector_store_path for document tasks
                        for file_obj in uploaded_files:
                            file_dict = file_obj if isinstance(file_obj, dict) else (file_obj.__dict__ if hasattr(file_obj, '__dict__') else {})
                            if file_dict.get('file_type') == 'document' and file_dict.get('vector_store_path'):
                                if 'vector_store_path' not in task.primary.payload:
                                    task.primary.payload['vector_store_path'] = file_dict['vector_store_path']
                                    logger.info(f"✅ AUTO-INJECTED vector_store_path for '{task.task_name}': {file_dict['vector_store_path']}")
                                break
                        
                        # Auto-inject query if not provided
                        if 'query' not in task.primary.payload:
                            original_prompt = state.get('original_prompt', '')
                            if any(word in original_prompt.lower() for word in ['what', 'summarize', 'summary', 'about', 'describe', 'analyze']):
                                task.primary.payload['query'] = original_prompt
                                logger.info(f"✅ AUTO-INJECTED query for '{task.task_name}': {original_prompt}")
                
                if simple_plan:
                    serializable_plan = [[task.model_dump(mode='json') for task in simple_plan]]
                    output_state = {"task_plan": serializable_plan, "user_response": None}
                    print(f"!!! PLAN_EXECUTION: Created simple plan with {len(simple_plan)} tasks !!!")
                    logger.info("Created simplified plan as fallback")
                else:
                    print(f"!!! PLAN_EXECUTION: Failed to create simple plan - no tasks created !!!")
                    output_state = {"task_plan": [], "user_response": None}
        except Exception as e:
            logger.error(f"Initial planning failed: {e}")
            # Try a simpler approach as fallback
            try:
                # Create a simple plan with one batch containing all tasks
                from schemas import ExecutionStep
                import uuid
                simple_plan = []
                for pair in task_agent_pairs:
                    if pair.primary and pair.primary.endpoints:
                        # Take the first endpoint as default
                        endpoint = pair.primary.endpoints[0]
                        planned_task = PlannedTask(
                            task_name=pair.task_name,
                            task_description=pair.task_description,
                            primary=ExecutionStep(
                                id=str(uuid.uuid4()),
                                http_method=endpoint.http_method,
                                endpoint=endpoint.endpoint,
                                payload={}  # Empty payload for now, will be filled below
                            )
                        )
                        simple_plan.append(planned_task)
                
                # AUTO-INJECT PARAMETERS: Fill in parameters from uploaded files BEFORE serialization
                uploaded_files = state.get("uploaded_files", [])
                if uploaded_files and simple_plan:
                    logger.info(f"AUTO-INJECT (simple_plan path 2): Processing {len(uploaded_files)} uploaded files")
                    for task in simple_plan:
                        # Get the parsed task to access pre-extracted parameters
                        parsed_task = next((t for t in state.get('parsed_tasks', []) if t.task_name == task.task_name), None)
                        if parsed_task and parsed_task.parameters:
                            # Start with pre-extracted parameters
                            if not task.primary.payload:
                                task.primary.payload = {}
                            task.primary.payload.update(parsed_task.parameters)
                            logger.info(f"Using pre-extracted parameters for '{task.task_name}': {task.primary.payload}")
                        elif not task.primary.payload:
                            task.primary.payload = {}
                        
                        # Auto-inject vector_store_path for document tasks
                        for file_obj in uploaded_files:
                            file_dict = file_obj if isinstance(file_obj, dict) else (file_obj.__dict__ if hasattr(file_obj, '__dict__') else {})
                            if file_dict.get('file_type') == 'document' and file_dict.get('vector_store_path'):
                                if 'vector_store_path' not in task.primary.payload:
                                    task.primary.payload['vector_store_path'] = file_dict['vector_store_path']
                                    logger.info(f"✅ AUTO-INJECTED vector_store_path for '{task.task_name}': {file_dict['vector_store_path']}")
                                break
                        
                        # Auto-inject query if not provided
                        if 'query' not in task.primary.payload:
                            original_prompt = state.get('original_prompt', '')
                            if any(word in original_prompt.lower() for word in ['what', 'summarize', 'summary', 'about', 'describe', 'analyze']):
                                task.primary.payload['query'] = original_prompt
                                logger.info(f"✅ AUTO-INJECTED query for '{task.task_name}': {original_prompt}")
                
                if simple_plan:
                    serializable_plan = [[task.model_dump(mode='json') for task in simple_plan]]
                    output_state = {"task_plan": serializable_plan, "user_response": None}
                    logger.info("Created simplified plan as fallback")
                else:
                    output_state = {"task_plan": [], "user_response": None}
            except Exception as fallback_error:
                logger.error(f"Simplified plan creation also failed: {fallback_error}")
                output_state = {"task_plan": []}

    # Save the new or modified plan to the file system immediately.
    # We create a temporary state to pass the object version of the plan for readable file output.
    temp_state_for_saving = {**state, **output_state}
    if 'task_plan' in output_state and output_state['task_plan']:
        rehydrated_plan = [[PlannedTask.model_validate(task) for task in batch] for batch in output_state['task_plan']]
        temp_state_for_saving['task_plan'] = rehydrated_plan
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id:
        save_plan_to_file({**temp_state_for_saving, "thread_id": thread_id})
    else:
        logger.warning("No thread_id found in config, skipping plan save")
    
    # If planning mode is on, set flag to indicate approval is needed
    if state.get("planning_mode"):
        logger.info("=== PLAN EXECUTION: Planning mode ON. Setting needs_approval flag ===")
        print(f"!!! PLAN_EXECUTION: Setting needs_approval=True, pending_user_input=True !!!")
        output_state["needs_approval"] = True
        output_state["pending_user_input"] = True
        output_state["question_for_user"] = "Please review and approve the execution plan."
        logger.info(f"Plan execution output state: needs_approval={output_state['needs_approval']}, pending_user_input={output_state['pending_user_input']}")
    else:
        logger.info("=== PLAN EXECUTION: Planning mode OFF. No approval needed ===")
    
    print(f"!!! PLAN_EXECUTION COMPLETE: Returning task_plan with {len(output_state.get('task_plan', []))} batches !!!")
    logger.info(f"Plan execution complete. Returning {len(output_state.get('task_plan', []))} batches")
    
    return output_state


def pause_for_plan_approval(state: State, config: RunnableConfig):
    '''
    WebSocket-compatible approval checkpoint that pauses after plan creation.
    
    This allows users to review:
    - Parsed tasks
    - Selected agents with ratings
    - Execution plan with estimated costs
    - Total estimated cost
    
    The workflow pauses here and waits for user approval via WebSocket.
    Only pauses if planning_mode is enabled.
    '''
    # Check if planning mode is enabled
    planning_mode = state.get("planning_mode", False)
    plan_approved = state.get("plan_approved", False)
    
    logger.info(f"=== APPROVAL CHECKPOINT ENTRY: planning_mode={planning_mode}, plan_approved={plan_approved} ===")
    
    if not planning_mode:
        logger.info("=== APPROVAL CHECKPOINT: Planning mode disabled, skipping approval ===")
        return {}  # Skip approval, proceed directly to execution
    
    if plan_approved:
        logger.info("=== APPROVAL CHECKPOINT: Plan already approved, skipping approval ===")
        return {}  # Plan was already approved, don't ask again
    
    logger.info("=== APPROVAL CHECKPOINT: Planning mode enabled, pausing for user approval ===")
    
    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    
    # Get plan details
    task_plan = state.get("task_plan", [])
    task_agent_pairs = state.get("task_agent_pairs", [])
    
    # Calculate total estimated cost
    total_cost = 0.0
    task_count = 0
    for batch in task_plan:
        for task_dict in batch:
            task_count += 1
            if isinstance(task_dict, dict):
                primary_agent = task_dict.get('primary', {})
                cost = primary_agent.get('price_per_call_usd', 0.0)
                if cost:
                    total_cost += cost
    
    logger.info(f"Plan summary: {task_count} tasks, estimated cost: ${total_cost:.4f}")
    
    # Set state to indicate we're waiting for approval
    return {
        "pending_user_input": True,
        "question_for_user": f"Review the execution plan: {task_count} tasks will be executed with an estimated cost of ${total_cost:.4f}. Type 'approve' to proceed or 'cancel' to stop.",
        "approval_required": True,
        "estimated_cost": total_cost,
        "task_count": task_count
    }


def validate_plan_for_execution(state: State):
    '''
    Performs an advanced pre-flight check on the next task, now with full file
    context awareness to prevent premature pauses.
    '''
    print(f"!!! VALIDATION ENTRY !!!")
    logger.info("Performing dynamic validation of the execution plan...")
    
    # --- ARTIFACT INTEGRATION: Summarize completed tasks to reduce context size ---
    completed_tasks_for_context = state.get('completed_tasks', [])
    if ARTIFACT_INTEGRATION_ENABLED and completed_tasks_for_context:
        # Create summarized version of completed tasks for context
        summarized_tasks = []
        for task in completed_tasks_for_context:
            task_summary = {
                "task_name": task.get("task_name", "unknown"),
                "status": "completed"
            }
            # Extract key result info without full data
            result = task.get("result", {})
            if isinstance(result, str):
                task_summary["result_preview"] = result[:500] if len(result) > 500 else result
            elif isinstance(result, dict):
                # Keep only essential keys
                if "status" in result:
                    task_summary["result_status"] = result["status"]
                if "summary" in result:
                    task_summary["result_summary"] = result["summary"][:500]
                elif "task_summary" in result:
                    task_summary["result_summary"] = result["task_summary"][:500]
                # For browser results, extract key data
                if "extracted_data" in result:
                    extracted = result["extracted_data"]
                    if isinstance(extracted, dict):
                        task_summary["extracted_keys"] = list(extracted.keys())[:10]
                    elif isinstance(extracted, list):
                        task_summary["extracted_count"] = len(extracted)
            summarized_tasks.append(task_summary)
        completed_tasks_for_context = summarized_tasks
        logger.info(f"Summarized {len(completed_tasks_for_context)} completed tasks for validation context")
    
    # Rehydrate the plan
    task_plan_dicts = state.get("task_plan", [])
    if not task_plan_dicts or not task_plan_dicts[0]:
        logger.info("Plan is empty or complete. No validation needed.")
        return {"replan_reason": None, "pending_user_input": False}
    task_plan = [[PlannedTask.model_validate(batch_item) for batch_item in batch] for batch in task_plan_dicts]

    all_capabilities, _ = get_all_capabilities()
    capabilities_str = ", ".join(all_capabilities)
    
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    task_to_validate = task_plan[0][0]

    # Rehydrate the pairs
    task_agent_pair_dicts = state.get('task_agent_pairs', [])
    task_agent_pairs = [TaskAgentPair.model_validate(d) for d in task_agent_pair_dicts]

    task_agent_pair = next((p for p in task_agent_pairs if p.task_name == task_to_validate.task_name), None)
    if not task_agent_pair:
        logger.warning(f"Could not find matching task_agent_pair for task '{task_to_validate.task_name}'. Skipping validation.")
        return {"replan_reason": None, "pending_user_input": False}

    agent_card = task_agent_pair.primary
    selected_endpoint = next((ep for ep in agent_card.endpoints if str(ep.endpoint) == str(task_to_validate.primary.endpoint)), None)
    required_params = [p.name for p in selected_endpoint.parameters if p.required] if selected_endpoint else []

    if not required_params:
        logger.info(f"Task '{task_to_validate.task_name}' has no required parameters. Validation successful.")
        return {"replan_reason": None, "pending_user_input": False}

    # Rehydrate uploaded files
    uploaded_file_dicts = state.get("uploaded_files", [])
    uploaded_files_typed = []
    for f in uploaded_file_dicts:
        try:
            if isinstance(f, dict):
                uploaded_files_typed.append(FileObject.model_validate(f))
            else:
                # Handle case where files are already FileObject instances
                uploaded_files_typed.append(f)
        except Exception as e:
            logger.warning(f"Failed to validate file object: {e}")
            continue
    
    file_context = ""
    if uploaded_files_typed:
        file_details = []
        for file_obj in uploaded_files_typed:
            detail = f"- File Name: '{file_obj.file_name}', Type: {file_obj.file_type}, Path: '{file_obj.file_path}'"
            if file_obj.file_type == 'document' and hasattr(file_obj, 'vector_store_path') and file_obj.vector_store_path:
                detail += f", Vector Store Path: '{file_obj.vector_store_path}'"
            file_details.append(detail)
        
        file_context = f'''
        **Available File Context:**
        The user has uploaded files. Use their paths to fill the required parameters.
        {os.linesep.join(file_details)}
        ---
        '''

    # --- Create a formatted history of the conversation ---
    history = ""
    if messages := state.get('messages'):
        # Limit to the last few messages to keep the prompt concise
        for msg in messages[-20:]: # Using the last 20 messages as context
            if hasattr(msg, 'type') and msg.type == "human":
                history += f"Human: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history += f"AI: {msg.content}\n"

    prompt = f'''
    You are an intelligent execution validator. Your job is to determine if a task can run based on the available information.

    **Context:**
    - Original User Prompt: "{state['original_prompt']}"
    - Conversation History:
    {history}
    - Previously Completed Tasks: {json.dumps(completed_tasks_for_context, indent=2, default=str)}
    - Task to Validate: "{task_to_validate.task_description}"
    - Required Parameters for this Task: {required_params}
    {file_context}
    - All Available System Capabilities: [{capabilities_str}]

    **Your Decision Process:**
    1.  **Check Context:** Can all `Required Parameters` (e.g., 'image_path', 'vector_store_path', 'query') be filled using the `Original User Prompt`, `Conversation History`, `Previously Completed Tasks`, or the `Available File Context`? The file paths provided are the values you should use.
    
    2.  **Special Case - Document Tasks:** If the task involves documents (summarize, analyze, answer questions):
        - If 'vector_store_path' is required and files are uploaded with vector stores, use the vector_store_path from the file context.
        - If 'query' is required and the user asks about the document (e.g., "What is this about?", "Summarize this"), infer the query as "Provide a comprehensive summary" or use the user's exact question.
        - **CRITICAL:** If files with vector stores are available in the file context, the task is READY. Do NOT ask for paths that are already provided in the file context!
        - In this case, respond with `status: "ready"` because all information is available.
    
    3. **If YES (all parameters can be filled):** The task is ready to run. Respond with `status: "ready"` and `reasoning: null`.
    
    4.  **If NO:** Determine the root cause.
        a. **Can another agent find the missing info?** If a value is missing (e.g., a city name) but a capability like "perform web search and summarize" could find it, respond with `status: "replan_needed"` and a clear `reasoning` (e.g., "Missing coordinates for the city mentioned, which can be found via web search.").
        b. **Is user input the only way?** If the information is something only the user would know AND cannot be inferred from context, respond with `status: "user_input_required"` and a clear, direct `question` for the user.

    Respond in a valid JSON format conforming to the PlanValidationResult schema.
    '''
    
    try:
        validation = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, PlanValidationResult)
        logger.info(f"Validation result for task '{task_to_validate.task_name}': {validation.status}")

        if validation.status == "replan_needed":
            return {"replan_reason": validation.reasoning, "pending_user_input": False, "question_for_user": None}
        elif validation.status == "user_input_required":
            return {"pending_user_input": True, "question_for_user": validation.question, "replan_reason": None}
        
        # Default to "ready" status
        return {"replan_reason": None, "pending_user_input": False, "question_for_user": None}
    except Exception as e:
        logger.error(f"Plan validation LLM call failed: {e}. Assuming plan is ready to avoid stalling.")
        return {"replan_reason": None, "pending_user_input": False, "question_for_user": None}

async def execute_mcp_agent(planned_task: PlannedTask, agent_details: AgentCard, state: State, config: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute an MCP agent by calling its tools via the MCP protocol.
    
    Args:
        planned_task: The task to execute
        agent_details: The MCP agent details
        state: Current state
        config: Configuration including user_id
        payload: The parameters for the tool call
        
    Returns:
        Dictionary with task result
    """
    try:
        # Import MCP client
        from mcp import ClientSession
        import mcp.client.sse
        from models import AgentCredential
        from utils.encryption import decrypt
        from database import SessionLocal
        
        # Get user_id from config
        user_id = config.get("configurable", {}).get("user_id", "system")
        
        # Get MCP server URL from agent config
        url = agent_details.connection_config.get("url") if agent_details.connection_config else None
        if not url:
            return {
                "task_name": planned_task.task_name,
                "result": "Error: MCP agent has no URL configured"
            }
        
        # Fetch user credentials for this agent
        from services.credential_service import get_agent_credentials
        db = SessionLocal()
        try:
            credentials = get_agent_credentials(db, agent_details.id, user_id)
            
            # Build headers from credentials
            headers = {}
            if 'composio_api_key' in credentials:
                headers['x-api-key'] = credentials['composio_api_key']
            elif 'api_key' in credentials:
                headers['x-api-key'] = credentials['api_key']
            
            # Override URL if provided in credentials
            if 'gmail_mcp_url' in credentials:
                url = credentials['gmail_mcp_url']
            elif 'mcp_url' in credentials:
                url = credentials['mcp_url']
            
            logger.info(f"Connecting to MCP server: {url} with {len(headers)} auth headers")
            
            # Connect to MCP server
            sse_url = f"{url}/sse" if not url.endswith("/sse") else url
            
            # Use sse_client with headers parameter (newer MCP SDK version)
            async with mcp.client.sse.sse_client(sse_url, headers=headers, timeout=60.0) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize session
                    await session.initialize()
                    
                    # Call the tool
                    tool_name = planned_task.primary.endpoint
                    logger.info(f"Calling MCP tool: {tool_name} with payload: {payload}")
                    
                    result = await session.call_tool(tool_name, payload)
                    
                    # Extract result content
                    if hasattr(result, 'content') and result.content:
                        # MCP returns content as a list of content items
                        content_items = []
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_items.append(item.text)
                            elif hasattr(item, 'data'):
                                content_items.append(str(item.data))
                        
                        result_text = "\n".join(content_items) if content_items else str(result)
                    else:
                        result_text = str(result)
                    
                    logger.info(f"MCP tool call successful for task '{planned_task.task_name}'")
                    
                    return {
                        "task_name": planned_task.task_name,
                        "result": result_text,
                        "raw_response": result_text
                    }
                        
        finally:
            db.close()
            
    except Exception as e:
        error_msg = f"MCP agent call failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "task_name": planned_task.task_name,
            "result": error_msg,
            "status_code": 500
        }


async def run_agent(planned_task: PlannedTask, agent_details: AgentCard, state: State, config: Dict[str, Any], last_error: Optional[str] = None, force_execute: bool = False):
    '''
    OPTIMIZED EXECUTION: Builds the payload and runs a single agent.
    - Checks if pre-extracted parameters match required params (skips LLM if match)
    - Implements GET request caching
    - Semantic retries and rate limit handling
    - force_execute: If True, sets show_preview=False to actually execute (used after confirmation)
    '''
    logger.info(f"Running agent '{agent_details.name}' for task: '{planned_task.task_name}' (force_execute={force_execute})")
    
    # Construct full URL from base_url + endpoint
    endpoint_path = str(planned_task.primary.endpoint)
    # CRITICAL FIX: Safely access connection_config which may be None or not a dict
    connection_config = agent_details.connection_config
    if connection_config and isinstance(connection_config, dict):
        base_url = connection_config.get('base_url', '')
    else:
        base_url = ''
    
    # If endpoint is already a full URL, use it as-is; otherwise prepend base_url
    if endpoint_path.startswith('http://') or endpoint_path.startswith('https://'):
        endpoint_url = endpoint_path
    else:
        endpoint_url = f"{base_url}{endpoint_path}" if base_url else endpoint_path
    
    http_method = planned_task.primary.http_method.upper()
    
    # Match endpoint by path only (not full URL)
    selected_endpoint = next((ep for ep in agent_details.endpoints if str(ep.endpoint) == endpoint_path), None)
    
    # CRITICAL FIX: Override http_method with the correct one from agent config
    # LLM sometimes suggests GET when agent requires POST
    if selected_endpoint:
        correct_method = selected_endpoint.http_method.upper()
        if http_method != correct_method:
            logger.warning(f"🔧 HTTP METHOD CORRECTION: LLM suggested {http_method} for {endpoint_path}, but agent config requires {correct_method}")
            http_method = correct_method

    if not selected_endpoint:
        # ROBUST FALLBACK: For document agent, fall back to /edit endpoint with natural language
        if 'document' in agent_details.name.lower() or 'document' in agent_details.id.lower():
            logger.warning(f"Endpoint '{endpoint_path}' not found on document agent. Falling back to /edit with natural language.")
            # Find the /edit endpoint
            edit_endpoint = next((ep for ep in agent_details.endpoints if '/edit' in str(ep.endpoint)), None)
            if edit_endpoint:
                selected_endpoint = edit_endpoint
                # Override the endpoint path to use /edit
                endpoint_path = str(edit_endpoint.endpoint)
                endpoint_url = f"{agent_details.endpoints[0].endpoint.rsplit('/', 1)[0]}/edit" if '/' in str(agent_details.endpoints[0].endpoint) else endpoint_path
                logger.info(f"✅ Falling back to /edit endpoint with natural language instruction")
            else:
                error_msg = f"Critical Error: Could not find /edit endpoint on document agent '{agent_details.name}'."
                logger.error(error_msg)
                return {"task_name": planned_task.task_name, "result": error_msg}
        else:
            error_msg = f"Critical Error: Could not find endpoint details for '{endpoint_url}' on agent '{agent_details.name}'."
            logger.error(error_msg)
            return {"task_name": planned_task.task_name, "result": error_msg}

    # Initialize both primary and fallback LLMs for payload generation
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    failed_attempts = []
    
    # --- CRITICAL: Prepare file context FIRST before checking skip_llm_generation ---
    # This ensures file_id_mapping is available for injection into pre-extracted params
    file_context = ""
    file_id_mapping = {}
    uploaded_files = state.get("uploaded_files", [])
    
    if uploaded_files and FILE_INTEGRATION_ENABLED:
        try:
            # Prepare files for this specific agent (uploads if needed)
            file_id_mapping, file_context = await prepare_files_for_task(
                state=state,
                agent_details=agent_details,
                endpoint_path=endpoint_path,
                orchestrator_config=config  # Fixed: use orchestrator_config parameter name
            )
            logger.info(f"File integration: prepared {len(file_id_mapping)} file mappings for agent {agent_details.id}")
            logger.info(f"[FILE_ID_DEBUG] file_id_mapping contents: {file_id_mapping}")
        except Exception as e:
            logger.error(f"File integration error: {e}")
            # Fallback to basic file context
            # Normalize file paths to use forward slashes to avoid JSON escaping issues
            normalized_files = []
            for file_obj in uploaded_files:
                file_dict = file_obj if isinstance(file_obj, dict) else file_obj.__dict__
                normalized_dict = file_dict.copy()
                # Normalize paths to use forward slashes
                if 'file_path' in normalized_dict:
                    normalized_dict['file_path'] = normalized_dict['file_path'].replace('\\', '/')
                if 'vector_store_path' in normalized_dict:
                    normalized_dict['vector_store_path'] = normalized_dict['vector_store_path'].replace('\\', '/')
                normalized_files.append(normalized_dict)
            
            file_context = f'''
            **Available File Context:**
            The user has uploaded files. You MUST use the file information below to populate the required payload parameters (like 'image_path' or 'vector_store_path').
            ```json
            {json.dumps(normalized_files, indent=2)}
            ```
            ---
            '''
    elif uploaded_files:
        # File integration not enabled, use basic context
        # Normalize file paths to use forward slashes to avoid JSON escaping issues
        normalized_files = []
        for file_obj in uploaded_files:
            file_dict = file_obj if isinstance(file_obj, dict) else file_obj.__dict__
            normalized_dict = file_dict.copy()
            # Normalize paths to use forward slashes
            if 'file_path' in normalized_dict:
                normalized_dict['file_path'] = normalized_dict['file_path'].replace('\\', '/')
            if 'vector_store_path' in normalized_dict:
                normalized_dict['vector_store_path'] = normalized_dict['vector_store_path'].replace('\\', '/')
            normalized_files.append(normalized_dict)
        
        file_context = f'''
        **Available File Context:**
        The user has uploaded files. You MUST use the file information below to populate the required payload parameters (like 'image_path' or 'vector_store_path').
        ```json
        {json.dumps(normalized_files, indent=2)}
        ```
        ---
        '''
    
    # OPTIMIZATION: Check if we can skip LLM payload generation
    # PRIORITY 1: Use payload from PlannedTask.primary.payload (set by plan_execution node with auto-injections)
    # PRIORITY 2: Fall back to pre-extracted parameters from parsed_tasks
    pre_extracted_params = {}
    
    # Check if the planned task already has a payload (from plan_execution node)
    if hasattr(planned_task, 'primary') and planned_task.primary and hasattr(planned_task.primary, 'payload') and planned_task.primary.payload:
        pre_extracted_params = planned_task.primary.payload.copy()
        logger.info(f"✅ Using payload from PlannedTask.primary.payload: {list(pre_extracted_params.keys())}")
    else:
        # Fall back to parsed_tasks parameters
        for task in state.get('parsed_tasks', []):
            if task.task_name == planned_task.task_name:
                pre_extracted_params = task.parameters or {}
                break
    
    # AUTO-INJECT: Automatically add vector_store_path/paths for document tasks if files are uploaded
    if uploaded_files:
        # Check if endpoint accepts vector_store_path or vector_store_paths
        param_names = [p.name for p in selected_endpoint.parameters]
        has_single_path = 'vector_store_path' in param_names
        has_multiple_paths = 'vector_store_paths' in param_names
        
        # Collect all document vector store paths
        document_vector_stores = []
        for file_obj in uploaded_files:
            file_dict = file_obj if isinstance(file_obj, dict) else file_obj.__dict__
            if file_dict.get('file_type') == 'document' and file_dict.get('vector_store_path'):
                document_vector_stores.append(file_dict['vector_store_path'])
        
        if document_vector_stores:
            # If endpoint supports plural paths and we have multiple files, use that
            if has_multiple_paths and len(document_vector_stores) > 1:
                if 'vector_store_paths' not in pre_extracted_params:
                    pre_extracted_params['vector_store_paths'] = document_vector_stores
                    logger.info(f"AUTO-INJECTED vector_store_paths ({len(document_vector_stores)} files): {document_vector_stores}")
            # Otherwise use single path (first file)
            elif has_single_path:
                if 'vector_store_path' not in pre_extracted_params:
                    pre_extracted_params['vector_store_path'] = document_vector_stores[0]
                    logger.info(f"AUTO-INJECTED vector_store_path: {document_vector_stores[0]}")
    
    # AUTO-INJECT: Automatically add query for document tasks if not provided
    if 'query' in [p.name for p in selected_endpoint.parameters] and 'query' not in pre_extracted_params:
        # Infer query from the original prompt
        original_prompt = state.get('original_prompt', '')
        if any(word in original_prompt.lower() for word in ['what', 'summarize', 'summary', 'about', 'describe', 'analyze', 'scan', 'extract', 'list', 'find']):
            pre_extracted_params['query'] = original_prompt
            logger.info(f"AUTO-INJECTED query from original prompt: {original_prompt}")
    
    # NATURAL LANGUAGE FALLBACK: For /edit endpoint, use task description as instruction
    if '/edit' in endpoint_path and 'instruction' in [p.name for p in selected_endpoint.parameters]:
        if 'instruction' not in pre_extracted_params:
            # Use the task description as the natural language instruction
            pre_extracted_params['instruction'] = planned_task.task_description
            logger.info(f"✅ AUTO-INJECTED natural language instruction for /edit: {planned_task.task_description}")
        
        # Ensure file_path is present for document edits
        if 'file_path' not in pre_extracted_params and uploaded_files:
            # ✅ ENHANCED: Check for valid file_path (not None)
            for file_obj in uploaded_files:
                file_dict = file_obj if isinstance(file_obj, dict) else file_obj.__dict__
                if file_dict.get('file_type') == 'document':
                    candidate_path = file_dict.get('file_path')
                    # Validate candidate_path is not None/empty
                    if candidate_path and candidate_path.strip():
                        pre_extracted_params['file_path'] = candidate_path
                        logger.info(f"✅ AUTO-INJECTED file_path for /edit: {candidate_path}")
                        break
                    else:
                        logger.warning(f"⚠️ Found document file but file_path is None or empty: {file_dict}")
            
            # If still no file_path found, log detailed error
            if 'file_path' not in pre_extracted_params:
                logger.error(f"❌ AUTO-INJECTION FAILED for /edit: No valid document file_path found")
                logger.error(f"   uploaded_files count: {len(uploaded_files)}")
                doc_files = [f if isinstance(f, dict) else f.__dict__ for f in uploaded_files if (f if isinstance(f, dict) else f.__dict__).get('file_type') == 'document']
                logger.error(f"   document files: {doc_files}")
    
    # AUTO-INJECT: Spreadsheet file_id for spreadsheet agent endpoints
    # PRIORITY: Files uploaded in the current turn (is_current_turn=True) should be used first
    if 'file_id' in [p.name for p in selected_endpoint.parameters] and 'file_id' not in pre_extracted_params and uploaded_files:
        # First pass: look for current turn spreadsheet files
        current_turn_file = None
        fallback_file = None
        
        for file_obj in uploaded_files:
            file_dict = file_obj if isinstance(file_obj, dict) else file_obj.__dict__
            if file_dict.get('file_type') == 'spreadsheet' or file_dict.get('file_name', '').lower().endswith(('.csv', '.xlsx', '.xls')):
                file_id = file_dict.get('file_id') or file_dict.get('content_id')
                if file_id:
                    # Prioritize current turn files
                    if file_dict.get('is_current_turn'):
                        current_turn_file = file_id
                        logger.info(f"📎 Found CURRENT TURN spreadsheet file_id: {file_id}")
                        break  # Current turn file found, stop searching
                    elif not fallback_file:
                        # Keep as fallback if no current turn file found
                        fallback_file = file_id
        
        # Use current turn file if available, otherwise use fallback
        selected_file_id = current_turn_file or fallback_file
        if selected_file_id:
            pre_extracted_params['file_id'] = selected_file_id
            if current_turn_file:
                logger.info(f"✅ AUTO-INJECTED file_id for spreadsheet (CURRENT TURN): {selected_file_id}")
            else:
                logger.info(f"✅ AUTO-INJECTED file_id for spreadsheet (fallback): {selected_file_id}")
    
    # AUTO-INJECT: Instruction for execute_pandas endpoint
    if '/execute_pandas' in endpoint_path and 'instruction' in [p.name for p in selected_endpoint.parameters]:
        if 'instruction' not in pre_extracted_params:
            pre_extracted_params['instruction'] = planned_task.task_description
            logger.info(f"✅ AUTO-INJECTED instruction for /execute_pandas: {planned_task.task_description}")
    
    # AUTO-INJECT: Question for nl_query endpoint (similar to document agent query)
    if '/nl_query' in endpoint_path and 'question' in [p.name for p in selected_endpoint.parameters]:
        if 'question' not in pre_extracted_params:
            pre_extracted_params['question'] = planned_task.task_description
            logger.info(f"✅ AUTO-INJECTED question for /nl_query: {planned_task.task_description}")

    
    # Check if all required parameters are already extracted
    required_params = [p.name for p in selected_endpoint.parameters if p.required]
    
    # MULTI-DOCUMENT FIX: Check if vector_store_paths (plural) satisfies vector_store_path (singular) requirement
    def param_is_satisfied(param_name: str, params_dict: dict) -> bool:
        """Check if a parameter is satisfied, accounting for plural/singular variants."""
        if param_name in params_dict:
            return True
        # Special case: vector_store_paths (plural) satisfies vector_store_path (singular)
        if param_name == 'vector_store_path' and 'vector_store_paths' in params_dict:
            return True
        # Special case: vector_store_path (singular) can satisfy vector_store_paths (plural) for backward compat
        if param_name == 'vector_store_paths' and 'vector_store_path' in params_dict:
            return True
        return False
    
    params_match = all(param_is_satisfied(param, pre_extracted_params) for param in required_params)
    
    if params_match and pre_extracted_params:
        logger.info(f"OPTIMIZATION: All required parameters pre-extracted. Skipping LLM payload generation.")
        logger.info(f"Using parameters: {pre_extracted_params}")
        # Transform pre-extracted parameters to match expected types
        payload = transform_payload_types(pre_extracted_params, selected_endpoint.parameters)
        
        # Inject correct file_id if we have file mappings (even for pre-extracted params)
        if file_id_mapping and FILE_INTEGRATION_ENABLED:
            payload = inject_file_id_into_payload(
                payload=payload,
                content_id_mapping=file_id_mapping,
                endpoint_details=selected_endpoint,
                uploaded_files=uploaded_files
            )
        
        skip_llm_generation = True
    else:
        logger.info(f"Required params: {required_params}, Pre-extracted: {list(pre_extracted_params.keys())}")
        skip_llm_generation = False

    # --- Create a formatted history of the conversation ---
    history = ""
    if messages := state.get('messages'):
        # Limit to the last few messages to keep the prompt concise
        for msg in messages[-20:]: # Using the last 20 messages as context
            if hasattr(msg, 'type') and msg.type == "human":
                history += f"Human: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history += f"AI: {msg.content}\n"

    # This loop handles semantic retries (e.g., valid but empty/useless responses)
    for attempt in range(3):
        # OPTIMIZATION: Skip LLM generation if parameters already match
        if skip_llm_generation and attempt == 0:
            logger.info("Using pre-extracted parameters, skipping LLM payload generation.")
            # payload already set above
        else:
            # Need to generate payload via LLM
            failed_attempts_context = ""
            if failed_attempts:
                failed_attempts_str = "\n".join([f"- Payload: {att['payload']}\n  - Result: {att['result']}" for att in failed_attempts])
                failed_attempts_context = f'''
                IMPORTANT: Your previous attempt(s) failed because the agent returned empty or unsatisfactory results. Do NOT repeat the same mistakes. Analyze the following failed attempts and generate a NEW, MODIFIED payload.

                <failed_attempts>
                {failed_attempts_str}
                </failed_attempts>
                '''

            http_error_context = f"\nIMPORTANT: The last API call failed with a client error. Please correct the payload based on this feedback:\n<error>\n{last_error}\n</error>\n" if last_error else ""

            # Build parameter schema with explicit type requirements
            param_schema_str = "**Parameter Schema (STRICT TYPE REQUIREMENTS):**\n"
            for p in selected_endpoint.parameters:
                required_str = "REQUIRED" if p.required else "optional"
                type_example = ""
                if p.param_type == "array":
                    type_example = ' (MUST be a JSON array like ["value1", "value2"], even for single items like ["single@email.com"])'
                elif p.param_type == "string":
                    type_example = ' (MUST be a JSON string like "value")'
                elif p.param_type == "integer":
                    type_example = ' (MUST be a JSON number like 42)'
                elif p.param_type == "boolean":
                    type_example = ' (MUST be a JSON boolean: true or false)'
                
                param_schema_str += f"- {p.name}: {p.param_type}{type_example} - {required_str} - {p.description}\n"
            
            payload_prompt = f'''
            You are an expert at creating API requests. Your task is to generate a valid JSON payload for the following endpoint, based on all the provided context.

            Endpoint Description: "{selected_endpoint.description}"
            
            {param_schema_str}
            
            **CRITICAL TYPE RULES:**
            1. If param_type is "array", the value MUST be a JSON array [], even for single items
            2. If param_type is "string", the value MUST be a JSON string ""
            3. If param_type is "integer", the value MUST be a JSON number
            4. If param_type is "boolean", the value MUST be true or false (no quotes)
            
            High-Level Task: "{planned_task.task_description}"
            Conversation History:
            {history}
            Historical Context (previous task results): {json.dumps(_summarize_completed_tasks_for_context(state.get('completed_tasks', [])), indent=2, default=str)}
            {file_context}
            {http_error_context}
            {failed_attempts_context}
            
            Your response MUST be only the JSON payload object, with no extra text or markdown.
            Ensure all parameter types match the schema exactly.
            '''
            
            # Add exponential backoff for rate limit handling when generating payloads
            payload_generated = False
            payload_generation_error = None
            payload = {}  # Initialize payload to avoid unbound variable error
            for payload_attempt in range(5):  # Up to 5 attempts for payload generation
                try:
                    # Use the fallback wrapper for LLM calls
                    payload_str = invoke_llm_with_fallback(primary_llm, fallback_llm, payload_prompt, None).__str__()
                    cleaned_payload_str = strip_think_tags(payload_str)
                    json_str = extract_json_from_response(cleaned_payload_str)
                    if not json_str:
                        raise json.JSONDecodeError("No JSON found in LLM response", cleaned_payload_str, 0)
                    payload = json.loads(json_str)
                    
                    # CRITICAL FIX: Transform payload parameters to match expected types
                    # This fixes issues like "to" being a string instead of array
                    payload = transform_payload_types(payload, selected_endpoint.parameters)
                    
                    # NEW: Inject correct file_id if we have file mappings
                    logger.info(f"[INJECTION_CHECK] file_id_mapping={file_id_mapping}, FILE_INTEGRATION_ENABLED={FILE_INTEGRATION_ENABLED}")
                    if file_id_mapping and FILE_INTEGRATION_ENABLED:
                        logger.info(f"[INJECTION_CHECK] Calling inject_file_id_into_payload...")
                        payload = inject_file_id_into_payload(
                            payload=payload,
                            content_id_mapping=file_id_mapping,
                            endpoint_details=selected_endpoint,
                            uploaded_files=uploaded_files
                        )
                        logger.info(f"[INJECTION_CHECK] Payload after injection: {payload}")
                    else:
                        logger.warning(f"[INJECTION_CHECK] Skipping injection - file_id_mapping empty or FILE_INTEGRATION disabled")
                    
                    logger.info(f"LLM generated payload for task '{planned_task.task_name}': {payload}")
                    payload_generated = True
                    break  # Success, exit retry loop
                except Exception as e:
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    if "429" in error_str or "rate" in error_str or "queue_exceeded" in error_str or "high traffic" in error_str:
                        # Apply exponential backoff
                        wait_time = (2 ** payload_attempt) + (0.1 * payload_attempt)  # Exponential backoff with jitter
                        logger.warning(f"Rate limit hit during payload generation. Waiting {wait_time:.2f} seconds before retry {payload_attempt + 1}/5")
                        await asyncio.sleep(wait_time)
                        continue  # Retry
                    else:
                        # Non-rate limit error, don't retry
                        payload_generation_error = e
                        break
            
            if not payload_generated:
                error_msg = f"Error building payload for task '{planned_task.task_name}': {payload_generation_error}"
                logger.error(error_msg)
                return {"task_name": planned_task.task_name, "result": error_msg}

        # --- MCP EXECUTION PATH ---
        from models import AgentType, AgentCredential
        if agent_details.agent_type == AgentType.MCP_HTTP:
            logger.info(f"Executing MCP agent: {agent_details.name}")
            return await execute_mcp_agent(planned_task, agent_details, state, config, payload)
        
        # --- HTTP REST EXECUTION PATH ---
        # Load credentials from database (new unified system)
        user_id = config.get("configurable", {}).get("user_id", "system")
        
        from services.credential_service import get_credentials_for_headers
        db = SessionLocal()
        try:
            headers = get_credentials_for_headers(db, agent_details.id, user_id, agent_details.agent_type)
        finally:
            db.close()
        
        # Fallback to .env for backward compatibility (will be deprecated)
        if not headers:
            if api_key := os.getenv(f"{agent_details.id.upper().replace('-', '_')}_API_KEY"):
                headers["Authorization"] = f"Bearer {api_key}"
                logger.warning(f"Using .env API key for {agent_details.id} - please migrate to credential management UI")

        # Wait for agent to be ready if it's still starting
        agent_name = f"{agent_details.id}_agent"
        from main import wait_for_agent_ready, agent_status, agent_status_lock
        
        async with agent_status_lock:
            agent_info = agent_status.get(agent_name)
        
        if agent_info and agent_info['status'] == 'starting':
            agent_ready = await wait_for_agent_ready(agent_name, agent_info['port'], timeout=30.0)
            if not agent_ready:
                error_msg = f"Agent '{agent_details.name}' failed to start or timed out"
                logger.error(error_msg)
                return {"task_name": planned_task.task_name, "result": error_msg, "status_code": 503}
        
        async with httpx.AsyncClient() as client:
            try:
                # CRITICAL AUTO-INJECTION: For document agent, ensure vector_store paths are injected
                is_document_agent = 'document' in agent_details.id.lower() or 'document' in agent_details.name.lower()
                
                if is_document_agent and uploaded_files:
                    # Auto-inject query from original prompt if not present
                    if 'query' not in payload or not payload.get('query'):
                        # Extract meaningful query from task description or original prompt
                        task_desc = planned_task.task_description.lower()
                        original_prompt = state.get('original_prompt', '')
                        
                        if any(word in task_desc for word in ['scan', 'extract', 'list', 'find', 'get', 'analyze']):
                            payload['query'] = original_prompt
                            logger.info(f"🔧 AUTO-INJECT: query = '{original_prompt}' (from original_prompt)")
                        else:
                            payload['query'] = planned_task.task_description
                            logger.info(f"🔧 AUTO-INJECT: query = '{planned_task.task_description}' (from task)")
                    
                    # Auto-inject vector_store_path(s) from uploaded_files
                    doc_files = [f for f in uploaded_files if f.get('file_type') == 'document']
                    
                    if doc_files:
                        # Filter for valid vector_store_path values (not None, not empty)
                        vector_paths = []
                        for f in doc_files:
                            vs_path = f.get('vector_store_path')
                            if vs_path and vs_path.strip():
                                vector_paths.append(vs_path)
                            elif f.get('file_path') and f.get('file_path').strip():
                                # Fallback: use file_path if vector_store_path is None/empty
                                logger.warning(f"Document '{f.get('file_name')}' has None vector_store_path, using file_path: {f.get('file_path')}")
                                vector_paths.append(f['file_path'])
                            else:
                                logger.error(f"Document '{f.get('file_name')}' has both None vector_store_path AND file_path")
                        
                        if vector_paths:
                            if len(vector_paths) == 1:
                                # Single document - use singular parameter
                                if 'vector_store_path' not in payload or not payload.get('vector_store_path'):
                                    payload['vector_store_path'] = vector_paths[0]
                                    logger.info(f"🔧 AUTO-INJECT: vector_store_path = {vector_paths[0]}")
                            else:
                                # Multiple documents - use plural parameter
                                if 'vector_store_paths' not in payload or not payload.get('vector_store_paths'):
                                    payload['vector_store_paths'] = vector_paths
                                    logger.info(f"🔧 AUTO-INJECT: vector_store_paths = {vector_paths}")
                        else:
                            logger.error(f"AUTO-INJECTION FAILED: Found {len(doc_files)} documents but no valid paths. Files: {[f.get('file_name') for f in doc_files]}")
                
                # CRITICAL VALIDATION: Ensure required parameters are present
                # This prevents NoneType errors when agents expect file paths
                required_params = [p for p in selected_endpoint.parameters if p.required]
                missing_params = []
                none_params = []
                empty_params = []
                
                for param in required_params:
                    param_value = payload.get(param.name)
                    
                    if param.name not in payload:
                        missing_params.append(param.name)
                        logger.error(f"❌ MISSING REQUIRED PARAMETER: '{param.name}' not in payload")
                    elif param_value is None:
                        none_params.append(param.name)
                        logger.error(f"❌ NONE REQUIRED PARAMETER: '{param.name}' is None")
                    elif isinstance(param_value, str) and param_value.strip() == '':
                        empty_params.append(param.name)
                        logger.error(f"❌ EMPTY REQUIRED PARAMETER: '{param.name}' is empty string")
                    elif isinstance(param_value, list) and len(param_value) == 0:
                        empty_params.append(param.name)
                        logger.error(f"❌ EMPTY REQUIRED PARAMETER: '{param.name}' is empty list")
                
                # Log uploaded_files for debugging
                if missing_params or none_params or empty_params:
                    logger.error(f"📂 UPLOADED FILES COUNT: {len(uploaded_files)}")
                    for idx, uf in enumerate(uploaded_files):
                        logger.error(f"  File {idx+1}: name={uf.get('file_name')}, type={uf.get('file_type')}, path={uf.get('file_path')}, vector={uf.get('vector_store_path')}")
                
                if missing_params or none_params or empty_params:
                    all_invalid = missing_params + none_params + empty_params
                    error_details = []
                    if missing_params:
                        error_details.append(f"missing: {missing_params}")
                    if none_params:
                        error_details.append(f"None: {none_params}")
                    if empty_params:
                        error_details.append(f"empty: {empty_params}")
                    error_msg = f"Invalid parameters for {agent_details.name}/{endpoint_path}: {', '.join(error_details)}. Payload: {payload}"
                    logger.error(error_msg)
                    logger.error(f"📋 Full payload: {json.dumps(payload, indent=2)}")
                    logger.error(f"📋 Uploaded files: {json.dumps(uploaded_files, indent=2)}")
                    return {
                        "task_name": planned_task.task_name,
                        "result": f"Configuration error: {error_msg}",
                        "status_code": 400
                    }
                
                logger.info(f"✅ Payload validation passed. Required params present: {[p.name for p in required_params]}")
                logger.info(f"📤 Final payload: {json.dumps(payload, indent=2)}")
                logger.info(f"Calling agent '{agent_details.name}' at {endpoint_url}")
                
                # Check if this is a browser agent - pass thread_id for push-based streaming
                is_browser_agent = 'browser' in agent_details.id.lower() or 'browser' in agent_details.name.lower()
                
                if is_browser_agent and '/browse' in endpoint_url:
                    logger.info(f"🌐 Browser agent detected - enabling push-based streaming")
                    
                    # Get thread_id from config
                    thread_id = config.get('configurable', {}).get('thread_id', 'unknown')
                    
                    # Add thread_id as query parameter
                    endpoint_with_thread = f"{endpoint_url}?thread_id={thread_id}"
                    
                    logger.info(f"📡 Sending browser task with thread_id: {thread_id}")
                    
                    # Start browser task with longer timeout for vision-heavy tasks
                    browser_task = asyncio.create_task(
                        client.post(endpoint_with_thread, json=payload, headers=headers, timeout=600.0)
                    )
                    
                    # Wait for task to complete (browser agent will push updates directly)
                    logger.info(f"⏳ Waiting for browser task to complete (push-based streaming enabled)")
                    
                    # Task completed - get final result
                    response = await browser_task
                    response.raise_for_status()
                    result_data = response.json()
                    
                    logger.info(f"✅ Browser task completed (push-based streaming)")
                    
                    # Return the final result
                    return {
                        "task_name": planned_task.task_name,
                        "result": result_data.get('task_summary', 'Task completed'),
                        "raw_response": result_data,
                        "status_code": response.status_code,
                        "screenshot_files": result_data.get('screenshot_files', [])
                    }
                
                # Not a browser agent - use standard synchronous call
                # Increase timeout for document agents when files are present
                has_uploaded_files = state.get('uploaded_files', [])
                is_document_agent = 'document' in agent_details.name.lower() or 'analyze' in planned_task.task_name.lower()
                
                if has_uploaded_files and is_document_agent:
                    # Use very long timeout for document analysis with multiple files
                    timeout_seconds = 300.0  # 5 minutes for analyzing multiple large PDFs
                    logger.info(f"📄 Using extended timeout ({timeout_seconds}s) for document agent with {len(has_uploaded_files)} files")
                else:
                    timeout_seconds = 30.0  # Standard timeout
                
                # OPTIMIZATION: Check cache for GET requests
                if http_method == 'GET':
                    cache_key = f"{endpoint_url}:{json.dumps(payload, sort_keys=True)}"
                    cached_entry = get_request_cache.get(cache_key)
                    
                    if cached_entry and (time.time() - cached_entry['timestamp']) < GET_CACHE_DURATION_SECONDS:
                        logger.info(f"CACHE HIT: Using cached response for GET {endpoint_url}")
                        result = cached_entry['result']
                    else:
                        response = await client.get(endpoint_url, params=payload, headers=headers, timeout=timeout_seconds)
                        response.raise_for_status()
                        result = response.json()
                        
                        # Cache the result
                        get_request_cache[cache_key] = {
                            'result': result,
                            'timestamp': time.time()
                        }
                        logger.info(f"CACHE MISS: Cached response for GET {endpoint_url}")
                else:  # POST
                    # Check if agent expects form data instead of JSON
                    # Priority: endpoint-specific > agent connection_config > default (json)
                    use_form_data = False
                    
                    # First check endpoint-specific request_format (highest priority)
                    endpoint_format = getattr(selected_endpoint, 'request_format', None)
                    logger.info(f"[REQUEST_FORMAT_DEBUG] Endpoint: {endpoint_path}, selected_endpoint.request_format={endpoint_format}")
                    logger.info(f"[REQUEST_FORMAT_DEBUG] selected_endpoint type: {type(selected_endpoint)}, has request_format attr: {hasattr(selected_endpoint, 'request_format')}")
                    
                    if endpoint_format:
                        use_form_data = endpoint_format == 'form'
                        logger.info(f"Using endpoint-specific request_format: {endpoint_format} -> use_form_data={use_form_data}")
                    else:
                        # Fall back to agent connection_config
                        if connection_config and isinstance(connection_config, dict):
                            use_form_data = connection_config.get('request_format', 'json') == 'form'
                            logger.info(f"Using agent connection_config request_format: {connection_config.get('request_format')} -> use_form_data={use_form_data}")
                    
                    if use_form_data:
                        logger.info(f"Using form data for agent '{agent_details.name}'")
                        response = await client.post(endpoint_url, data=payload, headers=headers, timeout=timeout_seconds)
                    else:
                        logger.info(f"Using JSON data for agent '{agent_details.name}'")
                        response = await client.post(endpoint_url, json=payload, headers=headers, timeout=timeout_seconds)
                    
                    response.raise_for_status()
                    result = response.json()

                # **INTELLIGENT VALIDATION** for semantic failure
                is_result_empty = not result or (isinstance(result, list) and not result) or (isinstance(result, dict) and not any(result.values()))
                if is_result_empty:
                    logger.warning(f"Agent returned a successful but empty response. Retrying...")
                    failed_attempts.append({"payload": payload, "result": str(result)})
                    continue  # Continue to the next attempt in the loop
                
                # **CANVAS DISPLAY HANDLING** - Extract canvas_display from agent response
                canvas_display = None
                if isinstance(result, dict):
                    # Check at top level first
                    if 'canvas_display' in result:
                        canvas_display = result.get('canvas_display')
                    # Also check inside 'result' field (some agents nest it there)
                    elif 'result' in result and isinstance(result.get('result'), dict):
                        canvas_display = result.get('result', {}).get('canvas_display')
                    
                    if canvas_display:
                        # CRITICAL FIX: Only access .get() if canvas_display is not None
                        canvas_type = canvas_display.get('canvas_type') if canvas_display else None
                        requires_confirmation = canvas_display.get('requires_confirmation', False) if canvas_display else False
                        canvas_title = canvas_display.get('canvas_title') if canvas_display else None
                        logger.info(f"✅ Agent returned canvas display: type={canvas_type}, requires_confirmation={requires_confirmation}, title={canvas_title}")
                    else:
                        logger.info(f"ℹ️ Agent did not return canvas_display (checked both top level and nested)")
                
                logger.info(f"✅ Agent call successful for task '{planned_task.task_name}'. Payload used: show_preview={payload.get('show_preview', 'not set')}")
                return {
                    "task_name": planned_task.task_name, 
                    "result": result,
                    "canvas_display": canvas_display
                }
            
            except httpx.HTTPStatusError as e:
                # Handle rate limit errors from agents with exponential backoff
                if e.response.status_code == 429:
                    # Apply exponential backoff
                    wait_time = (2 ** attempt) + (0.1 * attempt)  # Exponential backoff with jitter
                    logger.warning(f"Rate limit hit for agent '{agent_details.name}'. Waiting {wait_time:.2f} seconds before retry {attempt + 1}/3")
                    await asyncio.sleep(wait_time)
                    continue  # Retry the agent call
                else:
                    error_msg = f"Agent call failed with status {e.response.status_code}: {e.response.text}"
                    return {"task_name": planned_task.task_name, "result": error_msg, "raw_response": e.response.text, "status_code": e.response.status_code}
            except httpx.RequestError as e:
                # Get detailed error information
                error_details = str(e) if str(e) else repr(e)
                error_type = type(e).__name__
                
                # Special handling for timeout errors
                if isinstance(e, httpx.ReadTimeout):
                    error_msg = f"Agent call timed out after {timeout_seconds}s. The agent at {endpoint_url} did not respond in time."
                elif isinstance(e, httpx.ConnectTimeout):
                    error_msg = f"Connection to agent timed out. Could not connect to {endpoint_url} within {timeout_seconds}s."
                elif isinstance(e, httpx.ConnectError):
                    error_msg = f"Connection failed. Could not reach agent at {endpoint_url}. Is the agent running?"
                else:
                    error_msg = f"Agent call failed with a network error: {error_type} - {error_details}"
                
                logger.error(f"Network error calling agent '{agent_details.name}': {error_msg}")
                logger.error(f"Endpoint: {endpoint_url}, Payload: {payload}")
                logger.error(f"Error details: {error_type} - {error_details}")
                
                return {"task_name": planned_task.task_name, "result": error_msg, "raw_response": error_details, "status_code": 500}
    
    # This block is reached only if all semantic retries in the loop fail
    final_error_msg = f"Agent returned empty or unsatisfactory results for task '{planned_task.task_name}' after {len(failed_attempts)} attempts."
    logger.error(final_error_msg)
    return {"task_name": planned_task.task_name, "result": final_error_msg, "status_code": 500}

async def execute_confirmed_task(state: State, config: RunnableConfig):
    '''
    Re-executes a task after user confirmation.
    Called when canvas_confirmation_action is set in state.
    '''
    confirmation_action = state.get('canvas_confirmation_action')
    confirmation_task_name = state.get('canvas_confirmation_task')
    pending_task = state.get('pending_confirmation_task')
    
    if not confirmation_action or not pending_task:
        logger.warning("execute_confirmed_task called but no confirmation data found")
        return {}
    
    logger.info(f"🔄 Executing confirmed task: {confirmation_task_name}")
    
    # Get the task details from pending_confirmation_task
    task_name = pending_task.get('task_name')
    agent_name = pending_task.get('agent_name')
    
    # Find the original task and agent from task_agent_pairs
    task_agent_pairs = state.get('task_agent_pairs', [])
    task_pair = None
    for pair in task_agent_pairs:
        if isinstance(pair, dict):
            if pair.get('task_name') == task_name:
                task_pair = TaskAgentPair.model_validate(pair)
                break
        elif hasattr(pair, 'task_name') and pair.task_name == task_name:
            task_pair = pair
            break
    
    if not task_pair:
        logger.error(f"Could not find task pair for confirmed task: {task_name}")
        return {
            "canvas_confirmation_action": None,
            "canvas_confirmation_task": None,
            "pending_confirmation": False,
            "pending_confirmation_task": None
        }
    
    # Create a PlannedTask from the task_pair
    planned_task = PlannedTask(
        task_name=task_pair.task_name,
        task_description=task_pair.task_description,
        primary_agent_id=task_pair.primary.id,
        fallback_agent_ids=[fb.id for fb in task_pair.fallbacks] if task_pair.fallbacks else []
    )
    
    # Execute the task with show_preview=False (actually execute)
    logger.info(f"📧 Executing task '{task_name}' with agent '{agent_name}' (confirmed)")
    result = await run_agent(planned_task, task_pair.primary, state, config, force_execute=True)
    logger.info(f"📊 Task execution result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
    logger.info(f"📊 Canvas display in result: {'present' if result.get('canvas_display') else 'missing'}")
    
    # Add to completed tasks
    completed_tasks = state.get('completed_tasks', [])
    completed_tasks.append(CompletedTask(
        task_name=task_name,
        result=result.get('result', {}),
        raw_response=result.get('raw_response', {})
    ))
    
    logger.info(f"✅ Confirmed task '{task_name}' executed successfully")
    
    # **AUTO-DISPLAY**: Check if the agent already returned a canvas display (e.g., from /edit endpoint)
    canvas_display = result.get('canvas_display')
    logger.info(f"📊 Confirmed task result: canvas_display={'present' if canvas_display else 'missing'}")
    
    # If no canvas display from agent but this is a document edit, fetch the display
    file_path = pending_task.get('file_path')
    if not canvas_display and file_path and agent_name == 'document_analysis_agent':
        logger.info(f"🔄 Auto-displaying updated document after edit: {file_path}")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                display_response = await client.post(
                    "http://localhost:8070/display",
                    json={"file_path": file_path}
                )
                if display_response.status_code == 200:
                    display_result = display_response.json()
                    canvas_display = display_result.get("canvas_display")
                    logger.info(f"✅ Fetched display for updated document: {file_path}")
                else:
                    logger.error(f"❌ Failed to fetch display: status={display_response.status_code}")
        except Exception as display_err:
            logger.error(f"❌ Failed to auto-display updated document: {display_err}", exc_info=True)
    
    # If we have a canvas display (either from agent or fetched), return it
    if canvas_display:
        logger.info(f"📄 Returning canvas display after confirmed task: type={canvas_display.get('canvas_type')}, title={canvas_display.get('canvas_title')}")
        return {
            "completed_tasks": [task.model_dump() if hasattr(task, 'model_dump') else task for task in completed_tasks],
            "canvas_confirmation_action": None,
            "canvas_confirmation_task": None,
            "pending_confirmation": False,
            "pending_confirmation_task": None,
            "has_canvas": True,
            "canvas_content": canvas_display.get('canvas_content'),
            "canvas_data": canvas_display.get('canvas_data'),
            "canvas_type": canvas_display.get('canvas_type'),
            "canvas_title": canvas_display.get('canvas_title')
        }
    else:
        logger.warning(f"⚠️ No canvas display available after confirmed task '{task_name}'")
    
    # No canvas display - clear confirmation state and return
    return {
        "completed_tasks": [task.model_dump() if hasattr(task, 'model_dump') else task for task in completed_tasks],
        "canvas_confirmation_action": None,
        "canvas_confirmation_task": None,
        "pending_confirmation": False,
        "pending_confirmation_task": None,
        "has_canvas": False,
        "canvas_content": None,
        "canvas_data": None
    }


async def execute_batch(state: State, config: RunnableConfig):
    '''Executes a single batch of tasks from the plan.'''
    print(f"!!! EXECUTE_BATCH: Starting execution !!!")
    
    # PHASE 5: Enhanced uploaded_files debugging
    uploaded_files = state.get('uploaded_files', [])
    print(f"!!! EXECUTE_BATCH: uploaded_files count = {len(uploaded_files)} !!!")
    if uploaded_files:
        for idx, uf in enumerate(uploaded_files):
            print(f"!!!   File {idx+1}: name={uf.get('file_name')}, type={uf.get('file_type')}, path={uf.get('file_path')}, vector={uf.get('vector_store_path')} !!!")
    else:
        print(f"!!! EXECUTE_BATCH: NO FILES IN STATE - This may cause auto-injection failures !!!")
    
    # Rehydrate the plan
    task_plan_dicts = state.get('task_plan', [])
    if not task_plan_dicts:
        logger.info("No task plan to execute.")
        return {}
    task_plan = [[PlannedTask.model_validate(d) for d in batch] for batch in task_plan_dicts]

    current_batch_plan = task_plan[0]
    remaining_plan_objects = task_plan[1:]
    logger.info(f"Executing batch of {len(current_batch_plan)} tasks.")
    
    # Initialize task events list for real-time status tracking
    # Use a list that can be modified by nested async functions
    task_events = []
    
    # Get callback for real-time event emission (if available in config)
    task_event_callback = config.get("configurable", {}).get("task_event_callback")
    
    # Rehydrate the pairs
    task_agent_pair_dicts = state.get('task_agent_pairs', [])
    task_agent_pairs = [TaskAgentPair.model_validate(d) for d in task_agent_pair_dicts]
    task_agent_pairs_map = {pair.task_name: pair for pair in task_agent_pairs}
    
    async def try_task_with_fallbacks(planned_task: PlannedTask):
        nonlocal task_events  # Ensure we're modifying the outer scope's task_events
        
        # ROBUST FIX: Check if PlannedTask already has agent info embedded
        if hasattr(planned_task, 'primary') and planned_task.primary:
            # Task has embedded agent info - use it directly
            logger.info(f"✅ Using embedded agent info from PlannedTask for '{planned_task.task_name}'")
            # Get the full agent details from the execution step
            agent_id = planned_task.primary.id
            # Find the full agent card from task_agent_pairs
            original_task_pair = None
            for pair in task_agent_pairs:
                if pair.primary.id == agent_id:
                    original_task_pair = pair
                    logger.info(f"✅ Found matching agent card for embedded agent '{agent_id}'")
                    break
            
            # If not found in pairs, this is a problem - fall through to lookup
            if not original_task_pair:
                logger.warning(f"Could not find agent card for embedded agent '{agent_id}', falling back to lookup")
        else:
            original_task_pair = None
        
        if not original_task_pair:
            # Fall back to lookup in task_agent_pairs_map
            original_task_pair = task_agent_pairs_map.get(planned_task.task_name)
            
            # CONTINUATION TASK FIX: If exact match fails, try to find a matching agent
            if not original_task_pair:
                logger.warning(f"Could not find exact task pair for '{planned_task.task_name}'. Attempting intelligent match...")
                
                # Try to match by agent name in the task description or by looking at recent successful tasks
                matched_pair = None
                
                # Strategy 0 (MOST RELIABLE): Match by agent_id from PlannedTask.primary
                # This is populated by the LLM during replan and is the most reliable for replanned tasks
                if hasattr(planned_task, 'primary') and planned_task.primary and hasattr(planned_task.primary, 'id'):
                    target_agent_id = planned_task.primary.id
                    for pair in task_agent_pairs:
                        if pair.primary.id == target_agent_id:
                            matched_pair = pair
                            logger.info(f"✅ Matched task '{planned_task.task_name}' to agent '{target_agent_id}' by primary.id (replan match)")
                            break
                
                # Strategy 1: Check if any task pair's agent matches the task description
                if not matched_pair:
                    task_desc_lower = planned_task.task_description.lower()
                    for pair in task_agent_pairs:
                        agent_name_lower = pair.primary.name.lower()
                        # Check if agent name or capabilities match the task
                        if agent_name_lower in task_desc_lower or any(cap.lower() in task_desc_lower for cap in pair.primary.capabilities[:5]):
                            matched_pair = pair
                            logger.info(f"✅ Matched task to agent '{pair.primary.name}' by description")
                            break
                
                # Strategy 2: If still no match, use the most recent successful task's agent (for continuation tasks)
                if not matched_pair and state.get('completed_tasks'):
                    recent_tasks = state['completed_tasks'][-5:]  # Last 5 tasks
                    for recent_task in reversed(recent_tasks):
                        # Check if recent task was successful and used document agent
                        if 'document' in recent_task.get('task_name', '').lower():
                            # Find the document agent in task_agent_pairs
                            for pair in task_agent_pairs:
                                if 'document' in pair.primary.name.lower():
                                    matched_pair = pair
                                    logger.info(f"✅ Matched continuation task to document agent from recent history")
                                    break
                            if matched_pair:
                                break
                
                if matched_pair:
                    original_task_pair = matched_pair
                else:
                    error_msg = f"Could not find original task pair for '{planned_task.task_name}' to get fallbacks."
                    logger.error(error_msg)
                    return {"task_name": planned_task.task_name, "result": error_msg}

        agents_to_try = [original_task_pair.primary] + original_task_pair.fallbacks
        final_error_result = None
        
        # EMIT: Task started event
        import time
        task_start_time = time.time()
        logger.info(f"🚀 Task started: '{planned_task.task_name}' with agent '{agents_to_try[0].name}'")
        
        # Record task started event
        started_event = {
            "event_type": "task_started",
            "task_name": planned_task.task_name,
            "task_description": planned_task.task_description,
            "agent_name": agents_to_try[0].name,
            "timestamp": time.time()
        }
        task_events.append(started_event)
        logger.info(f"✅ Added task_started event for '{planned_task.task_name}', total events: {len(task_events)}")
        
        # Emit event in real-time if callback available
        if task_event_callback:
            try:
                await task_event_callback(started_event)
                logger.info(f"📡 Streamed task_started event for '{planned_task.task_name}'")
            except Exception as e:
                logger.error(f"Failed to stream task_started event: {e}")
        
        for agent_to_try in agents_to_try:
            max_retries = 3 if agent_to_try.id == original_task_pair.primary.id else 1
            last_error = None
            for i in range(max_retries):
                logger.info(f"Attempting task '{planned_task.task_name}' with agent '{agent_to_try.name}' (Attempt {i+1})...")
                
                # The state object is passed directly to run_agent
                task_result = await run_agent(planned_task, agent_to_try, state, config, last_error=last_error)
                
                result_data = task_result.get('result', {})
                is_error = isinstance(result_data, str) and "Error:" in result_data
                
                if not is_error:
                    # EMIT: Task completed event
                    task_end_time = time.time()
                    execution_time = round(task_end_time - task_start_time, 2)
                    logger.info(f"✅ Task completed: '{planned_task.task_name}' in {execution_time}s with agent '{agent_to_try.name}'")
                    
                    # Record task completed event
                    completed_event = {
                        "event_type": "task_completed",
                        "task_name": planned_task.task_name,
                        "agent_name": agent_to_try.name,
                        "execution_time": execution_time,
                        "timestamp": time.time()
                    }
                    task_events.append(completed_event)
                    logger.info(f"✅ Added task_completed event for '{planned_task.task_name}', total events: {len(task_events)}")
                    
                    # Emit event in real-time if callback available
                    if task_event_callback:
                        try:
                            await task_event_callback(completed_event)
                            logger.info(f"📡 Streamed task_completed event for '{planned_task.task_name}'")
                        except Exception as e:
                            logger.error(f"Failed to stream task_completed event: {e}")
                    
                    # Add execution metadata to result
                    task_result['execution_time'] = execution_time
                    task_result['agent_used'] = agent_to_try.name
                    return task_result
                
                final_error_result = task_result
                raw_response = task_result.get('raw_response', 'No raw response available.')
                logger.warning(f"Agent '{agent_to_try.name}' failed for task '{planned_task.task_name}'. Error: {result_data}")
                
                status_code = task_result.get("status_code")
                if isinstance(status_code, int) and 400 <= status_code < 500:
                    last_error = raw_response
                    logger.warning(f"Client error for agent '{agent_to_try.name}', task '{planned_task.task_name}': {raw_response}")
                else:
                    logger.error(f"Server error or other issue for agent '{agent_to_try.name}', task '{planned_task.task_name}': {raw_response}")
                    break
        
        # EMIT: Task failed event
        task_end_time = time.time()
        execution_time = round(task_end_time - task_start_time, 2)
        logger.error(f"❌ Task failed: '{planned_task.task_name}' after {execution_time}s - all agents exhausted")
        
        # Record task failed event
        failed_event = {
            "event_type": "task_failed",
            "task_name": planned_task.task_name,
            "execution_time": execution_time,
            "error": str(final_error_result.get('result', 'Unknown error')) if final_error_result else 'Unknown error',
            "timestamp": time.time()
        }
        task_events.append(failed_event)
        logger.error(f"✅ Added task_failed event for '{planned_task.task_name}', total events: {len(task_events)}")
        
        # Emit event in real-time if callback available
        if task_event_callback:
            try:
                await task_event_callback(failed_event)
                logger.error(f"📡 Streamed task_failed event for '{planned_task.task_name}'")
            except Exception as e:
                logger.error(f"Failed to stream task_failed event: {e}")
        
        if final_error_result:
            final_error_result['execution_time'] = execution_time
            final_error_result['status'] = 'failed'
        
        return final_error_result

    batch_results = await asyncio.gather(*(try_task_with_fallbacks(planned_task) for planned_task in current_batch_plan))
    
    print(f"!!! EXECUTE_BATCH: Got {len(batch_results)} results !!!")
    logger.info(f"Got {len(batch_results)} batch results")
    
    # Get thread_id for artifact storage
    thread_id_for_artifacts = config.get("configurable", {}).get("thread_id") if config else None
    
    completed_tasks_with_desc = []
    canvas_displays = []  # Collect canvas displays from agents
    
    for res in batch_results:
        task_name = res['task_name']
        result_preview = str(res.get('result', {}))[:200]
        execution_time = res.get('execution_time', 0)
        agent_used = res.get('agent_used', 'Unknown')
        status = res.get('status', 'completed')
        
        print(f"!!! EXECUTE_BATCH: Task '{task_name}' {status} in {execution_time}s with {agent_used} !!!")
        logger.info(f"Task '{task_name}' {status} in {execution_time}s - Result preview: {result_preview}")
        
        # --- CANVAS DISPLAY HANDLING ---
        logger.info(f"🔍 Checking for canvas display in task '{task_name}': has_canvas_display={('canvas_display' in res)}, result_type={type(res.get('result'))}")
        canvas_display = res.get('canvas_display')
        
        # Check if canvas_display is nested inside result (some agents return it this way)
        if not canvas_display and isinstance(res.get('result'), dict):
            logger.info(f"🔍 Checking nested canvas_display in result for task '{task_name}'")
            canvas_display = res.get('result', {}).get('canvas_display')
            if canvas_display:
                logger.info(f"📊 Found canvas display nested in result for task '{task_name}'")
        
        if canvas_display:
            logger.info(f"📊 Canvas display received from task '{task_name}': type={canvas_display.get('canvas_type')}, title={canvas_display.get('canvas_title')}, requires_confirmation={canvas_display.get('requires_confirmation')}")
            canvas_displays.append({
                'task_name': task_name,
                'agent_name': agent_used,
                **canvas_display
            })
            
            # Check if confirmation is required (check both canvas_display and result level)
            requires_conf = canvas_display.get('requires_confirmation') or (isinstance(res.get('result'), dict) and res.get('result', {}).get('requires_confirmation'))
            if requires_conf:
                logger.info(f"⏸️  Task '{task_name}' requires user confirmation - pausing execution")
                # Store confirmation state for this task
                # This will be handled by the orchestrator's confirmation flow
        else:
            logger.info(f"ℹ️ No canvas display from task '{task_name}'")
        
        # --- ARTIFACT INTEGRATION: Compress large task results ---
        task_result = res.get('result', {})
        raw_response = res.get('raw_response', {})
        
        if ARTIFACT_INTEGRATION_ENABLED and artifact_hooks and thread_id_for_artifacts:
            try:
                # Check if result is large enough to compress
                result_size = len(json.dumps(task_result, default=str))
                if result_size > 2000:  # Compress results > 2KB
                    compressed = await artifact_hooks.on_task_complete(task_name, task_result, thread_id_for_artifacts)
                    if '_artifact_ref' in compressed:
                        logger.info(f"Compressed task result '{task_name}' to artifact ({result_size} bytes)")
                        task_result = compressed
            except Exception as artifact_err:
                logger.warning(f"Failed to compress task result: {artifact_err}")
        
        completed_tasks_with_desc.append(CompletedTask(
            task_name=task_name,
            result=task_result,
            raw_response=raw_response
        ))

    completed_tasks = state.get('completed_tasks', []) + completed_tasks_with_desc
    print(f"!!! EXECUTE_BATCH: Total completed tasks: {len(completed_tasks)}, Latest: {len(completed_tasks_with_desc)} !!!")
    logger.info(f"Batch execution complete. Total completed: {len(completed_tasks)}, Latest: {len(completed_tasks_with_desc)}")
    
    # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
    remaining_plan_dicts = [[task.model_dump(mode='json') for task in batch] for batch in remaining_plan_objects]

    # Convert CompletedTask objects to dicts for proper serialization
    completed_tasks_dicts = [task.model_dump() if hasattr(task, 'model_dump') else task for task in completed_tasks]
    latest_completed_tasks_dicts = [task.model_dump() if hasattr(task, 'model_dump') else task for task in completed_tasks_with_desc]

    # Check if any task requires confirmation
    requires_confirmation = any(cd.get('requires_confirmation') for cd in canvas_displays)
    pending_confirmation_task = None
    
    print(f"!!! EXECUTE_BATCH: canvas_displays count={len(canvas_displays)}, requires_confirmation={requires_confirmation} !!!")
    logger.info(f"Canvas displays collected: {len(canvas_displays)}, requires_confirmation: {requires_confirmation}")
    
    if requires_confirmation:
        # Find the task that requires confirmation
        for cd in canvas_displays:
            if cd.get('requires_confirmation'):
                pending_confirmation_task = {
                    'task_name': cd.get('task_name'),
                    'agent_name': cd.get('agent_name'),
                    'canvas_display': cd
                }
                print(f"!!! EXECUTE_BATCH: ⏸️  Setting pending_confirmation=True for task: {cd.get('task_name')} !!!")
                logger.info(f"⏸️  Execution paused - waiting for confirmation on task: {cd.get('task_name')}")
                break
    
    output_state = {
        "task_plan": remaining_plan_dicts,
        "completed_tasks": completed_tasks_dicts,
        "latest_completed_tasks": latest_completed_tasks_dicts,
        "task_events": task_events,  # Include task status events for real-time updates
        "canvas_displays": canvas_displays,  # Include canvas displays from agents
        "pending_confirmation": requires_confirmation,
        "pending_confirmation_task": pending_confirmation_task
    }
    
    # If we have canvas displays, update the canvas state with the latest one
    if canvas_displays:
        latest_canvas = canvas_displays[-1]  # Use the most recent canvas display
        # Support both old (canvas_content) and new (canvas_data) formats
        output_state["canvas_content"] = latest_canvas.get('canvas_content')  # Legacy HTML/markdown
        output_state["canvas_data"] = latest_canvas.get('canvas_data')  # Structured data (preferred)
        output_state["canvas_type"] = latest_canvas.get('canvas_type')
        output_state["canvas_title"] = latest_canvas.get('canvas_title')
        output_state["canvas_requires_confirmation"] = latest_canvas.get('requires_confirmation', False)
        output_state["canvas_confirmation_message"] = latest_canvas.get('confirmation_message')
        output_state["has_canvas"] = True
        logger.info(f"📊 Updated canvas with display from '{latest_canvas.get('task_name')}'")
    
    print(f"!!! EXECUTE_BATCH: Returning output_state with latest_completed_tasks count={len(output_state['latest_completed_tasks'])}, task_events count={len(task_events)} !!!")
    logger.info(f"Output state keys: {list(output_state.keys())}")

    # Save the updated plan (using the object version for readability)
    temp_save_state = {**state, **output_state}
    temp_save_state['task_plan'] = remaining_plan_objects
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id:
        save_plan_to_file({**temp_save_state, "thread_id": thread_id})
    else:
        logger.warning("No thread_id found in config, skipping plan save")
    
    return output_state

def evaluate_agent_response(state: State):
    '''
    Critically evaluates the result of the last executed task to ensure it is
    logically correct and satisfies the user's intent before proceeding.
    '''
    print(f"!!! EVALUATE_AGENT_RESPONSE: Starting evaluation !!!")
    print(f"!!! EVALUATE: State keys: {list(state.keys())} !!!")
    print(f"!!! EVALUATE: latest_completed_tasks in state: {'latest_completed_tasks' in state} !!!")
    print(f"!!! EVALUATE: pending_confirmation={state.get('pending_confirmation')}, canvas_displays={len(state.get('canvas_displays', []))} !!!")
    
    # CANVAS CONFIRMATION FIX: Skip evaluation if confirmation is pending
    if state.get('pending_confirmation'):
        print(f"!!! EVALUATE: Skipping evaluation - confirmation pending !!!")
        logger.info("Skipping evaluation because confirmation is pending (preview result)")
        return {
            "pending_user_input": False, 
            "question_for_user": None,
            "eval_status": "pending_confirmation"  # Mark as pending confirmation, not failed
        }
    else:
        print(f"!!! EVALUATE: NOT skipping - pending_confirmation is {state.get('pending_confirmation')} !!!")
    
    latest_tasks = state.get("latest_completed_tasks", [])
    print(f"!!! EVALUATE: latest_tasks type={type(latest_tasks)}, value={latest_tasks} !!!")
    
    if not latest_tasks:
        # No new tasks to evaluate
        print(f"!!! EVALUATE: No tasks to evaluate (empty or None) !!!")
        logger.info("No tasks to evaluate")
        logger.info(f"State keys available: {list(state.keys())}")
        logger.info(f"completed_tasks: {state.get('completed_tasks', [])}")
        return {"pending_user_input": False, "question_for_user": None}

    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    task_to_evaluate = latest_tasks[-1] # Evaluate the most recent task
    
    print(f"!!! EVALUATE: Task='{task_to_evaluate.get('task_name')}', Result preview={str(task_to_evaluate.get('result', ''))[:200]} !!!")
    logger.info(f"Evaluating task: {task_to_evaluate.get('task_name')}")
    logger.info(f"Result preview: {str(task_to_evaluate.get('result', ''))[:500]}")

    # If the agent itself reported an error, we don't need to evaluate it further
    if isinstance(task_to_evaluate.get('result'), str) and "Error:" in task_to_evaluate.get('result', ''):
        print(f"!!! EVALUATE: Agent reported error, skipping evaluation !!!")
        logger.info("Agent reported error, skipping evaluation")
        return {"pending_user_input": False, "question_for_user": None}

    # Build conversation history for context
    conversation_history = ""
    if messages := state.get('messages'):
        recent_messages = messages[-10:]  # Last 10 messages for context
        for msg in recent_messages:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    conversation_history += f"User: {msg.content}\n"
                elif msg.type == "ai":
                    conversation_history += f"Assistant: {msg.content}\n"
    
    # Include completed tasks for context
    completed_context = ""
    if completed_tasks := state.get('completed_tasks', []):
        completed_context = "\n**Previously Completed Tasks:**\n"
        for task in completed_tasks[-3:]:  # Last 3 tasks
            task_name = task.get('task_name', 'Unknown')
            result_preview = str(task.get('result', ''))[:200]
            completed_context += f"- {task_name}: {result_preview}...\n"
    
    # Check if this is a browser agent result with partial success
    result_str = str(task_to_evaluate.get('result', ''))
    is_browser_task = 'screenshot' in result_str.lower() or 'browser' in result_str.lower()
    has_partial_success = 'completed' in result_str.lower() and ('subtask' in result_str.lower() or 'failed' in result_str.lower())
    
    # If browser agent completed some subtasks, consider it successful
    if is_browser_task and has_partial_success:
        print(f"!!! EVALUATE: Browser agent with partial success - auto-approving !!!")
        logger.info("Browser agent completed some subtasks - considering successful")
        return {"pending_user_input": False, "question_for_user": None}
    
    prompt = f'''
    You are a Quality Assurance AI with REACTIVE ROUTING capabilities. Determine if an agent's output successfully fulfills its task and decide the next action.

    **Conversation History:**
    {conversation_history}
    {completed_context}

    **Current Request:** "{state['original_prompt']}"
    **Task:** "{task_to_evaluate.get('task_description', 'N/A')}"
    **Agent's Result:**
    ```json
    {json.dumps(task_to_evaluate['result'], indent=2)[:1000]}
    ```

    **REACTIVE EVALUATION - Choose ONE of these statuses:**
    
    1. **"complete"**: Task fully successful, proceed to next task
       - Result contains all expected data
       - Agent accomplished the goal
    
    2. **"partial_success"**: Task partially successful, but acceptable
       - Got SOME useful data (e.g., "Completed 3/5 subtasks")
       - Partial results are better than nothing
       - Proceed to next task with warning
    
    3. **"failed"**: Task failed but can be recovered automatically
       - Agent returned error or empty result
       - Another agent or approach might work
       - System will AUTO-REPLAN (loop back to planning)
       - Provide `feedback_for_replanning` with specific guidance
    
    4. **"user_input_required"**: Task needs user clarification
       - Result is ambiguous or requires user decision
       - Missing information only user can provide
       - Provide a clear `question` for the user

    **Rules:**
    - **Browser agent results:** If shows "Completed X/Y subtasks" where X > 0, use "partial_success" or "complete"
    - **Data presence:** If result has ANY useful data, use "complete" or "partial_success"
    - **Be pragmatic:** Prefer "partial_success" over "failed" when there's some value
    - **Auto-recovery:** Use "failed" with feedback when another attempt might work
    - **Last resort:** Only use "user_input_required" if truly stuck

    **Output Format:**
    {{
        "status": "complete|partial_success|failed|user_input_required",
        "reasoning": "Brief explanation",
        "feedback_for_replanning": "Specific guidance if status is 'failed'",
        "question": "Clear question if status is 'user_input_required'"
    }}
    '''
    try:
        from schemas import AgentResponseEvaluationEnhanced
        evaluation = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, AgentResponseEvaluationEnhanced)
        print(f"!!! EVALUATE: LLM evaluation status={evaluation.status} !!!")
        logger.info(f"Evaluation result: status={evaluation.status}, reasoning={evaluation.reasoning}")
        
        # REACTIVE ROUTING: Handle different statuses
        if evaluation.status == "failed":
            # Check replan count to prevent infinite loops
            replan_count = state.get("replan_count", 0)
            MAX_REPLANS = 3  # Maximum number of replans before giving up
            
            if replan_count >= MAX_REPLANS:
                print(f"!!! EVALUATE: Max replans ({MAX_REPLANS}) reached - stopping auto-replan !!!")
                logger.error(f"Task '{task_to_evaluate['task_name']}' failed after {replan_count} replans. Stopping auto-replan.")
                return {
                    "pending_user_input": False,
                    "question_for_user": None,
                    "eval_status": "complete",  # Mark as complete to stop the loop
                    "final_response": f"I attempted to complete the task '{task_to_evaluate['task_name']}' but encountered repeated failures. The last error was: {task_to_evaluate.get('result', 'Unknown error')}"
                }
            
            print(f"!!! EVALUATE: Task FAILED - triggering auto-replan (attempt {replan_count + 1}/{MAX_REPLANS}) !!!")
            logger.warning(f"Task '{task_to_evaluate['task_name']}' failed. Triggering auto-replan (attempt {replan_count + 1}/{MAX_REPLANS}).")
            logger.warning(f"Replan feedback: {evaluation.feedback_for_replanning}")
            
            return {
                "pending_user_input": False,
                "question_for_user": None,
                "replan_reason": evaluation.feedback_for_replanning or f"Task '{task_to_evaluate['task_name']}' failed: {evaluation.reasoning}",
                "eval_status": "failed",
                "replan_count": replan_count + 1
            }
        
        elif evaluation.status == "user_input_required":
            print(f"!!! EVALUATE: User input required - question='{evaluation.question}' !!!")
            logger.warning(f"Result for task '{task_to_evaluate['task_name']}' requires user input.")
            logger.warning(f"Evaluation question: {evaluation.question}")
            
            # Preserve canvas if it was set by a previous task
            result = {
                "pending_user_input": True,
                "question_for_user": evaluation.question,
                "eval_status": "user_input_required",
                "replan_count": 0  # Reset replan count on user input
            }
            
            # Check if we have canvas from previous tasks
            if state.get('has_canvas'):
                result['has_canvas'] = state['has_canvas']
                result['canvas_type'] = state.get('canvas_type')
                result['canvas_content'] = state.get('canvas_content')
                logger.info("Preserving canvas from previous successful task")
            
            return result
        
        elif evaluation.status == "partial_success":
            print(f"!!! EVALUATE: Task PARTIALLY successful - continuing !!!")
            logger.info(f"Task '{task_to_evaluate['task_name']}' partially successful: {evaluation.reasoning}")
            # Reset replan count on success
            return {"replan_count": 0}
        
        else:  # complete
            print(f"!!! EVALUATE: Task COMPLETE !!!")
            logger.info(f"Task '{task_to_evaluate['task_name']}' completed successfully")
            # Reset replan count on success
            
            # Check if this is a browser task with screenshots
            task_result = task_to_evaluate.get('result', {})
            if isinstance(task_result, dict) and task_result.get('screenshot_files'):
                screenshot_files = task_result['screenshot_files']
                logger.info(f"Found {len(screenshot_files)} screenshot files in browser result")
                
                # Create canvas content with slideshow for multiple screenshots
                if screenshot_files:
                    if len(screenshot_files) == 1:
                        # Single screenshot - just display it
                        file_path = screenshot_files[0].get('file_path', '').replace('\\', '/')
                        # Use absolute URL to backend server
                        canvas_html = f'<img src="http://localhost:8000/{file_path}" alt="Browser screenshot" style="width: 100%; border-radius: 8px;" />'
                    else:
                        # Multiple screenshots - create slideshow
                        screenshots_html = ""
                        for idx, screenshot in enumerate(screenshot_files):
                            file_path = screenshot.get('file_path', '').replace('\\', '/')
                            display = "block" if idx == 0 else "none"
                            # Use absolute URL to backend server
                            screenshots_html += f'<img id="screenshot-{idx}" src="http://localhost:8000/{file_path}" alt="Browser screenshot {idx+1}" style="width: 100%; border-radius: 8px; display: {display};" />\n'
                        
                        canvas_html = f'''
                        <div style="position: relative;">
                            {screenshots_html}
                            <div style="text-align: center; margin-top: 10px;">
                                <span id="screenshot-counter">1 / {len(screenshot_files)}</span>
                                <div style="margin-top: 5px;">
                                    <button onclick="prevScreenshot()" style="margin: 0 5px; padding: 5px 15px; cursor: pointer;">← Previous</button>
                                    <button onclick="nextScreenshot()" style="margin: 0 5px; padding: 5px 15px; cursor: pointer;">Next →</button>
                                    <button id="play-btn" onclick="toggleAutoPlay()" style="margin: 0 5px; padding: 5px 15px; cursor: pointer;">▶ Play</button>
                                </div>
                            </div>
                        </div>
                        <script>
                            let currentScreenshot = 0;
                            const totalScreenshots = {len(screenshot_files)};
                            let autoPlayInterval = null;
                            
                            function showScreenshot(index) {{
                                for (let i = 0; i < totalScreenshots; i++) {{
                                    document.getElementById('screenshot-' + i).style.display = 'none';
                                }}
                                document.getElementById('screenshot-' + index).style.display = 'block';
                                document.getElementById('screenshot-counter').textContent = (index + 1) + ' / ' + totalScreenshots;
                                currentScreenshot = index;
                            }}
                            
                            function nextScreenshot() {{
                                currentScreenshot = (currentScreenshot + 1) % totalScreenshots;
                                showScreenshot(currentScreenshot);
                            }}
                            
                            function prevScreenshot() {{
                                currentScreenshot = (currentScreenshot - 1 + totalScreenshots) % totalScreenshots;
                                showScreenshot(currentScreenshot);
                            }}
                            
                            function toggleAutoPlay() {{
                                const btn = document.getElementById('play-btn');
                                if (autoPlayInterval) {{
                                    clearInterval(autoPlayInterval);
                                    autoPlayInterval = null;
                                    btn.textContent = '▶ Play';
                                }} else {{
                                    autoPlayInterval = setInterval(nextScreenshot, 2000);
                                    btn.textContent = '⏸ Pause';
                                }}
                            }}
                            
                            // Auto-play on load
                            toggleAutoPlay();
                        </script>
                        '''
                    
                    logger.info(f"Setting canvas with {len(screenshot_files)} screenshot(s)")
                    
                    # Create plan view showing task plan and progress
                    task_plan = state.get('task_plan', [])
                    plan_html = '<div style="padding: 20px; font-family: system-ui;">'
                    plan_html += '<h2 style="margin-bottom: 20px;">📋 Task Plan</h2>'
                    
                    if task_plan:
                        for i, task in enumerate(task_plan, 1):
                            status = task.get('status', 'pending')
                            status_icon = '✅' if status == 'completed' else '⏳' if status == 'pending' else '❌'
                            status_color = '#10b981' if status == 'completed' else '#6b7280' if status == 'pending' else '#ef4444'
                            
                            plan_html += f'''
                            <div style="margin-bottom: 15px; padding: 15px; border-left: 4px solid {status_color}; background: #f9fafb; border-radius: 4px;">
                                <div style="font-weight: 600; margin-bottom: 5px;">
                                    {status_icon} Task {i}: {task.get('subtask', 'Unknown')}
                                </div>
                                <div style="font-size: 0.875rem; color: #6b7280;">
                                    Status: <span style="color: {status_color}; font-weight: 500;">{status}</span>
                                </div>
                            </div>
                            '''
                    else:
                        plan_html += '<p style="color: #6b7280;">No task plan available</p>'
                    
                    plan_html += '</div>'
                    
                    return {
                        "pending_user_input": False,
                        "question_for_user": None,
                        "has_canvas": True,
                        "canvas_type": "html",
                        "canvas_content": canvas_html,
                        "browser_view": canvas_html,
                        "plan_view": plan_html
                    }
    except Exception as e:
        print(f"!!! EVALUATE: Evaluation failed with error: {e} !!!")
        logger.error(f"Failed to evaluate agent response for task '{task_to_evaluate['task_name']}': {e}")
    
    return {"pending_user_input": False, "question_for_user": None, "replan_count": 0}

def ask_user(state: State):
    """
    Formats the question for the user and prepares it as the final response.
    This is a terminal node that ends the graph's execution for the current run.
    """
    # Get any existing question or create a default one based on parsing failures
    question = state.get("question_for_user")
    
    if not question:
        # Generate a default question based on the context
        parsing_error = state.get("parsing_error_feedback")
        original_prompt = state.get("original_prompt", "")
        
        if parsing_error:
            question = f"I couldn't find suitable agents for your request: '{original_prompt}'. Could you please provide more specific details about what you'd like me to help you with?"
        else:
            question = f"I need more information to help you with: '{original_prompt}'. Could you please provide more specific details about what you'd like me to do?"
    
    logger.info(f"Asking user for clarification: {question}")

    # Create AI message with timestamp metadata
    import hashlib
    timestamp = time.time()
    unique_string = f"ai:{question}:{timestamp}"
    msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
    ai_message = AIMessage(
        content=question,
        additional_kwargs={"timestamp": timestamp, "id": msg_id}
    )
    
    # Preserve approval-related fields if they exist (for plan approval flow)
    result = {
        "pending_user_input": True,
        "question_for_user": question,
        "final_response": None, # Clear any previous final response
        "messages": [ai_message]
    }
    
    # Preserve approval state if this is a plan approval request
    if state.get("approval_required"):
        result["approval_required"] = state.get("approval_required")
        result["estimated_cost"] = state.get("estimated_cost")
        result["task_count"] = state.get("task_count")
        logger.info(f"Preserving approval state: approval_required={result['approval_required']}, cost={result['estimated_cost']}, tasks={result['task_count']}")
    
    # CRITICAL: Also preserve plan_approved flag for routing after approval
    if state.get("plan_approved"):
        result["plan_approved"] = state.get("plan_approved")
        logger.info(f"Preserving plan_approved={result['plan_approved']} for post-approval routing")
    
    return result


def render_canvas_output(state: State):
    """
    Renders canvas output when needed for complex visualizations, documents, or webpages.
    This function is called after generate_final_response and uses the canvas decision made there.
    """
    logger.info("=== CANVAS RENDER: Starting ===")
    logger.info(f"Canvas type: {state.get('canvas_type')}, needs_canvas: {state.get('needs_canvas')}")
    
    # Check if canvas is needed based on the decision from generate_final_response
    needs_canvas = state.get("needs_canvas")
    if not needs_canvas:
        logger.info("CANVAS RENDER: Canvas not needed, returning empty state")
        logger.info("CANVAS RENDER: This means generate_final_response decided canvas was not needed")
        return {}
    
    # Check if we already have canvas content
    if state.get("has_canvas") and state.get("canvas_content"):
        logger.info("CANVAS RENDER: Canvas content already exists, skipping generation")
        logger.info("CANVAS RENDER: Canvas content found, will be used by frontend")
        return {}
    
    canvas_type = state.get("canvas_type")
    canvas_prompt = state.get("canvas_prompt") or state.get("original_prompt", "")
    
    if not canvas_type or not canvas_prompt:
        logger.info("CANVAS RENDER: Missing canvas_type or canvas_prompt, skipping generation")
        logger.info(f"CANVAS RENDER: canvas_type={canvas_type}, canvas_prompt={canvas_prompt}")
        return {}
    
    logger.info(f"CANVAS RENDER: Generating {canvas_type} canvas content")
    logger.info(f"CANVAS RENDER: Canvas prompt: {canvas_prompt}")
    
    # Initialize all LLMs for fallback mechanism
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    # Extract additional context from state
    original_prompt = state.get("original_prompt", "")
    messages = state.get("messages", [])
    completed_tasks = state.get("completed_tasks", [])
    uploaded_files = state.get("uploaded_files", [])
    
    # Format conversation history
    history_context = ""
    if messages:
        recent_messages = messages[-20:]  # Last 20 messages
        for msg in recent_messages:
            if hasattr(msg, 'type') and msg.type == "human":
                history_context += f"User: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history_context += f"Assistant: {msg.content}\n"
    
    # Format completed tasks
    tasks_context = ""
    if completed_tasks:
        tasks_info = []
        for task in completed_tasks[-10:]:  # Last 10 tasks
            task_name = task.get('task_name', 'Unknown Task')
            task_result = task.get('result', {})
            # Truncate long results
            result_str = json.dumps(task_result, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"
            tasks_info.append(f"- {task_name}: {result_str}")
        tasks_context = "Completed Tasks:\n" + "\n".join(tasks_info) + "\n\n"
    
    # Format uploaded files
    files_context = ""
    if uploaded_files:
        files_info = []
        for file_obj in uploaded_files:
            # Convert dict to FileObject if needed for attribute access
            if isinstance(file_obj, dict):
                try:
                    file_obj = FileObject.model_validate(file_obj)
                except Exception:
                    # If validation fails, use the dict directly
                    pass
            
            if isinstance(file_obj, dict):
                file_info = f"- {file_obj.get('file_name', 'Unknown')} ({file_obj.get('file_type', 'Unknown')}) at {file_obj.get('file_path', 'Unknown')}"
                if file_obj.get('file_type') == 'document' and file_obj.get('vector_store_path'):
                    file_info += f", vector store: {file_obj.get('vector_store_path')}"
            else:
                file_info = f"- {getattr(file_obj, 'file_name', 'Unknown')} ({getattr(file_obj, 'file_type', 'Unknown')}) at {getattr(file_obj, 'file_path', 'Unknown')}"
                if getattr(file_obj, 'file_type', '') == 'document' and getattr(file_obj, 'vector_store_path', None):
                    file_info += f", vector store: {getattr(file_obj, 'vector_store_path')}"
            files_info.append(file_info)
        files_context = "Uploaded Files:\n" + "\n".join(files_info) + "\n\n"
    
    # Create a prompt to generate the canvas content with full context
    prompt = f'''
    You are the Orbimesh Orchestrator with a built-in Canvas feature.
    
    **YOUR IDENTITY:**
    - You are Orbimesh, a multi-agent orchestration system
    - You have a built-in Canvas feature that can render interactive HTML/CSS/JavaScript and Markdown content
    - You are now using your Canvas capability to create content for the user
    
    **YOUR TASK:**
    Generate {canvas_type} content for the following request:
    
    Original User Request: "{original_prompt}"
    
    Canvas-Specific Request: "{canvas_prompt}"
    
    **Additional Context:**
    {files_context}
    {tasks_context}
    Conversation History:
    {history_context}
    
    **IMPORTANT INSTRUCTIONS FOR CANVAS GENERATION:**
    
    **PURPOSE AND ENVIRONMENT:**
    - You are generating content for your built-in canvas display area
    - The canvas is displayed in an iframe with limited capabilities
    - The canvas should be a standalone, self-contained {canvas_type} document
    - Do NOT assume any existing libraries or frameworks are available unless explicitly imported
    
    **RESPONSE FORMAT REQUIREMENTS:**
    - Generate ONLY the complete {canvas_type} content, nothing else
    - Do NOT include any markdown code block formatting
    - Do NOT include any explanations or comments outside the {canvas_type} content
    
    **FOR HTML CANVAS CONTENT SPECIFICALLY:**
    - Include complete HTML5 structure with <!DOCTYPE html> declaration
    - For any external libraries (React, ReactDOM, Babel, etc.), use external CDN imports in <script> tags
    - Example for React:
      <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
      <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
      <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    - All JavaScript must be contained within <script> tags
    - Use inline CSS in <style> tags or inline styles
    - Ensure all functionality works in a standalone HTML file
    - Avoid complex build tools or bundlers - everything must work directly in the browser
    
    **FOR MARKDOWN CANVAS CONTENT:**
    - Use proper markdown syntax and formatting
    - Focus on clear, readable content presentation
    - Use appropriate headers, lists, and emphasis as needed
    
    **CONTENT QUALITY REQUIREMENTS:**
    - Make the content interactive and engaging when appropriate
    - Ensure visual appeal with good styling and layout
    - Utilize the provided context to create relevant content
    - For interactive elements, ensure they work correctly in the canvas environment
    - Test that all functionality works in a standalone environment
    
    Generate the complete {canvas_type} content that can be rendered directly in a browser.
    '''
    
    try:
        if canvas_type == "html":
            # Generate HTML content
            canvas_content = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, None).__str__()
            
            # Strip any markdown code block formatting
            if isinstance(canvas_content, str):
                canvas_content = re.sub(r"```html\s*", "", canvas_content)
                canvas_content = re.sub(r"\s*```", "", canvas_content)
                canvas_content = re.sub(r"```\s*", "", canvas_content)
            
            # Ensure it's proper HTML and fix JavaScript issues
            logger.info(f"CANVAS RENDER: Original canvas content length: {len(canvas_content)}")
            logger.info(f"CANVAS RENDER: Canvas content preview: {canvas_content[:200]}...")
            
            # Always wrap in complete HTML structure to ensure proper rendering
            if not canvas_content.strip().startswith('<!DOCTYPE html>'):
                canvas_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Canvas Content</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; }}
        button {{ padding: 10px 20px; font-size: 16px; margin: 10px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 4px; }}
        button:hover {{ background-color: #45a049; }}
        #game-board {{ display: grid; grid-template-columns: repeat(3, 100px); grid-gap: 5px; margin: 20px auto; }}
        .cell {{ width: 100px; height: 100px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; font-size: 2em; cursor: pointer; }}
        .cell:hover {{ background-color: #e0e0e0; }}
        #status {{ margin: 20px; font-size: 1.2em; font-weight: bold; }}
    </style>
</head>
<body>
    {canvas_content}
</body>
</html>'''
                
                logger.info("CANVAS RENDER: Wrapped content in complete HTML structure")
            
            # Fix common JavaScript issues - ensure canvas_content is a string
            if isinstance(canvas_content, str):
                # Fix escaped quotes
                canvas_content = canvas_content.replace('onclick=\\"', 'onclick="')
                canvas_content = canvas_content.replace('\\"', '"')
                canvas_content = canvas_content.replace("\\'", "'")
                
                # Ensure proper script tags are closed
                # Count opening and closing script tags
                open_script_count = canvas_content.count('<script')
                close_script_count = canvas_content.count('</script>')
                
                # Add closing script tags if needed
                if open_script_count > close_script_count:
                    for _ in range(open_script_count - close_script_count):
                        canvas_content += '</script>'
                        
                # Fix any malformed script tags
                canvas_content = re.sub(r'<script([^>]*)>(.*?)(?=</script>|$)', r'<script\1>\2</script>', canvas_content, flags=re.DOTALL)
            else:
                logger.warning(f"CANVAS RENDER: canvas_content is not a string, got {type(canvas_content)}")
                canvas_content = str(canvas_content)
            
            logger.info("CANVAS RENDER: Generated HTML canvas content")
            return {
                "has_canvas": True,
                "canvas_type": "html",
                "canvas_content": canvas_content
            }
        elif canvas_type == "markdown":
            # Generate Markdown content
            canvas_content = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, None).__str__()
            
            # Strip any markdown code block formatting
            canvas_content = re.sub(r"```markdown\s*", "", canvas_content)
            canvas_content = re.sub(r"\s*```", "", canvas_content)
            canvas_content = re.sub(r"```\s*", "", canvas_content)
            
            logger.info("CANVAS RENDER: Generated Markdown canvas content")
            return {
                "has_canvas": True,
                "canvas_type": "markdown",
                "canvas_content": canvas_content
            }
        else:
            logger.error(f"CANVAS RENDER: Unknown canvas type: {canvas_type}")
            return {}
            
    except Exception as e:
        logger.error(f"CANVAS RENDER: Canvas content generation failed: {e}")
        # Even if generation fails, we still want to indicate that canvas was needed
        # This will help with debugging and ensure the frontend knows to expect canvas content
        return {
            "has_canvas": True,
            "canvas_type": canvas_type,
            "canvas_content": f"<p>Error generating canvas content: {str(e)}</p>"
        }


def generate_text_answer(state: State):
    """
    Generates a simple text answer for the user's request.
    This is the first step in the final response generation pipeline.
    """
    logger.info("=== GENERATE_TEXT_ANSWER: Starting text answer generation ===")
    
    # Check if canvas is needed - if so, we should generate a more concise text response
    needs_canvas = state.get("needs_canvas", False)
    canvas_type = state.get("canvas_type", "")
    
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    # Check if this is a simple request that was handled directly by analyze_request
    if state.get("needs_complex_processing") is False:
        # This is a simple request that was handled by analyze_request
        final_response = state.get("final_response", "")
        if not final_response:
            # Fallback: generate a new simple response
            
            # Build comprehensive context from conversation history
            history_context = ""
            if state.get('messages'):
                recent_messages = state['messages'][-20:]
                for msg in recent_messages:
                    if hasattr(msg, 'type') and msg.type == "human":
                        history_context += f"User: {msg.content}\n"
                    elif hasattr(msg, 'type') and msg.type == "ai":
                        history_context += f"Assistant: {msg.content}\n"
            
            # Include uploaded files context
            files_context = ""
            uploaded_files = state.get('uploaded_files', [])
            if uploaded_files:
                files_info = []
                for file_obj in uploaded_files:
                    if isinstance(file_obj, dict):
                        file_obj = FileObject.model_validate(file_obj)
                    file_info = f"- {file_obj.file_name} ({file_obj.file_type}) at {file_obj.file_path}"
                    if file_obj.file_type == 'document' and file_obj.vector_store_path:
                        file_info += f", vector store: {file_obj.vector_store_path}"
                    files_info.append(file_info)
                files_context = "Uploaded files:\n" + "\n".join(files_info) + "\n\n"
            
            prompt = f'''
            You are a helpful AI assistant. Answer the user's request directly and concisely.
            
            Consider the following context:
            {files_context}
            Conversation history:
            {history_context}
            
            User's current request: "{state['original_prompt']}"
            
            Please provide a helpful response to the user's request.
            '''
            
            final_response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, None).__str__()
            logger.info("Simple text answer generated.")
        else:
            logger.info("Using existing final_response for simple request.")
        
        # Create AI message with timestamp metadata
        import hashlib
        timestamp = time.time()
        unique_string = f"ai:{final_response}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        ai_message = AIMessage(
            content=final_response,
            additional_kwargs={"timestamp": timestamp, "id": msg_id}
        )
        # Use MessageManager to add message without duplicates
        from orchestrator.message_manager import MessageManager
        existing_messages = state.get("messages", [])
        updated_messages = MessageManager.add_message(existing_messages, ai_message)
        logger.info(f"Added AI message. Total messages: {len(updated_messages)}")
        return {"final_response": final_response, "messages": updated_messages}
    else:
        # This is a complex request, synthesize results from completed tasks
        completed_tasks = state.get('completed_tasks', [])
        if not completed_tasks:
            logger.warning("Complex request indicated but no completed tasks found. Generating default response.")
            final_response = "I've processed your request, but I don't have specific results to share."
        else:
            # Use summarized task results for context
            # Note: Unified content system is for file artifacts, not task result summarization
            summarized_results = _summarize_completed_tasks_for_context(completed_tasks)
            context_str = json.dumps(summarized_results, indent=2)
            logger.info(f"📊 Generated context from {len(completed_tasks)} completed tasks: {len(context_str)} chars")
            
            # Hard limit on context (roughly 20k tokens = 80k chars)
            if len(context_str) > 80000:
                logger.warning(f"Context too large ({len(context_str)} chars), truncating to 80k chars")
                context_str = context_str[:80000] + "\n\n[... context truncated due to length ...]"
            
            # If canvas is needed, generate a more concise text response that references the canvas
            if needs_canvas and canvas_type:
                # Limit context to prevent token overflow (50k chars ≈ 12.5k tokens)
                safe_context = context_str[:50000] if len(context_str) > 50000 else context_str
                prompt = f'''
                You are an expert project manager's assistant. Your job is to synthesize the results from a team of AI agents into a single, clean, and coherent final report for the user.
                
                **CRITICAL RULE: You MUST base your response ONLY on the actual results provided below. DO NOT invent, assume, or hallucinate any information that is not explicitly stated in the results.**
                
                The user's original request was:
                "{state['original_prompt']}"
                
                The following tasks were completed, with these results:
                ---
                {safe_context}
                ---
                
                A {canvas_type} visualization has been prepared to display this information. 
                Please generate a brief, human-readable summary that references the visualization 
                and highlights the key findings without reproducing the raw data or code.
                
                **FORMATTING REQUIREMENTS:**
                - Use clear paragraph breaks for readability
                - Use bullet points (•) or numbered lists for multiple items
                - **Use markdown tables for structured data** (events, schedules, comparisons, lists with multiple attributes)
                - Bold important information using **text**
                - Keep paragraphs concise (2-3 sentences max)
                - Add line breaks between sections for better visual separation
                - Use proper capitalization and punctuation
                - Structure: Brief intro → Key findings → Reference to visualization
                
                **TABLE FORMATTING:**
                When presenting structured data (like events, schedules, contests, products, etc.), use markdown tables:
                ```
                | Column 1 | Column 2 | Column 3 |
                |----------|----------|----------|
                | Data 1   | Data 2   | Data 3   |
                ```
                
                **CONTENT REQUIREMENTS:**
                - If the results indicate something was NOT found or NOT present, clearly state that in your response
                - Do NOT make up details that are not in the actual results
                - Be factual and accurate based solely on the provided data
                - Pay special attention to the 'extracted_data' field - if it's null or empty, that means NO data was found
                - If actions_taken show the agent looked for something but didn't find it, report that accurately
                '''
            else:
                # Limit context to prevent token overflow (50k chars ≈ 12.5k tokens)
                safe_context = context_str[:50000] if len(context_str) > 50000 else context_str
                prompt = f'''
                You are an expert project manager's assistant. Your job is to synthesize the results from a team of AI agents into a single, clean, and coherent final report for the user.
                
                **CRITICAL RULE: You MUST base your response ONLY on the actual results provided below. DO NOT invent, assume, or hallucinate any information that is not explicitly stated in the results.**
                
                The user's original request was:
                "{state['original_prompt']}"
                
                The following tasks were completed, with these results:
                ---
                {safe_context}
                ---
                
                Please generate a final, human-readable response that directly answers the user's original request based on the collected results.
                
                **FORMATTING REQUIREMENTS:**
                - Use clear paragraph breaks for readability
                - Use bullet points (•) or numbered lists for multiple items
                - **Use markdown tables for structured data** (events, schedules, comparisons, lists with multiple attributes)
                - Bold important information using **text**
                - Keep paragraphs concise (2-3 sentences max)
                - Add line breaks between sections for better visual separation
                - Use proper capitalization and punctuation
                - Structure your response logically with clear sections
                - For lists of data, use consistent formatting (e.g., "• Item 1", "• Item 2")
                
                **TABLE FORMATTING:**
                When presenting structured data (like events, schedules, contests, products, comparisons), use markdown tables:
                ```
                | Column 1 | Column 2 | Column 3 |
                |----------|----------|----------|
                | Data 1   | Data 2   | Data 3   |
                ```
                Tables are especially useful for:
                - Event schedules (name, date, time, duration, status)
                - Product comparisons (name, price, features)
                - Contest listings (title, start time, duration, registration)
                - Any data with multiple attributes per item
                
                **CONTENT REQUIREMENTS:**
                - If the results indicate something was NOT found or NOT present, clearly state that in your response
                - Do NOT make up details that are not in the actual results
                - Be factual and accurate based solely on the provided data
                - Pay special attention to the 'extracted_data' field - if it's null or empty, that means NO data was found
                - If actions_taken show the agent looked for something but didn't find it, report that accurately
                - If the agent could not complete a task or found nothing, report that honestly
                '''
            
            final_response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, None).__str__()
            logger.info("Complex text answer generated.")
        
        # If canvas is needed, provide a more concise message that references the canvas
        if needs_canvas and canvas_type:
            # Check if the final_response contains HTML/Markdown code that should be in the canvas
            html_indicators = ['<!DOCTYPE html>', '<html', '<button', '<script>', 'onclick=', 'onClick=',
                              '<div', '<span', '<p>', '<h1>', '<h2>', '<h3>', '<style>', '<head>']
            markdown_indicators = ['```html', '```markdown', '# ', '## ', '### ', '**', '* ', '- ']
            
            # If the response contains code that should be in the canvas, create a more concise response
            contains_html = any(indicator in final_response for indicator in html_indicators)
            contains_markdown_code = any(indicator in final_response for indicator in ['```html', '```markdown'])
            
            if contains_html or contains_markdown_code:
                # Generate a more concise response that references the canvas
                concise_prompt = f'''
                The user's original request was:
                "{state['original_prompt']}"
                
                A {canvas_type} visualization has been created to display the results.
                Please generate a brief, human-readable message that tells the user to check 
                the {canvas_type} visualization for the results, without including any code.
                '''
                final_response = invoke_llm_with_fallback(primary_llm, fallback_llm, concise_prompt, None).__str__()
                logger.info("Generated concise text response for canvas visualization.")
        
        # Create AI message with timestamp metadata
        import hashlib
        timestamp = time.time()
        unique_string = f"ai:{final_response}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        ai_message = AIMessage(
            content=final_response,
            additional_kwargs={"timestamp": timestamp, "id": msg_id}
        )
        # Use MessageManager to add message without duplicates
        from orchestrator.message_manager import MessageManager
        existing_messages = state.get("messages", [])
        updated_messages = MessageManager.add_message(existing_messages, ai_message)
        logger.info(f"Added AI message. Total messages: {len(updated_messages)}")
        return {"final_response": final_response, "messages": updated_messages}

def generate_final_response(state: State):
    """
    UNIFIED FINAL RESPONSE: Generates both text and canvas in a single optimized call.
    This replaces the old two-step process (generate_text_answer + render_canvas_output).
    """
    logger.info("=== GENERATE_FINAL_RESPONSE: Starting unified generation ===")
    print(f"!!! GENERATE_FINAL_RESPONSE: pending_confirmation={state.get('pending_confirmation')} !!!")
    print(f"!!! GENERATE_FINAL_RESPONSE: has_canvas={state.get('has_canvas')}, canvas_type={state.get('canvas_type')} !!!")
    print(f"!!! GENERATE_FINAL_RESPONSE: canvas_data exists={state.get('canvas_data') is not None}, canvas_content exists={state.get('canvas_content') is not None} !!!")
    
    # Check if confirmation is pending - if so, skip text generation and just show canvas
    if state.get('pending_confirmation'):
        logger.info("⏸️ Confirmation pending - skipping text generation, showing canvas only")
        print(f"!!! GENERATE_FINAL_RESPONSE: Returning canvas for confirmation !!!")
        print(f"!!! GENERATE_FINAL_RESPONSE: canvas_type={state.get('canvas_type')}, canvas_title={state.get('canvas_title')} !!!")
        # Don't set final_response to avoid adding a message to chat
        # The canvas will show the confirmation UI
        return {
            "final_response": "",  # Empty to avoid chat message
            "messages": state.get('messages', []),
            "needs_canvas": False,
            "has_canvas": True,
            "canvas_type": state.get('canvas_type'),
            "canvas_content": state.get('canvas_content'),
            "canvas_data": state.get('canvas_data'),
            "canvas_title": state.get('canvas_title'),
            "pending_confirmation": True,
            "pending_confirmation_task": state.get('pending_confirmation_task'),
            "canvas_confirmation_message": state.get('canvas_confirmation_message') or "Review the changes and confirm to apply them to the document"
        }
    
    # Check if canvas was already set by evaluate_agent_response (e.g., browser screenshots)
    # or by preprocess_files (e.g., uploaded documents)
    # Support both canvas_content (string) and canvas_data (structured object)
    if state.get('has_canvas') and (state.get('canvas_content') or state.get('canvas_data')):
        logger.info("Canvas already set by previous node, preserving it")
        # Still need to generate the text response
        text_result = generate_text_answer(state)
        return {
            "final_response": text_result.get('final_response', ''),
            "messages": text_result.get('messages', []),
            "needs_canvas": False,  # Canvas already rendered
            "has_canvas": True,
            "canvas_type": state.get('canvas_type'),
            "canvas_content": state.get('canvas_content'),
            "canvas_data": state.get('canvas_data'),
            "canvas_title": state.get('canvas_title')
        }
    
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    # Check if this is a simple request (no complex processing)
    is_simple_request = state.get("needs_complex_processing") is False
    completed_tasks = state.get('completed_tasks', [])
    
    # UNIFIED GENERATION: Generate text + canvas decision + canvas content in ONE call
    if is_simple_request:
        # Simple request - just generate text response
        text_result = generate_text_answer(state)
        final_response = text_result.get('final_response', '')
        
        # Check if response contains HTML that should be in canvas
        contains_html = any(indicator in final_response for indicator in [
            '<!DOCTYPE html>', '<html', '<button', '<script>', 'onclick=', 'onClick=',
            '<div', '<span>', '<p>', '<h1>', '<h2>', '<h3>', '<style>', '<head>'
        ])
        
        if contains_html:
            logger.info("Simple request with HTML detected - extracting to canvas")
            # Extract HTML to canvas
            return {
                "final_response": "I've created an interactive visualization for you. Check it out in the canvas!",
                "messages": text_result.get('messages', []),
                "has_canvas": True,
                "canvas_type": "html",
                "canvas_content": final_response
            }
        else:
            # Preserve existing canvas state if it was set by previous nodes (e.g., preprocess_files)
            result = {
                "final_response": final_response,
                "messages": text_result.get('messages', []),
            }
            # Only clear canvas if it wasn't already set
            if not state.get('has_canvas'):
                result["has_canvas"] = False
                result["canvas_type"] = None
                result["canvas_content"] = None
            return result
    
    # Complex request - check if canvas was already set by agents
    # If canvas was set by execute_batch (from agent canvas_display), preserve it
    if state.get('has_canvas') and (state.get('canvas_content') or state.get('canvas_data')):
        logger.info("Canvas already set by agent (execute_batch), preserving it")
        # Still need to generate the text response
        text_result = generate_text_answer(state)
        return {
            "final_response": text_result.get('final_response', ''),
            "messages": text_result.get('messages', []),
            "needs_canvas": False,  # Canvas already rendered
            "has_canvas": True,
            "canvas_type": state.get('canvas_type'),
            "canvas_content": state.get('canvas_content'),
            "canvas_data": state.get('canvas_data'),
            "canvas_title": state.get('canvas_title')
        }
    
    # Complex request - use unified generation
    logger.info("Complex request - using unified text + canvas generation")
    
    # UNIFIED PROMPT: Generate text response + canvas decision + canvas content in ONE call
    # Use summarized completed tasks to avoid token limit issues
    # The _summarize_completed_tasks_for_context function extracts key info while removing bloat
    summarized_results = _summarize_completed_tasks_for_context(completed_tasks)
    
    # ARTIFACT EXPANSION: Restore compressed results for LLM context
    if ARTIFACT_INTEGRATION_ENABLED and artifact_hooks:
        thread_id = state.get('thread_id')
        if thread_id:
            for task in summarized_results:
                result = task.get('result', {})
                if isinstance(result, dict) and '_artifact_ref' in result:
                    artifact_id = result['_artifact_ref'].get('id')
                    if artifact_id:
                        try:
                            expanded = artifact_hooks.expand_artifact(artifact_id, thread_id)
                            if expanded:
                                task['result'] = expanded
                                logger.info(f"Expanded artifact {artifact_id} for LLM context")
                        except Exception as e:
                            logger.warning(f"Failed to expand artifact {artifact_id}: {e}")
    
    # Log the summarization for debugging
    original_size = len(json.dumps(completed_tasks, default=str))
    summarized_size = len(json.dumps(summarized_results, default=str))
    logger.info(f"📊 Context optimization: {original_size} chars -> {summarized_size} chars ({100 - (summarized_size/original_size*100):.1f}% reduction)")
    
    prompt = f'''
    You are the Orbimesh Orchestrator. Generate a UNIFIED response with both text and optional canvas content.
    
    **USER'S REQUEST:** "{state['original_prompt']}"
    
    **COMPLETED TASKS:**
    {json.dumps(summarized_results, indent=2)}
    
    **YOUR JOB: Generate a complete response in ONE output**
    
    1. **response_text**: A clear, human-readable summary for the user
       - Base ONLY on actual results (no hallucination)
       - If data is missing/null, say so clearly
       - Be factual and accurate
       - **IMPORTANT**: If a task shows "edit_status": "success" or result contains "successfully", the operation WAS completed successfully
       - **IMPORTANT**: If you see "Document edited successfully" in the results, acknowledge the successful edit in your response
       
       **FORMATTING REQUIREMENTS for response_text:**
       - Use clear paragraph breaks for readability
       - Use bullet points (•) or numbered lists for multiple items
       - **Use markdown tables for structured data** (events, schedules, comparisons, lists with multiple attributes)
       - Bold important information using **text**
       - Keep paragraphs concise (2-3 sentences max)
       - Add line breaks between sections for better visual separation
       - Use proper capitalization and punctuation
       - Structure: Brief intro → Key findings → Next steps (if applicable)
       
       **TABLE FORMATTING:**
       When presenting structured data, use markdown tables:
       | Column 1 | Column 2 | Column 3 |
       |----------|----------|----------|
       | Data 1   | Data 2   | Data 3   |
    
    2. **canvas_required**: Decide if canvas visualization would help
       - true: For games, visualizations, interactive content, data displays
       - false: For simple text answers
    
    3. **canvas_type** (if canvas_required=true):
       - "html": For interactive elements, games, visualizations
       - "markdown": For formatted documents
    
    4. **canvas_content** (if canvas_required=true):
       - Generate the COMPLETE HTML or Markdown content
       - For HTML: Include full <!DOCTYPE html> structure with styles and scripts
       - Make it self-contained and functional
       - Use CDN links for external libraries if needed
    
    **CANVAS GUIDELINES:**
    - Use canvas for: games, charts, interactive demos, data visualizations
    - Keep response_text concise if canvas is used (reference the canvas)
    - Make canvas content visually appealing and functional
    
    **CRITICAL RULES:**
    - Base everything on actual task results. Do not invent data.
    - If results show "success" or "edited successfully", acknowledge the successful completion
    - Do NOT say "no confirmation was received" if the task actually completed successfully
    
    Output as JSON conforming to FinalResponse schema.
    '''
    
    try:
        # UNIFIED GENERATION: Get everything in one LLM call
        from schemas import FinalResponse
        
        response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, FinalResponse)
        
        logger.info(f"UNIFIED RESPONSE: canvas_required={response.canvas_required}")
        
        # Create AI message with timestamp metadata
        import hashlib
        timestamp = time.time()
        unique_string = f"ai:{response.response_text}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        ai_message = AIMessage(
            content=response.response_text,
            additional_kwargs={"timestamp": timestamp, "id": msg_id}
        )
        
        # Use MessageManager to add message without duplicates
        from orchestrator.message_manager import MessageManager
        existing_messages = state.get("messages", [])
        updated_messages = MessageManager.add_message(existing_messages, ai_message)
        
        if response.canvas_required and response.canvas_content:
            logger.info(f"Canvas generated: type={response.canvas_type}, content_length={len(response.canvas_content)}")
            
            # --- ARTIFACT INTEGRATION: Store canvas as artifact ---
            canvas_content = response.canvas_content
            canvas_artifact_ref = None
            if ARTIFACT_INTEGRATION_ENABLED and artifact_hooks:
                try:
                    # Get thread_id from state or generate one
                    thread_id = state.get("thread_id") or f"canvas_{int(time.time())}"
                    # Note: This should be awaited but function is not async - skipping for now
                    # compressed = await artifact_hooks.on_canvas_generated(...)
                    compressed = None  # Disabled until function is made async
                    if '_artifact_ref' in compressed:
                        canvas_artifact_ref = compressed['_artifact_ref']
                        logger.info(f"Stored canvas as artifact: {canvas_artifact_ref['id']}")
                except Exception as artifact_err:
                    logger.warning(f"Failed to store canvas as artifact: {artifact_err}")
            
            return {
                "final_response": response.response_text,
                "messages": updated_messages,
                "has_canvas": True,
                "canvas_type": response.canvas_type,
                "canvas_content": canvas_content,  # Keep full content for immediate display
                "_canvas_artifact_ref": canvas_artifact_ref  # Store reference for later retrieval
            }
        else:
            logger.info("Text-only response generated")
            # Preserve existing canvas state if it was set by previous nodes
            result = {
                "final_response": response.response_text,
                "messages": updated_messages,
            }
            if not state.get('has_canvas'):
                result["has_canvas"] = False
                result["canvas_type"] = None
                result["canvas_content"] = None
            return result
            
    except Exception as e:
        logger.error(f"Unified response generation failed: {e}. Falling back to simple text.")
        # Fallback: Generate simple text response
        text_result = generate_text_answer(state)
        result = {
            "final_response": text_result.get('final_response', 'I apologize, but I encountered an error generating the response.'),
            "messages": text_result.get('messages', []),
        }
        # Preserve existing canvas state if it was set by previous nodes
        if not state.get('has_canvas'):
            result["has_canvas"] = False
            result["canvas_type"] = None
            result["canvas_content"] = None
        return result



def load_conversation_history(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {}

    # CRITICAL: Preserve uploaded_files from incoming state (set by main.py)
    incoming_uploaded_files = state.get("uploaded_files", [])
    logger.info(f"🔍 LOAD_HISTORY: Preserving {len(incoming_uploaded_files)} uploaded files from incoming state")

    # Check if this is a resume after approval
    is_resuming_after_approval = state.get("plan_approved") == True
    
    if is_resuming_after_approval:
        print(f"!!! LOAD_HISTORY: Resuming after approval - will load history for context !!!")
        logger.info(f"Resuming after approval for thread {thread_id}, loading history for context")
        # Continue to load history for context, but preserve critical state fields
    
    # If messages already exist in state (from checkpointer) and we're not resuming after approval
    # This prevents duplicate messages when continuing a conversation
    if state.get("messages") and len(state.get("messages", [])) > 0 and not is_resuming_after_approval:
        logger.info(f"Messages already exist in state for thread {thread_id}, skipping file load")
        # Still preserve uploaded_files even when skipping file load
        return {"uploaded_files": incoming_uploaded_files}

    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")

    if not os.path.exists(history_path):
        # Preserve uploaded_files even when no history file exists
        return {"uploaded_files": incoming_uploaded_files}

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {history_path}")
                return {"messages": []}

        if not isinstance(data, dict):
            logger.error(f"Conversation data is not a dictionary: {type(data)}")
            return {"messages": []}
        
        # --- ARTIFACT INTEGRATION: Expand artifact references after loading ---
        if ARTIFACT_INTEGRATION_ENABLED and artifact_hooks:
            try:
                # Note: This should be awaited but function is not async - skipping for now
                # data = await artifact_hooks.after_load(data)
                # logger.info(f"Expanded artifacts for thread {thread_id}")
                pass  # Disabled until function is made async
            except Exception as artifact_err:
                logger.warning(f"Artifact expansion failed: {artifact_err}")

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            logger.error(f"Messages data is not a list: {type(messages)}")
            return {"messages": []}

        valid_messages = []
        for msg_data in messages:
            if not isinstance(msg_data, dict):
                continue

            msg_type = msg_data.get("type", "").lower()
            content = msg_data.get("content", "")
            metadata = msg_data.get("metadata", {})
            msg_id = msg_data.get("id")
            timestamp = msg_data.get("timestamp")

            try:
                # Prepare additional_kwargs with timestamp and id
                additional_kwargs = metadata.copy() if metadata else {}
                if timestamp:
                    additional_kwargs['timestamp'] = timestamp
                if msg_id:
                    additional_kwargs['id'] = msg_id
                
                if msg_type == "user":
                    msg = HumanMessage(content=content, additional_kwargs=additional_kwargs)
                elif msg_type == "assistant":
                    msg = AIMessage(content=content, additional_kwargs=additional_kwargs)
                else:
                    msg = SystemMessage(content=content, additional_kwargs=additional_kwargs)

                valid_messages.append(msg)

            except Exception as e:
                logger.warning(f"Failed to create message object: {e}")
                continue

        # Deduplicate messages before returning
        from orchestrator.message_manager import MessageManager
        valid_messages = MessageManager.deduplicate_messages(valid_messages)
        
        logger.info(f"Successfully loaded {len(valid_messages)} messages for conversation {thread_id}")
        
        # When resuming after approval, only return messages (don't overwrite final_response)
        if state.get("plan_approved"):
            print(f"!!! LOAD_HISTORY: Loaded {len(valid_messages)} messages for context after approval !!!")
            return {
                "messages": valid_messages,
                "uploaded_files": incoming_uploaded_files  # Preserve uploaded files
            }
        
        # Load the pre-seeded plan if it exists (for workflow re-execution)
        task_plan = data.get("task_plan", [])
        task_agent_pairs = data.get("task_agent_pairs", [])
        original_prompt = data.get("original_prompt", "")
        
        # Load approval-related flags for saved workflow execution
        needs_approval = data.get("needs_approval", False)
        plan_approved = data.get("plan_approved", False)
        approval_required = data.get("approval_required", False)
        pending_user_input = data.get("pending_user_input", False)
        question_for_user = data.get("question_for_user")
        status = data.get("status")
        
        if task_plan:
            logger.info(f"Loaded pre-seeded task_plan with {len(task_plan)} batches")
        if task_agent_pairs:
            logger.info(f"Loaded pre-seeded task_agent_pairs with {len(task_agent_pairs)} tasks")
        if needs_approval:
            logger.info(f"Loaded needs_approval=True - workflow requires approval before execution")
        
        return {
            "messages": valid_messages,
            "thread_id": data.get("thread_id"),
            "final_response": data.get("final_response"),
            "task_plan": task_plan,
            "task_agent_pairs": task_agent_pairs,
            "original_prompt": original_prompt if original_prompt else state.get("original_prompt", ""),
            "needs_approval": needs_approval,
            "plan_approved": plan_approved,
            "approval_required": approval_required,
            "pending_user_input": pending_user_input,
            "question_for_user": question_for_user,
            "status": status,
            "uploaded_files": incoming_uploaded_files  # Preserve uploaded files from incoming state
        }

    except Exception as e:
        logger.error(f"Failed to load conversation history for {thread_id}: {e}")
        return {"messages": [], "uploaded_files": incoming_uploaded_files}

def get_serializable_state(state: dict | State, thread_id: str) -> dict:
    """
    Takes the current graph state and a thread_id, and returns a dictionary
    that is fully JSON-serializable, containing all necessary information
    for the frontend to render the conversation and its metadata.
    """
    # Convert State object to dict if needed
    if not isinstance(state, dict):
        # Handle the case where state might be a State object (TypedDict)
        state_dict = {}
        # Get all attributes that exist in the state
        for key in ['messages', 'task_agent_pairs', 'final_response', 'pending_user_input', 
                   'question_for_user', 'original_prompt', 'completed_tasks', 'parsed_tasks',
                   'uploaded_files', 'task_plan', 'canvas_content', 'canvas_type', 'has_canvas',
                   'needs_canvas']:
            if hasattr(state, key):
                state_dict[key] = getattr(state, key)
            # Also try to get from __getitem__ if it's a dict-like object
            try:
                state_dict[key] = state[key]
            except (KeyError, TypeError):
                pass
        state = state_dict
    
    # Safely serialize messages
    messages = state.get("messages", [])
    try:
        # This handles LangChain message objects
        langchain_messages = messages_to_dict(messages)
        
        # Convert LangChain format to frontend format and filter empty messages
        serializable_messages = []
        for msg in langchain_messages:
            msg_type = msg.get('type', '')
            msg_data = msg.get('data', {})
            content = msg_data.get('content', '')
            
            # Skip empty assistant/ai messages
            if msg_type == 'ai' and (not content or content.strip() == ''):
                logger.debug(f"Skipping empty AI message: {msg}")
                continue
            
            # Convert to frontend format
            # Generate DETERMINISTIC ID for each message to avoid duplicates
            # Check for ID and timestamp in additional_kwargs first (where we store them)
            additional_kwargs = msg_data.get('additional_kwargs', {})
            msg_id = msg.get('id') or msg_data.get('id') or additional_kwargs.get('id')
            timestamp = msg_data.get('timestamp') or additional_kwargs.get('timestamp') or time.time()
            
            if not msg_id:
                # Generate deterministic ID based on content + type + timestamp
                # This ensures the same message always gets the same ID
                import hashlib
                unique_string = f"{msg_type}:{content}:{timestamp}"
                msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
            
            # Convert timestamp to milliseconds for frontend (JavaScript expects ms)
            # If timestamp is in seconds (< year 3000), convert to milliseconds
            timestamp_ms = timestamp * 1000 if timestamp < 10000000000 else timestamp
            
            frontend_msg = {
                'id': msg_id,
                'type': 'assistant' if msg_type == 'ai' else 'user' if msg_type == 'human' else 'system',
                'content': content,
                'timestamp': timestamp_ms
            }
            serializable_messages.append(frontend_msg)
            logger.debug(f"Converted message: type={msg_type}, id={msg_id}, content_length={len(content)}")
            
    except Exception as e:
        logger.warning(f"Failed to serialize messages with messages_to_dict: {e}")
        # Fallback for plain dicts or other formats
        serializable_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Filter out empty assistant messages
                if msg.get('type') == 'assistant' and (not msg.get('content') or msg.get('content', '').strip() == ''):
                    continue
                serializable_messages.append(msg)
            elif hasattr(msg, 'dict'):
                msg_dict = msg.dict()
                # Filter out empty assistant messages
                if msg_dict.get('type') == 'assistant' and (not msg_dict.get('content') or msg_dict.get('content', '').strip() == ''):
                    continue
                serializable_messages.append(msg_dict)
            else:
                serializable_messages.append(str(msg)) # Failsafe
    
    # If canvas exists, attach canvas metadata to the last assistant message
    if state.get("has_canvas") and state.get("canvas_content") and serializable_messages:
        # Find the last assistant message
        for i in range(len(serializable_messages) - 1, -1, -1):
            if serializable_messages[i].get('type') == 'assistant':
                serializable_messages[i]['canvas_content'] = state.get('canvas_content')
                serializable_messages[i]['canvas_type'] = state.get('canvas_type')
                serializable_messages[i]['has_canvas'] = True
                logger.info(f"Attached canvas metadata to message {serializable_messages[i].get('id')}")
                break

    # Use the serialize_complex_object helper for other potentially complex fields
    # This ensures nested Pydantic models, HttpUrl, etc., are converted correctly.
    logger.info(f"Serialized {len(serializable_messages)} messages for thread {thread_id}")
    logger.info(f"Final response length: {len(state.get('final_response', '')) if state.get('final_response') else 0}")
    
    return {
        "thread_id": thread_id,
        "status": "pending_user_input" if state.get("pending_user_input") else "completed",
        "messages": serializable_messages,
        "task_agent_pairs": serialize_complex_object(state.get("task_agent_pairs", [])),
        "task_plan": serialize_complex_object(state.get("task_plan", [])),  # Save as task_plan for consistency
        "final_response": state.get("final_response"),
        "pending_user_input": state.get("pending_user_input", False),
        "question_for_user": state.get("question_for_user"),
        "needs_approval": state.get("needs_approval", False),
        "plan_approved": state.get("plan_approved", False),
        "approval_required": state.get("approval_required", False),
        # Metadata for the sidebar
        "metadata": {
            "original_prompt": state.get("original_prompt"),
            "completed_tasks": serialize_complex_object(state.get("completed_tasks", [])),
            "parsed_tasks": serialize_complex_object(state.get("parsed_tasks", [])),
            "currentStage": "completed",
            "stageMessage": "Orchestration completed successfully!",
            "progress": 100
        },
        # Attachments for the sidebar
        "uploaded_files": serialize_complex_object(state.get("uploaded_files", [])),
        # Plan for the sidebar (deprecated, use task_plan instead)
        "plan": serialize_complex_object(state.get("task_plan", [])),
        # Canvas fields for the sidebar
        "has_canvas": state.get("has_canvas", False),
        "canvas_content": state.get("canvas_content"),
        "canvas_data": state.get("canvas_data"),  # Structured canvas data
        "canvas_type": state.get("canvas_type"),
        "canvas_title": state.get("canvas_title"),  # Canvas title
        "needs_canvas": state.get("needs_canvas", False),
        "timestamp": time.time(),
    }

def generate_conversation_title(prompt: str, messages: List = None) -> str:
    """
    Generate a concise title for the conversation using LLM.
    Similar to ChatGPT's title generation.
    """
    try:
        # Use Cerebras for fast title generation
        llm = ChatCerebras(model="llama3.1-8b", temperature=0.3)
        
        # Get first user message if prompt is empty
        if not prompt and messages:
            for msg in messages:
                if hasattr(msg, 'type') and msg.type == "human":
                    prompt = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("type") == "human":
                    prompt = msg.get("content", "")
                    break
        
        if not prompt:
            return None
        
        # Generate title (max 6 words, descriptive)
        title_prompt = f"""Generate a short, descriptive title (maximum 6 words) for a conversation that starts with:

"{prompt[:200]}"

Rules:
- Maximum 6 words
- Descriptive and specific
- No quotes or punctuation at the end
- Capitalize first letter only

Title:"""
        
        response = llm.invoke(title_prompt)
        title = response.content.strip()
        
        # Clean up the title
        title = title.replace('"', '').replace("'", "").strip()
        if title.endswith('.'):
            title = title[:-1]
        
        # Limit to 60 chars
        if len(title) > 60:
            title = title[:57] + "..."
        
        logger.info(f"Generated title: {title}")
        return title
        
    except Exception as e:
        logger.error(f"Failed to generate title: {e}")
        # Fallback: use first 50 chars of prompt
        if prompt:
            return prompt[:50] + "..." if len(prompt) > 50 else prompt
        return None

def save_conversation_history(state: State, config: RunnableConfig):
    """
    Saves the full, serializable state of the conversation to a JSON file.
    This is the single source of truth for conversation history.
    Also registers the conversation in user_threads table for ownership tracking.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        logger.warning("No thread_id in config, cannot save history.")
        return {}
        
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
    
    try:
        # Convert State object to dict if needed
        state_dict = {}
        if not isinstance(state, dict):
            # Handle the case where state might be a State object (TypedDict)
            for key in ['messages', 'task_agent_pairs', 'final_response', 'pending_user_input', 
                       'question_for_user', 'original_prompt', 'completed_tasks', 'parsed_tasks',
                       'uploaded_files', 'task_plan', 'canvas_content', 'canvas_type', 'has_canvas',
                       'owner']:
                if hasattr(state, key):
                    state_dict[key] = getattr(state, key)
                # Also try to get from __getitem__ if it's a dict-like object
                try:
                    state_dict[key] = state[key]
                except (KeyError, TypeError):
                    pass
        else:
            state_dict = state
        
        # Create the fully serializable state object
        serializable_state = get_serializable_state(state_dict, thread_id)
        
        # --- ARTIFACT INTEGRATION: Compress large fields before saving ---
        if ARTIFACT_INTEGRATION_ENABLED and artifact_hooks:
            try:
                # Note: This should be awaited but function is not async - skipping for now
                # serializable_state = await artifact_hooks.before_save(serializable_state, thread_id)
                # logger.info(f"Applied artifact compression for thread {thread_id}")
                pass  # Disabled until function is made async
            except Exception as artifact_err:
                logger.warning(f"Artifact compression failed, saving uncompressed: {artifact_err}")

        # Write to file in a consistent JSON format
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved conversation history for thread {thread_id}.")

        # Register in user_threads table for ownership tracking
        try:
            owner = state_dict.get("owner") or config.get("configurable", {}).get("owner")
            if owner:
                user_id = None
                if isinstance(owner, str):
                    user_id = owner
                elif isinstance(owner, dict):
                    user_id = owner.get("user_id") or owner.get("sub") or owner.get("id")
                
                if user_id:
                    from database import SessionLocal
                    from models import UserThread
                    
                    db = SessionLocal()
                    try:
                        # Check if already exists
                        existing = db.query(UserThread).filter_by(thread_id=thread_id).first()
                        
                        if not existing:
                            # Create new entry with AI-generated title
                            original_prompt = state_dict.get("original_prompt", "")
                            messages = state_dict.get("messages", [])
                            
                            # Generate title using LLM
                            title = generate_conversation_title(original_prompt, messages)
                            
                            # Fallback if generation fails
                            if not title:
                                if original_prompt:
                                    title = original_prompt[:50] + "..." if len(original_prompt) > 50 else original_prompt
                                else:
                                    title = "Untitled Conversation"
                            
                            user_thread = UserThread(
                                user_id=user_id,
                                thread_id=thread_id,
                                title=title
                            )
                            db.add(user_thread)
                            db.commit()
                            logger.info(f"Registered conversation {thread_id} for user {user_id} with title: '{title}'")
                        else:
                            # Only update timestamp
                            from datetime import datetime
                            existing.updated_at = datetime.utcnow()
                            db.commit()
                            logger.info(f"Updated conversation {thread_id} timestamp")
                    finally:
                        db.close()
                else:
                    logger.warning(f"Could not extract user_id from owner: {owner}")
            else:
                logger.warning(f"No owner information for thread {thread_id}, skipping user_threads registration")
        except Exception as db_err:
            logger.error(f"Failed to register conversation in user_threads: {db_err}", exc_info=True)
            # Don't fail the entire save operation if DB registration fails

    except Exception as e:
        logger.error(f"Failed to write conversation history for {thread_id} to {history_path}: {e}")
        logger.exception("Full error details:")

    return {}

# --- Routing Functions ---
def route_after_search(state: State):
    '''Route after agent directory search based on whether agents were found'''
    if state.get("parsing_error_feedback"):
        if state.get("parse_retry_count", 0) >= 3:
            logger.warning("Max parse retries reached. Asking user for clarification.")
            return "ask_user"
        else:
            logger.info("Retrying parse_prompt.")
            return "parse_prompt"
    return "rank_agents"

def route_after_approval(state: State):
    '''Routes after plan approval checkpoint.'''
    approval_required = state.get("approval_required")
    pending_user_input = state.get("pending_user_input")
    
    logger.info(f"=== ROUTING AFTER APPROVAL: approval_required={approval_required}, pending_user_input={pending_user_input} ===")
    
    if approval_required and pending_user_input:
        logger.info("Routing to ask_user for plan approval.")
        return "ask_user"
    else:
        logger.info(f"Plan approved or no approval needed. Routing to validate_plan_for_execution.")
        return "validate_plan_for_execution"

def route_after_validation(state: State):
    '''Simple routing after validation.'''
    replan_reason = state.get("replan_reason")
    pending_user_input = state.get("pending_user_input")
    
    print(f"!!! ROUTER AFTER VALIDATION: replan={replan_reason}, pending={pending_user_input} !!!")
    
    if replan_reason:
        logger.info("Replan needed. Routing back to plan_execution.")
        return "plan_execution"
    if pending_user_input:
        logger.info("Pending user input. Routing to ask_user.")
        return "ask_user"
    else:
        logger.info("Plan is valid. Routing to execute_batch.")
        return "execute_batch"

def analyze_request(state: State):
    """Sophisticated analysis of user request to determine processing approach."""
    logger.info("Performing sophisticated analysis of user request...")
    logger.info(f"🔍 ANALYZE_REQUEST ENTRY: uploaded_files in state = {'uploaded_files' in state}")
    logger.info(f"🔍 ANALYZE_REQUEST ENTRY: uploaded_files count = {len(state.get('uploaded_files', []))}")
    
    # CANVAS CONFIRMATION FIX: Check if this is a confirmation follow-up message
    pending_confirmation = state.get("pending_confirmation", False)
    pending_task = state.get("pending_confirmation_task")
    original_prompt = state.get('original_prompt', '').lower()
    
    # Detect confirmation keywords in the message
    confirmation_keywords = ['apply', 'confirm', 'yes', 'proceed', 'continue', 'without showing preview', 'without preview']
    is_confirmation_message = any(keyword in original_prompt for keyword in confirmation_keywords)
    
    logger.info(f"🔍 CONFIRMATION CHECK: pending_confirmation={pending_confirmation}, pending_task={pending_task}, original_prompt='{original_prompt}', is_confirmation_message={is_confirmation_message}")
    print(f"!!! ANALYZE_REQUEST: pending_confirmation={pending_confirmation}, is_confirmation_message={is_confirmation_message}, original_prompt='{original_prompt}' !!!")
    
    # If there's a pending confirmation and user is confirming, clear the confirmation state and proceed with execution
    if pending_confirmation and is_confirmation_message:
        logger.info(f"✅ User confirmed canvas action for task '{pending_task}'. Clearing confirmation state and proceeding with execution.")
        print(f"!!! CONFIRMATION DETECTED: Clearing pending_confirmation and routing to execution !!!")
        print(f"!!! RETURNING: pending_confirmation=False, skip_preview_on_next_execution=True, needs_complex_processing=True !!!")
        
        # Clear confirmation state and set flag to skip preview on next execution
        # DON'T set canvas_confirmation_action as it routes to broken execute_confirmed_task node
        return {
            "pending_confirmation": False,
            "pending_confirmation_task": None,
            "canvas_requires_confirmation": False,
            "skip_preview_on_next_execution": True,  # Flag to inject show_preview=False in plan_execution
            "needs_complex_processing": True,  # Force complex processing to execute the task
            "analysis_reasoning": f"User confirmed the canvas action for task '{pending_task}'. Proceeding with execution without preview."
        }
    
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    # Build comprehensive context from conversation history
    history_context = ""
    if state.get('messages'):
        # Include recent conversation turns to provide context
        recent_messages = state['messages'][-20:]  # Last 20 messages
        for msg in recent_messages:
            if hasattr(msg, 'type') and msg.type == "human":
                history_context += f"User: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history_context += f"Assistant: {msg.content}\n"
    
    # Include uploaded files context
    files_context = ""
    uploaded_files = state.get('uploaded_files', [])
    if uploaded_files:
        files_info = []
        for file_obj in uploaded_files:
            # Convert dict to FileObject if needed for attribute access
            if isinstance(file_obj, dict):
                file_obj = FileObject.model_validate(file_obj)
            file_info = f"- {file_obj.file_name} ({file_obj.file_type}) at {file_obj.file_path}"
            if file_obj.file_type == 'document' and file_obj.vector_store_path:
                file_info += f", vector store: {file_obj.vector_store_path}"
            files_info.append(file_info)
        files_context = "Uploaded files:\n" + "\n".join(files_info) + "\n\n"
    
    # Include completed tasks context
    tasks_context = ""
    completed_tasks = state.get('completed_tasks', [])
    if completed_tasks:
        tasks_info = [f"- {task.get('task_name', '')}: {str(task.get('result', ''))[:200]}..." for task in completed_tasks[-3:]]  # Last 3 tasks
        tasks_context = "Previous results:\n" + "\n".join(tasks_info) + "\n\n"
    
    prompt = f'''
    You are the Orbimesh Orchestrator, an intelligent AI system that coordinates multiple specialized agents to complete user requests.
    
    **CONTEXT:**
    {files_context}
    {tasks_context}
    Conversation history:
    {history_context}
    
    User's current request: "{state['original_prompt']}"
    
    **YOUR TASK:**
    Determine if this request requires delegating work to specialized agents (complex processing) or if you can answer it directly (simple response).
    
    **SIMPLE REQUESTS** (you can answer directly without agents):
    - Greetings, thanks, acknowledgments: "hi", "thanks", "ok", "got it"
    - General knowledge questions: "what is Python?", "explain recursion"
    - Clarification questions: "what do you mean?", "can you explain?"
    - Creating interactive content with your built-in Canvas (games, visualizations, demos) that don't need external data
    
    **COMPLEX REQUESTS** (require specialized agents):
    - ANY work with uploaded files or documents (editing, analyzing, reading, converting, formatting)
    - Fetching external data (stock prices, news, weather, company info, web searches)
    - Performing calculations, analysis, or transformations on data
    - Sending emails, browsing websites, or interacting with external services
    - Multi-step workflows or tasks that build on previous results
    
    **CRITICAL RULES:**
    - If there are uploaded files in context and the request mentions them or their content, it ALWAYS needs complex processing
    - If the request is a follow-up to a previous task (continuing work on the same file/topic), it needs complex processing
    - Action verbs like "edit", "change", "convert", "analyze", "fetch", "get", "find", "search", "calculate" indicate complex processing
    - When in doubt, choose complex processing - it's better to delegate than to give an incorrect simple answer
    
    Respond with a JSON object:
    {{
        "needs_complex_processing": true/false,
        "reasoning": "Brief explanation for the decision",
        "response": "If needs_complex_processing is false, provide a direct response to the user's request"
    }}
    '''
    
    try:
        response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, AnalysisResult)
        logger.info(f"Analysis result: needs_complex_processing={response.needs_complex_processing}")
        
        # Ensure we return a complete state update with all required fields
        result = {
            "needs_complex_processing": response.needs_complex_processing,
            "analysis_reasoning": response.reasoning
        }
        
        # Handle final_response properly - only set it for simple requests
        if not response.needs_complex_processing and response.response:
            result["final_response"] = response.response
        # For complex requests, don't set final_response at all (let it be None/undefined)
        # This ensures generate_text_answer will handle complex requests properly
        
        logger.info(f"🔍 ANALYZE_REQUEST EXIT: Returning result with keys = {list(result.keys())}")
        logger.info(f"🔍 ANALYZE_REQUEST EXIT: NOT touching uploaded_files (should be preserved)")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}. Defaulting to complex processing.")
        return {
            "needs_complex_processing": True, 
            "analysis_reasoning": f"Analysis failed: {e}",
            "final_response": ""  # Explicitly clear for error case
        }


def route_after_load_history(state: State):
    """Route after loading history: skip orchestration for pre-approved workflows."""
    plan_approved = state.get("plan_approved", False)
    has_plan = state.get("task_plan") and len(state.get("task_plan", [])) > 0
    needs_approval = state.get("needs_approval", False)
    
    print(f"!!! ROUTE AFTER LOAD HISTORY: plan_approved={plan_approved}, has_plan={has_plan}, needs_approval={needs_approval} !!!")
    print(f"!!! STATE KEYS: {list(state.keys())} !!!")
    print(f"!!! TASK_PLAN VALUE: {state.get('task_plan')} !!!")
    
    # If workflow is pre-approved (saved workflow execution), skip directly to validation
    if plan_approved and has_plan:
        print("!!! PRE-APPROVED WORKFLOW DETECTED = SKIP TO VALIDATION !!!")
        logger.info("Pre-approved workflow detected. Skipping orchestration and jumping to validation.")
        return "validate_plan_for_execution"
    
    # Otherwise, proceed with normal orchestration flow
    print("!!! NO PRE-APPROVED PLAN = PROCEED TO ANALYSIS !!!")
    logger.info("No pre-approved plan. Proceeding to analyze_request.")
    return "analyze_request"


def route_after_analysis(state: State):
    """Route based on whether we have an existing plan or need to create one."""
    has_plan = state.get("task_plan") and len(state.get("task_plan", [])) > 0
    planning_mode = state.get("planning_mode", False)
    needs_complex = state.get("needs_complex_processing")
    plan_approved = state.get("plan_approved", False)
    uploaded_files_count = len(state.get("uploaded_files", []))
    
    logger.info(f"🔍 ROUTE_AFTER_ANALYSIS: uploaded_files count = {uploaded_files_count}")
    print(f"!!! ROUTE AFTER ANALYSIS: has_plan={has_plan}, planning_mode={planning_mode}, needs_complex={needs_complex}, plan_approved={plan_approved}, uploaded_files={uploaded_files_count} !!!")
    
    # If we have a plan and planning mode is OFF, skip to validation (normal continuation)
    if has_plan and not planning_mode:
        print("!!! HAS PLAN + NO PLANNING MODE = SKIP TO VALIDATION !!!")
        logger.info("Plan exists and planning mode is off. Skipping to validation.")
        return "validate_plan_for_execution"
    
    # Otherwise, normal routing (create new plan)
    if needs_complex:
        if state.get("uploaded_files"):
            return "preprocess_files"
        else:
            return "parse_prompt"
    else:
        return "generate_final_response"


def route_after_parse(state: State):
    '''Route after parsing: handle direct responses, no tasks, or proceed to search.'''
    # OPTIMIZATION: Short-circuit for direct responses (chitchat)
    if state.get('needs_complex_processing') is False and state.get('final_response'):
        logger.info("Direct response available. Short-circuiting to save_history.")
        return "save_history"
    
    if not state.get('parsed_tasks'):
        logger.warning("No tasks were parsed from prompt. Routing to ask_user for clarification.")
        return "ask_user"
    return "agent_directory_search"

def should_continue_or_finish(state: State):
    '''REACTIVE ROUTER: Runs after execution and evaluation to decide the next step.'''
    pending = state.get("pending_user_input")
    pending_confirmation = state.get("pending_confirmation")
    task_plan = state.get('task_plan')
    eval_status = state.get("eval_status")
    replan_reason = state.get("replan_reason")
    
    print(f"!!! SHOULD_CONTINUE_OR_FINISH: eval_status={eval_status}, pending={pending}, pending_confirmation={pending_confirmation}, replan_reason={replan_reason}, task_plan_length={len(task_plan) if task_plan else 0} !!!")
    logger.info(f"Reactive Router: eval_status={eval_status}, pending={pending}, pending_confirmation={pending_confirmation}, replan={bool(replan_reason)}, plan_length={len(task_plan) if task_plan else 0}")
    
    # CANVAS CONFIRMATION FIX: If confirmation is pending, go to generate_final_response to show canvas
    if pending_confirmation:
        print(f"!!! ROUTING TO GENERATE_FINAL_RESPONSE (pending_confirmation=True) !!!")
        logger.info("Routing to generate_final_response to display canvas with confirmation button")
        return "generate_final_response"
    
    # REACTIVE LOOP: If task failed, trigger auto-replan
    if eval_status == "failed" and replan_reason:
        print(f"!!! ROUTING TO PLAN_EXECUTION (auto-replan) !!!")
        logger.info("Task failed. Routing to plan_execution for auto-replan.")
        return "plan_execution"
    
    if pending:
        # If the evaluation requires user input, go to ask_user
        print(f"!!! ROUTING TO ASK_USER (pending_user_input=True) !!!")
        logger.info("Routing to ask_user due to pending_user_input")
        return "ask_user"
    
    if not task_plan:
        # If the plan is empty and evaluation passed, we are done
        print(f"!!! ROUTING TO GENERATE_FINAL_RESPONSE (plan complete) !!!")
        logger.info("Execution plan is complete. Routing to generate_final_response.")
        return "generate_final_response"
    else:
        # If there are more tasks and evaluation passed, continue to next batch
        print(f"!!! ROUTING TO VALIDATE (more batches) !!!")
        logger.info("Plan has more batches. Routing back to validation for the next batch.")
        return "validate_plan_for_execution"

# --- Build the State Graph ---
builder = StateGraph(State)

builder.add_node("load_history", load_conversation_history)
builder.add_node("save_history", save_conversation_history)
builder.add_node("analyze_request", analyze_request)
builder.add_node("parse_prompt", parse_prompt)  # OPTIMIZED: Super-parser (3-in-1)
builder.add_node("preprocess_files", preprocess_files)
builder.add_node("agent_directory_search", agent_directory_search)  # OPTIMIZED: Internal DB search
builder.add_node("rank_agents", rank_agents)  # OPTIMIZED: Enhanced LLM ranking
builder.add_node("plan_execution", plan_execution)
builder.add_node("pause_for_plan_approval", pause_for_plan_approval)
builder.add_node("validate_plan_for_execution", validate_plan_for_execution)
builder.add_node("execute_batch", execute_batch)  # OPTIMIZED: Parameter matching + caching
builder.add_node("evaluate_agent_response", evaluate_agent_response)  # OPTIMIZED: Reactive evaluation
builder.add_node("ask_user", ask_user)
builder.add_node("generate_final_response", generate_final_response)  # OPTIMIZED: Unified text + canvas
builder.add_node("execute_confirmed_task", execute_confirmed_task)  # Canvas confirmation flow
# Note: render_canvas_output node removed - now integrated into generate_final_response

builder.add_edge(START, "load_history")
builder.add_conditional_edges("load_history", route_after_load_history, {
    "analyze_request": "analyze_request",
    "validate_plan_for_execution": "validate_plan_for_execution"
})

builder.add_conditional_edges("analyze_request", route_after_analysis, {
    "preprocess_files": "preprocess_files",
    "parse_prompt": "parse_prompt",
    "generate_final_response": "generate_final_response",
    "validate_plan_for_execution": "validate_plan_for_execution",
    "execute_batch": "execute_batch"
})

builder.add_edge("preprocess_files", "parse_prompt")

builder.add_conditional_edges("parse_prompt", route_after_parse, {
    "ask_user": "ask_user",
    "agent_directory_search": "agent_directory_search",
    "save_history": "save_history"  # Short-circuit for direct responses
})

builder.add_edge("agent_directory_search", "rank_agents")
builder.add_edge("rank_agents", "plan_execution")

# Simple routing: if planning mode, stop for approval; otherwise continue
def route_after_plan_creation(state: State):
    '''Stop for approval if planning mode, otherwise continue to execution.'''
    planning_mode = state.get("planning_mode", False)
    plan_approved = state.get("plan_approved", False)
    
    print(f"!!! ROUTE AFTER PLAN: planning_mode={planning_mode}, plan_approved={plan_approved} !!!")
    logger.info(f"=== ROUTE AFTER PLAN: planning_mode={planning_mode}, plan_approved={plan_approved} ===")
    
    # If planning mode is ON (regardless of plan_approved), pause for approval
    # The execution will happen in the subgraph after approval
    if planning_mode:
        print("!!! PLANNING MODE ON - PAUSING FOR APPROVAL !!!")
        logger.info("=== ROUTING: Planning mode ON. Pausing for approval ===")
        # The plan_execution node already set pending_user_input=True and needs_approval=True
        # Just end the workflow here - execution will happen in subgraph after approval
        return "save_history"
    else:
        print("!!! CONTINUING TO VALIDATION !!!")
        logger.info("=== ROUTING: Planning mode OFF. Continuing to validation ===")
        return "validate_plan_for_execution"  # Normal execution

builder.add_conditional_edges("plan_execution", route_after_plan_creation, {
    "execute_batch": "execute_batch",
    "validate_plan_for_execution": "validate_plan_for_execution",
    "save_history": "save_history"
})

# Route from execute_batch: check if waiting for confirmation or proceed to evaluation
def route_after_execute_batch(state: State):
    '''Check if we're waiting for canvas confirmation, otherwise proceed to evaluation.'''
    pending_confirmation = state.get("pending_confirmation", False)
    canvas_confirmation_action = state.get("canvas_confirmation_action")
    
    # If user confirmed via WebSocket, execute the confirmed task
    if canvas_confirmation_action:
        logger.info(f"🔄 Canvas confirmation received: {canvas_confirmation_action}. Routing to execute_confirmed_task.")
        return "execute_confirmed_task"
    
    # CANVAS CONFIRMATION FIX: If waiting for confirmation, route to generate_final_response
    # This will display the canvas with the confirmation button
    if pending_confirmation:
        logger.info("⏸️ Waiting for canvas confirmation. Routing to generate_final_response to display canvas.")
        return "generate_final_response"
    
    # Normal flow: proceed to evaluation
    return "evaluate_agent_response"

builder.add_conditional_edges("execute_batch", route_after_execute_batch, {
    "evaluate_agent_response": "evaluate_agent_response",
    "execute_confirmed_task": "execute_confirmed_task",
    "generate_final_response": "generate_final_response"  # CANVAS CONFIRMATION FIX: Route to response generation
})

# Route from ask_user based on whether it was for approval or clarification  
def route_after_ask_user(state: State):
    '''Routes after ask_user based on context - approval vs clarification.'''
    plan_approved = state.get("plan_approved")
    pending_user_input = state.get("pending_user_input")
    
    logger.info(f"=== ROUTING AFTER ASK_USER: plan_approved={plan_approved}, pending_user_input={pending_user_input} ===")
    
    # Always end if we're waiting for user input (prevents loops)
    # The only exception is the FIRST time after approval when pending_user_input was just cleared
    if pending_user_input:
        logger.info("=== ROUTING: Workflow paused for user input. Routing to save_history (end) ===")
        return "save_history"
    
    # If plan was approved and we're NOT waiting for input, this is the continuation after approval
    # Route to validation ONLY ONCE
    if plan_approved:
        logger.info("=== ROUTING: Plan approved, continuing to validation (ONE TIME ONLY) ===")
        return "validate_plan_for_execution"
    
    # Default: end the workflow
    logger.info("=== ROUTING: Default route to save_history (end) ===")
    return "save_history"

builder.add_conditional_edges("ask_user", route_after_ask_user, {
    "validate_plan_for_execution": "validate_plan_for_execution",
    "save_history": "save_history"
})

# OPTIMIZATION: Unified response generation - no separate canvas rendering needed
builder.add_edge("generate_final_response", "save_history")
builder.add_edge("save_history", END)

builder.add_conditional_edges("agent_directory_search", route_after_search, {
    "parse_prompt": "parse_prompt", 
    "rank_agents": "rank_agents",
    "ask_user": "ask_user"
})

builder.add_conditional_edges("validate_plan_for_execution", route_after_validation, {
    "execute_batch": "execute_batch",
    "plan_execution": "plan_execution",
    "ask_user": "ask_user"
})

builder.add_conditional_edges("evaluate_agent_response", should_continue_or_finish, {
    "validate_plan_for_execution": "validate_plan_for_execution",
    "generate_final_response": "generate_final_response",
    "ask_user": "ask_user",
    "plan_execution": "plan_execution"  # REACTIVE LOOP: Auto-replan on failure
})

# After confirmed task execution, go to evaluation
builder.add_edge("execute_confirmed_task", "evaluate_agent_response")

# Compile the graph
graph = builder.compile()

def create_graph_with_checkpointer(checkpointer):
    """Attaches a checkpointer to the graph for persistent memory."""
    # Don't use interrupt_before - we'll handle pausing manually with pending_user_input
    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# EXECUTION SUBGRAPH - For post-approval execution in planning mode
# ============================================================================

def create_execution_subgraph(checkpointer):
    """
    Creates a lightweight execution-only subgraph for running approved plans.
    This skips all the planning/parsing/searching steps and goes straight to execution.
    
    Flow: load_history → execute_batch → evaluate → generate_final_response → save_history → END
    """
    from langgraph.graph import StateGraph, END
    
    exec_builder = StateGraph(State)
    
    # Add only the nodes needed for execution
    exec_builder.add_node("load_history", load_conversation_history)
    exec_builder.add_node("execute_batch", execute_batch)
    exec_builder.add_node("evaluate_agent_response", evaluate_agent_response)
    exec_builder.add_node("generate_final_response", generate_final_response)  # OPTIMIZED: Unified response
    exec_builder.add_node("save_history", save_conversation_history)
    
    # Set entry point
    exec_builder.set_entry_point("load_history")
    
    # Simple linear flow for execution
    exec_builder.add_edge("load_history", "execute_batch")
    exec_builder.add_conditional_edges("execute_batch", should_continue_or_finish, {
        "validate_plan_for_execution": "execute_batch",  # More batches
        "generate_final_response": "generate_final_response",
        "ask_user": "generate_final_response"  # Skip asking user, just generate response
    })
    exec_builder.add_edge("evaluate_agent_response", "generate_final_response")
    exec_builder.add_edge("generate_final_response", "save_history")  # OPTIMIZED: Direct to save
    exec_builder.add_edge("save_history", END)
    
    return exec_builder.compile(checkpointer=checkpointer)

# Create both graphs
execution_subgraph = None  # Will be initialized when needed

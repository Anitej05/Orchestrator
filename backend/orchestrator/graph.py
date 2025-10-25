# In Project_Agent_Directory/orchestrator/graph.py

from orchestrator.state import State, CompletedTask
from schemas import (
    ParsedRequest,
    TaskAgentPair,
    ExecutionPlan,
    AgentCard,
    PlannedTask,
    FileObject,
    AnalysisResult
)
from sentence_transformers import SentenceTransformer
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
from models import AgentCapability
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

# --- Imports for Document Processing ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def serialize_complex_object(obj):
    '''Helper function to serialize complex objects consistently'''
    try:
        # First try direct JSON serialization
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # Handle different object types
        if isinstance(obj, HttpUrl):
            return str(obj)  # Convert HttpUrl to string
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
                if all(hasattr(item, '_type') for item in obj if item is not None):
                    # Use LangChain's messages_to_dict for message objects
                    return messages_to_dict(obj)
                else:
                    return [serialize_complex_object(item) for item in obj]
            except:
                pass
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
            if any(keyword in error_msg for keyword in ["429", "rate", "too_many_requests", "high traffic", "queue_exceeded"]):
                logger.warning(f"{current_llm_name} LLM failed with external API issue: {e}. Will try next LLM provider.")
                if is_last_attempt:
                    # If this is the last attempt and it failed, we've exhausted all options
                    pass
                else:
                    continue  # Continue to try next LLM provider
            else:
                # For non-rate limit errors, re-raise immediately
                logger.error(f"{current_llm_name} LLM failed with non-API error: {e}")
                raise

    # If we've exhausted all attempts, create a graceful fallback response
    logger.error("All LLM attempts exhausted. Creating graceful fallback response.")
    logger.error(f"Errors encountered: {errors}")
    
    if pydantic_schema is not None:
        # For Pydantic schemas, try to create a minimal valid object
        try:
            # Create a minimal valid object with default values
            minimal_obj = pydantic_schema()
            logger.warning(f"Creating minimal fallback object for {pydantic_schema.__name__}")
            return minimal_obj
        except Exception:
            # If we can't create a minimal object, raise the last error we had
            logger.error("Could not create minimal fallback object.")
            # Return the first error we encountered
            if errors:
                raise list(errors.values())[0]
            else:
                raise ValueError("Unable to process request with available LLMs.")
    else:
        # For non-Pydantic responses, return a simple error message
        logger.warning("Returning simple error message as fallback")
        return "I'm sorry, but I'm currently unable to process your request due to technical issues. Please try again later."

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
CACHE_DURATION_SECONDS = 300

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
def preprocess_files(state: State):
    '''
    Processes uploaded files. For images, it now only confirms the path.
    For documents, it creates the vector store. This is the full, robust version.
    '''
    logger.info("Starting file preprocessing...")
    uploaded_files = state.get("uploaded_files", [])
    if not uploaded_files:
        logger.info("No files to preprocess.")
        return state

    processed_files = []
    # Convert dictionaries from state back to Pydantic objects for safe access
    for file_obj_dict in uploaded_files:
        try:
            file_obj = FileObject.model_validate(file_obj_dict)
            
            # For images, we just ensure the path is valid and pass it along.
            # The image agent is responsible for reading and encoding.
            if file_obj.file_type == 'image':
                if not os.path.exists(file_obj.file_path):
                    logger.warning(f"Image file not found at path: {file_obj.file_path}. Skipping.")
                    continue
                # No other action is needed for images here.
            
            # For documents, we perform the full RAG preprocessing.
            elif file_obj.file_type == 'document':
                file_path = file_obj.file_path
                if not os.path.exists(file_path):
                    logger.warning(f"Document file not found at path: {file_path}. Skipping.")
                    continue

                logger.info(f"Processing document: {file_path}")
                ext = os.path.splitext(file_path)[1].lower()
                
                # Select the appropriate document loader based on file extension
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                # Create vector embeddings and save to a FAISS index
                vector_store = FAISS.from_documents(texts, get_hf_embeddings())
                index_path = f"storage/vector_store/{os.path.basename(file_path)}.faiss"
                vector_store.save_local(index_path)
                
                # Add the path to the vector store to our file object
                file_obj.vector_store_path = index_path
                logger.info(f"Document processed. Vector store saved to: {index_path}")

            processed_files.append(file_obj)

        except Exception as e:
            logger.error(f"Failed to process file {file_obj_dict.get('file_name', 'N/A')}: {e}")
            continue
            
    # Convert Pydantic objects back to dictionaries for state serialization
    return {"uploaded_files": [pf.model_dump(mode='json', exclude_none=True) for pf in processed_files]}


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
    # --- Create a formatted history of the conversation ---
    history = ""
    if messages := state.get('messages'):
        # Limit to the last few messages to keep the prompt concise
        for msg in messages[-20:]:  # Using the last 20 messages as context
            if hasattr(msg, 'type') and msg.type == "human":
                history += f"Human: {msg.content}\n"
            elif hasattr(msg, 'type') and msg.type == "ai":
                history += f"AI: {msg.content}\n"

    capability_texts, _ = get_all_capabilities()
    capabilities_list_str = ", ".join(f"'{c}'" for c in capability_texts)
    
    # Add file context if files are uploaded
    file_context = ""
    uploaded_files = state.get("uploaded_files", [])
    if uploaded_files:
        # Handle both dict and FileObject instances
        file_names = []
        for f in uploaded_files:
            if isinstance(f, dict):
                file_names.append(f.get('file_name', 'unknown'))
            else:
                file_names.append(f.file_name)
        
        file_context = f'''
        **UPLOADED FILES:**
        The user has uploaded the following file(s): {", ".join(file_names)}
        When the user refers to "this document", "the file", "the PDF", etc., they are referring to these uploaded files.
        You MUST include the file information in the task_description so the agent knows which file to process.
        For document-related tasks, the task_description should include: "Analyze the uploaded document: {file_names[0]}" or similar.
        '''

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

        Also extract any general user expectations (tone, urgency, budget, quality rating, etc.) from the prompt, if present. If not present, set them to null.

        The user's prompt will be provided like this:
        ---
        {state['original_prompt']}
        ---

        **EXAMPLES:**
        
        Example 1: If the user prompt is "Get the latest 10 news headlines for AAPL with publishers and article links.", your output should be:
        ```json
        {{
            "tasks": [
                {{
                    "task_name": "get company news headlines",
                    "task_description": "Get the latest 10 news headlines for AAPL, including the publisher and a link to the article for each headline."
                }}
            ],
            "user_expectations": {{}}
        }}
        ```
        
        Example 2: If the user uploads a file "resume.pdf" and says "Please summarise this document", your output should be:
        ```json
        {{
            "tasks": [
                {{
                    "task_name": "summarize_document",
                    "task_description": "Provide a comprehensive summary of the uploaded document 'resume.pdf', including key information, main points, and relevant details."
                }}
            ],
            "user_expectations": {{}}
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

    current_retry_count = state.get('parse_retry_count', 0)

    # Serialize parsed_tasks to ensure JSON compatibility
    serializable_tasks = []
    if parsed_tasks:
        for task in parsed_tasks:
            if hasattr(task, 'model_dump'):  # Pydantic v2
                serializable_tasks.append(task.model_dump(mode='json'))
            elif hasattr(task, 'dict'):  # Pydantic v1
                serializable_tasks.append(task.dict())
            elif isinstance(task, dict):
                serializable_tasks.append(task)
            else:
                serializable_tasks.append({"task_name": str(task), "task_description": ""})
    
    return {
        "parsed_tasks": serializable_tasks,
        "user_expectations": user_expectations or {},
        "parsing_error_feedback": None,
        "parse_retry_count": current_retry_count + 1
    }

async def agent_directory_search(state: State):
    parsed_tasks = state.get('parsed_tasks', [])
    # Extract task names for logging (handle both dict and object)
    task_names = [t.get('task_name') if isinstance(t, dict) else t.task_name for t in parsed_tasks]
    logger.info(f"Searching for agents for tasks: {task_names}")
    
    if not parsed_tasks:
        logger.warning("No valid tasks to process in agent_directory_search")
        return {"candidate_agents": {}}
    
    urls_to_fetch = []
    base_url = "http://127.0.0.1:8000/api/agents/search"
    user_expectations = state.get('user_expectations') or {}

    for task in parsed_tasks:
        # Handle both dict and Task object
        task_name = task.get('task_name') if isinstance(task, dict) else task.task_name
        task_description = task.get('task_description') if isinstance(task, dict) else task.task_description
        
        params: Dict[str, Any] = {'capabilities': task_name}
        if 'price' in user_expectations:
            params['max_price'] = user_expectations['price']
        if 'rating' in user_expectations:
            params['min_rating'] = user_expectations['rating']
        
        request = httpx.Request("GET", base_url, params=params)
        urls_to_fetch.append((task_name, str(request.url), task_description))
    
    logger.info(f"Dispatching {len(urls_to_fetch)} agent search requests.")
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch_agents_for_task(client, name, url) for name, url, desc in urls_to_fetch))
    
    candidate_agents_map = {res['task_name']: res['agents'] for res in results}
    logger.info("Agent search complete.")

    for task in parsed_tasks:
        # Handle both dict and Task object
        task_name = task.get('task_name') if isinstance(task, dict) else task.task_name
        task_description = task.get('task_description') if isinstance(task, dict) else task.task_description
        
        if not candidate_agents_map.get(task_name):
            error_feedback = (
                f"The previous attempt to parse the prompt resulted in the task description "
                f"'{task_description}', which was matched to the capability '{task_name}'. "
                f"However, no agents were found that could perform this task. Please generate a new, "
                f"more detailed and specific task description that better captures the user's intent."
            )
            logger.warning(f"Semantic failure for task '{task_name}'. Looping back to re-parse.")
            return {"candidate_agents": {}, "parsing_error_feedback": error_feedback}

    return {"candidate_agents": candidate_agents_map, "parsing_error_feedback": None}

class RankedAgents(BaseModel):
    ranked_agent_ids: List[str]

def rank_agents(state: State):
    parsed_tasks = state.get('parsed_tasks', [])
    # Extract task names for logging (handle both dict and object)
    task_names = [t.get('task_name') if isinstance(t, dict) else t.task_name for t in parsed_tasks]
    logger.info(f"Ranking agents for tasks: {task_names}")
    
    if not parsed_tasks:
        logger.warning("No tasks to rank in rank_agents")
        return {"task_agent_pairs": []}
    
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    final_selections = []
    for task in parsed_tasks:
        # Handle both dict and Task object
        task_name = task.get('task_name') if isinstance(task, dict) else task.task_name
        task_description = task.get('task_description') if isinstance(task, dict) else task.task_description
        
        # Rehydrate here. Convert dicts from state back into AgentCard objects.
        candidate_agent_dicts = state.get('candidate_agents', {}).get(task_name, [])
        candidate_agents = [AgentCard.model_validate(d) for d in candidate_agent_dicts]
        
        if not candidate_agents:
            continue

        if len(candidate_agents) == 1:
            primary_agent = candidate_agents[0]
            fallback_agents = []
        else:
            serializable_agents = [agent.model_dump(mode='json') for agent in candidate_agents]
            
            prompt = f'''
            You are an expert at selecting the best agent for a given task.
            The user's task is: "{task_description}"

            Here are the available agents that claim to have the capability '{task_name}':
            ---
            {json.dumps(serializable_agents, indent=2)}
            ---

            Please rank these agents in order of suitability for the task, from best to worst. The best agent should be the one whose description and capabilities most closely match the user's task.

            Your output should be a JSON object with a single key, "ranked_agent_ids", which is a list of agent IDs in the correct order.
            '''
            try:
                response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, RankedAgents)
                ranked_agent_ids = response.ranked_agent_ids
                
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
            task_description=task_description,
            primary=primary_agent,
            fallbacks=fallback_agents
        )
        final_selections.append(pair)
    
    # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
    serializable_pairs = [p.model_dump(mode='json') for p in final_selections]

    logger.info("Agent ranking complete.")
    logger.debug(f"Final agent selections: {[p for p in serializable_pairs]}")
    return {"task_agent_pairs": serializable_pairs}

def pause_for_user_approval(state: State):
    '''
    Pauses orchestration after agent selection to show parsed tasks and selected agents.
    Uses interrupt to actually pause the graph execution.
    '''
    logger.info("Pausing orchestration for user approval of tasks and agents.")
    
    # Serialize parsed_tasks to ensure they're JSON serializable
    parsed_tasks = state.get("parsed_tasks", [])
    serializable_tasks = []
    for task in parsed_tasks:
        if hasattr(task, 'model_dump'):  # Pydantic v2
            serializable_tasks.append(task.model_dump(mode='json'))
        elif hasattr(task, 'dict'):  # Pydantic v1
            serializable_tasks.append(task.dict())
        elif isinstance(task, dict):
            serializable_tasks.append(task)
        else:
            # Fallback for any other type
            serializable_tasks.append({"task_name": str(task), "task_description": ""})
    
    from langgraph.types import interrupt
    
    # This will actually interrupt the graph and wait for user approval
    # The value returned from interrupt() will be the user's response
    user_approval = interrupt({
        "type": "user_approval_required",
        "parsed_tasks": serializable_tasks,
        "task_agent_pairs": state.get("task_agent_pairs", []),
        "message": "Please review the parsed tasks and selected agents, then approve to continue."
    })
    
    logger.info(f"User approval received: {user_approval}")
    
    return {
        "parsed_tasks": serializable_tasks,
        "user_response": "approved"
    }

def plan_execution(state: State, config: RunnableConfig):
    '''
    Creates an initial execution plan or modifies an existing one if a replan is needed,
    and saves the result to a file.
    '''
    replan_reason = state.get("replan_reason")
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    output_state = {}

    if replan_reason:
        # --- REPLANNING MODE ---
        logger.info(f"Replanning initiated. Reason: {replan_reason}")
        
        all_capabilities, _ = get_all_capabilities()
        capabilities_str = ", ".join(all_capabilities)

        prompt = f'''
        You are an expert autonomous planner. The current execution plan is stalled. Your task is to surgically insert a new task into the plan to resolve the issue.

        **Reason for Replan:** "{replan_reason}"
        **Current Stalled Plan:** {json.dumps([task for batch in state.get('task_plan', []) for task in batch], indent=2)}
        **Original User Prompt:** "{state['original_prompt']}"
        **Full List of Available System Capabilities:** [{capabilities_str}]
        
        **Instructions:**
        1.  Analyze the `Reason for Replan` to understand what's missing (e.g., "missing coordinates for Hyderabad").
        2.  Identify the best capability from the `Available System Capabilities` to find this missing information. The **"perform web search and summarize"** capability is perfect for this.
        3. Create a new `PlannedTask`. The `task_description` should be a clear, self-contained instruction for another agent (e.g., "Find the latitude and longitude for Hyderabad, India using a web search"). You must select an agent and endpoint that provides the chosen capability.
        4.  **Insert this new task into the `Current Stalled Plan` *immediately before* the task that needs the information.**
        5. Return the entire modified plan. The output MUST be a valid JSON object conforming to the `ExecutionPlan` schema.
        '''
        try:
            response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, ExecutionPlan)
            # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
            if response and hasattr(response, 'plan'):
                serializable_plan = [[task.model_dump(mode='json') for task in batch] for batch in (response.plan or [])]
                output_state = {"task_plan": serializable_plan, "replan_reason": None} # Clear the reason after replanning
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
        
        Instructions:
        1. For each task, select the most appropriate endpoint from the primary agent's list.
        2. Create an ExecutionStep with the agent id, http_method, and endpoint.
        3. Do not generate a payload.
        4. Group tasks that can run in parallel into the same batch.
        5. Return a valid JSON object that conforms to the ExecutionPlan schema.

        Tasks to Plan: {json.dumps([p.model_dump(mode='json') for p in task_agent_pairs], indent=2)}
        '''
        try:
            response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, ExecutionPlan)
            # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
            if response and hasattr(response, 'plan'):
                serializable_plan = [[task.model_dump(mode='json') for task in batch] for batch in (response.plan or [])]
                output_state = {"task_plan": serializable_plan, "user_response": None}
            else:
                # If response is None or doesn't have plan attribute, create a simple plan
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
                                payload={}  # Empty payload for now, will be filled by run_agent
                            )
                        )
                        simple_plan.append(planned_task)
                
                if simple_plan:
                    serializable_plan = [[task.model_dump(mode='json') for task in simple_plan]]
                    output_state = {"task_plan": serializable_plan, "user_response": None}
                    logger.info("Created simplified plan as fallback")
                else:
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
                                payload={}  # Empty payload for now, will be filled by run_agent
                            )
                        )
                        simple_plan.append(planned_task)
                
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
    
    return output_state


def validate_plan_for_execution(state: State):
    '''
    Performs an advanced pre-flight check on the next task, now with full file
    context awareness to prevent premature pauses.
    '''
    logger.info("Performing dynamic validation of the execution plan...")
    
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
    - Previously Completed Tasks: {json.dumps(state.get('completed_tasks', []), indent=2, default=str)}
    - Task to Validate: "{task_to_validate.task_description}"
    - Required Parameters for this Task: {required_params}
    {file_context}
    - All Available System Capabilities: [{capabilities_str}]

    **Your Decision Process:**
    1.  **Check Context:** Can all `Required Parameters` (e.g., 'image_path', 'vector_store_path', 'query') be filled using the `Original User Prompt`, `Conversation History`, `Previously Completed Tasks`, or the `Available File Context`? The file paths provided are the values you should use.
    
    2.  **Special Case - Document Summarization:** If the task is about summarizing, analyzing, or describing a document, and a 'query' parameter is required:
        - If the user's prompt contains words like "summarize", "summary", "describe", "analyze", "what is in", or similar, treat this as a valid query.
        - The query can be inferred as "Provide a comprehensive summary of this document" or "Describe the contents of this document".
        - In this case, respond with `status: "ready"` because the intent is clear.
    
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

async def run_agent(planned_task: PlannedTask, agent_details: AgentCard, state: State, last_error: Optional[str] = None):
    '''
    Builds the payload and runs a single agent, now using file paths instead of content.
    This is the full, robust version with semantic retries and rate limit handling.
    '''
    logger.info(f"Running agent '{agent_details.name}' for task: '{planned_task.task_name}'")
    
    endpoint_url = str(planned_task.primary.endpoint)
    http_method = planned_task.primary.http_method.upper()
    
    selected_endpoint = next((ep for ep in agent_details.endpoints if str(ep.endpoint) == endpoint_url), None)

    if not selected_endpoint:
        error_msg = f"Critical Error: Could not find endpoint details for '{endpoint_url}' on agent '{agent_details.name}'."
        logger.error(error_msg)
        return {"task_name": planned_task.task_name, "result": error_msg}

    # Initialize both primary and fallback LLMs for payload generation
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    failed_attempts = []
    
    # --- Prepare detailed file context for the payload builder ---
    file_context = ""
    uploaded_files = state.get("uploaded_files", [])
    if uploaded_files:
        file_context = f'''
        **Available File Context:**
        The user has uploaded files. You MUST use the file information below to populate the required payload parameters (like 'image_path' or 'vector_store_path').
        ```json
        {json.dumps(uploaded_files, indent=2)}
        ```
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

    # This loop handles semantic retries (e.g., valid but empty/useless responses)
    for attempt in range(3):
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

        payload_prompt = f'''
        You are an expert at creating API requests. Your task is to generate a valid JSON payload for the following endpoint, based on all the provided context.

        Endpoint Description: "{selected_endpoint.description}"
        Endpoint Parameters: {[p.model_dump_json() for p in selected_endpoint.parameters]}
        High-Level Task: "{planned_task.task_description}"
        Conversation History:
        {history}
        Historical Context (previous task results): {json.dumps(state.get('completed_tasks', []), indent=2, default=str)}
        {file_context}
        {http_error_context}
        {failed_attempts_context}
        Your response MUST be only the JSON payload object, with no extra text or markdown.
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

        headers = {}
        if api_key := os.getenv(f"{agent_details.id.upper().replace('-', '_')}_API_KEY"):
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Calling agent '{agent_details.name}' at '{endpoint_url}' (Attempt {attempt + 1})")
                if http_method == 'GET':
                    response = await client.get(endpoint_url, params=payload, headers=headers, timeout=30.0)
                else: # POST
                    response = await client.post(endpoint_url, json=payload, headers=headers, timeout=30.0)

                response.raise_for_status()
                result = response.json()

                # **INTELLIGENT VALIDATION** for semantic failure
                is_result_empty = not result or (isinstance(result, list) and not result) or (isinstance(result, dict) and not any(result.values()))
                if is_result_empty:
                    logger.warning(f"Agent returned a successful but empty response. Retrying...")
                    failed_attempts.append({"payload": payload, "result": str(result)})
                    continue  # Continue to the next attempt in the loop
                
                logger.info(f"Agent call successful for task '{planned_task.task_name}'.")
                return {"task_name": planned_task.task_name, "result": result}
            
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
                error_msg = f"Agent call failed with a network error: {e}"
                return {"task_name": planned_task.task_name, "result": error_msg, "raw_response": str(e), "status_code": 500}
    
    # This block is reached only if all semantic retries in the loop fail
    final_error_msg = f"Agent returned empty or unsatisfactory results for task '{planned_task.task_name}' after {len(failed_attempts)} attempts."
    logger.error(final_error_msg)
    return {"task_name": planned_task.task_name, "result": final_error_msg, "status_code": 500}

async def execute_batch(state: State, config: RunnableConfig):
    '''Executes a single batch of tasks from the plan.'''
    # Rehydrate the plan
    task_plan_dicts = state.get('task_plan', [])
    if not task_plan_dicts:
        logger.info("No task plan to execute.")
        return {}
    task_plan = [[PlannedTask.model_validate(d) for d in batch] for batch in task_plan_dicts]

    current_batch_plan = task_plan[0]
    remaining_plan_objects = task_plan[1:]
    logger.info(f"Executing batch of {len(current_batch_plan)} tasks.")
    
    # Rehydrate the pairs
    task_agent_pair_dicts = state.get('task_agent_pairs', [])
    task_agent_pairs = [TaskAgentPair.model_validate(d) for d in task_agent_pair_dicts]
    task_agent_pairs_map = {pair.task_name: pair for pair in task_agent_pairs}
    
    async def try_task_with_fallbacks(planned_task: PlannedTask):
        original_task_pair = task_agent_pairs_map.get(planned_task.task_name)
        if not original_task_pair:
            error_msg = f"Could not find original task pair for '{planned_task.task_name}' to get fallbacks."
            logger.error(error_msg)
            return {"task_name": planned_task.task_name, "result": error_msg}

        agents_to_try = [original_task_pair.primary] + original_task_pair.fallbacks
        final_error_result = None
        
        for agent_to_try in agents_to_try:
            max_retries = 3 if agent_to_try.id == original_task_pair.primary.id else 1
            last_error = None
            for i in range(max_retries):
                logger.info(f"Attempting task '{planned_task.task_name}' with agent '{agent_to_try.name}' (Attempt {i+1})...")
                
                # The state object is passed directly to run_agent
                task_result = await run_agent(planned_task, agent_to_try, state, last_error=last_error)
                
                result_data = task_result.get('result', {})
                is_error = isinstance(result_data, str) and "Error:" in result_data
                
                if not is_error:
                    logger.info(f"Task '{planned_task.task_name}' succeeded with agent '{agent_to_try.name}'.")
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
        
        logger.error(f"All agents failed for task '{planned_task.task_name}'. Returning final error.")
        return final_error_result

    batch_results = await asyncio.gather(*(try_task_with_fallbacks(planned_task) for planned_task in current_batch_plan))
    
    completed_tasks_with_desc = []
    for res in batch_results:
        task_name = res['task_name']
        completed_tasks_with_desc.append(CompletedTask(
            task_name=task_name,
            result=res.get('result', {})
        ))

    completed_tasks = state.get('completed_tasks', []) + completed_tasks_with_desc
    logger.info("Batch execution complete.")
    
    # FIX: Use mode='json' to convert HttpUrl and other special types to strings.
    remaining_plan_dicts = [[task.model_dump(mode='json') for task in batch] for batch in remaining_plan_objects]

    output_state = {
        "task_plan": remaining_plan_dicts,
        "completed_tasks": completed_tasks,
        "latest_completed_tasks": completed_tasks_with_desc
    }

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
    latest_tasks = state.get("latest_completed_tasks", [])
    if not latest_tasks:
        # No new tasks to evaluate
        return {"pending_user_input": False, "question_for_user": None}

    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    task_to_evaluate = latest_tasks[-1] # Evaluate the most recent task

    # If the agent itself reported an error, we don't need to evaluate it further
    if isinstance(task_to_evaluate.get('result'), str) and "Error:" in task_to_evaluate.get('result', ''):
        return {"pending_user_input": False, "question_for_user": None}

    prompt = f'''
    You are a meticulous Quality Assurance AI. Your job is to determine if an agent's output is a successful and logical fulfillment of its assigned task.

    **Original User Prompt:** "{state['original_prompt']}"
    **Task Description:** "{task_to_evaluate.get('task_description', 'N/A')}"
    **Agent's Result:**
    ```json
    {json.dumps(task_to_evaluate['result'], indent=2)}
    ```

    **Instructions:**
    1.  **Check for Logical Consistency:** Does the `Agent's Result` make sense in the context of the `Task Description`? (e.g., If the task was to find a "technology company," is the result actually a tech company, not a newspaper?).
    2.  **Check for Completeness:** Is the result empty, or does it contain placeholders like "N/A" or 0.0 when a real value was expected?
    3.  **Check for Unverified Assumptions:** Does the result rely on information not present in the original prompt or task description?

    **Decision:**
    - If the result is logically sound and complete, respond with `{{"status": "complete"}}`.
    - If the result is logically flawed, incomplete, or based on a wrong assumption, respond with `{{"status": "user_input_required", "question": "Formulate a clear, direct question to the user to correct the course of the plan."}}`. For example, "The news search returned an article about the lumber industry, not a tech company. Could you specify a tech company you're interested in?"
    '''
    try:
        evaluation = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, AgentResponseEvaluation)
        if evaluation.status == "user_input_required":
            logger.warning(f"Result for task '{task_to_evaluate['task_name']}' is unsatisfactory. Pausing for user input.")
            # We add the failed task's result to the parsing feedback to prevent loops
            error_feedback = f"The previous attempt for a similar task resulted in an incorrect output: {task_to_evaluate['result']}. Please generate a more precise task to avoid this error."
            return {
                "pending_user_input": True,
                "question_for_user": evaluation.question,
                "parsing_error_feedback": error_feedback
            }
    except Exception as e:
        logger.error(f"Failed to evaluate agent response for task '{task_to_evaluate['task_name']}': {e}")
    
    return {"pending_user_input": False, "question_for_user": None}

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

    ai_message = AIMessage(content=question)
    return {
        "pending_user_input": True,
        "question_for_user": question,
        "final_response": None, # Clear any previous final response
        "messages": [ai_message]
    }


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
    - You are generating content for a canvas display area in a web application
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
        
        ai_message = AIMessage(content=final_response)
        return {"final_response": final_response, "messages": [ai_message]}
    else:
        # This is a complex request, synthesize results from completed tasks
        completed_tasks = state.get('completed_tasks', [])
        if not completed_tasks:
            logger.warning("Complex request indicated but no completed tasks found. Generating default response.")
            final_response = "I've processed your request, but I don't have specific results to share."
        else:
            # If canvas is needed, generate a more concise text response that references the canvas
            if needs_canvas and canvas_type:
                prompt = f'''
                You are an expert project manager's assistant. Your job is to synthesize the results from a team of AI agents into a single, clean, and coherent final report for the user.
                The user's original request was:
                "{state['original_prompt']}"
                
                The following tasks were completed, with these results:
                ---
                {json.dumps([serialize_complex_object(task) for task in completed_tasks], indent=2)}
                ---
                
                A {canvas_type} visualization has been prepared to display this information. 
                Please generate a brief, human-readable summary that references the visualization 
                and highlights the key findings without reproducing the raw data or code.
                '''
            else:
                prompt = f'''
                You are an expert project manager's assistant. Your job is to synthesize the results from a team of AI agents into a single, clean, and coherent final report for the user.
                The user's original request was:
                "{state['original_prompt']}"
                
                The following tasks were completed, with these results:
                ---
                {json.dumps([serialize_complex_object(task) for task in completed_tasks], indent=2)}
                ---
                Please generate a final, human-readable response that directly answers the user's original request based on the collected results.
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
        
        ai_message = AIMessage(content=final_response)
        return {"final_response": final_response, "messages": [ai_message]}

def generate_final_response(state: State):
    """
    Generates the final response and determines if canvas is needed.
    This replaces the old aggregate_responses node.
    """
    logger.info("=== GENERATE_FINAL_RESPONSE: Starting ===")
    
    # First generate the text answer
    text_result = generate_text_answer(state)
    
    # Check if the text result contains HTML content that should be in canvas
    final_response = text_result.get('final_response', '')
    
    # Enhanced canvas detection - check for HTML content in the response
    contains_html = False
    html_indicators = [
        '<!DOCTYPE html>', '<html', '<button', '<script>', 'onclick=', 'onClick=',
        '<div', '<span>', '<p>', '<h1>', '<h2>', '<h3>', '<style>', '<head>'
    ]
    
    for indicator in html_indicators:
        if indicator in final_response:
            contains_html = True
            logger.info(f"HTML content detected (indicator: {indicator})")
            break
    
    # Now analyze if canvas is needed using LLM
    # Initialize both primary and fallback LLMs
    primary_llm = ChatCerebras(model="gpt-oss-120b")
    fallback_llm = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    
    prompt = f'''
    Analyze the user's request and the generated text response to determine if a canvas visualization would enhance the user experience.
    
    **YOUR PURPOSE AND ENVIRONMENT:**
    You are a decision-making AI that determines whether to generate a canvas visualization for a web application.
    The canvas is a separate display area that can show interactive content, visualizations, or rich media.
    The chat area should only contain text responses, while the canvas area shows interactive content.
    
    **CONTEXT:**
    User's Original Request: "{state['original_prompt']}"
    Generated Text Response: "{final_response}"
    
    **CANVAS DECISION CRITERIA:**
    
    **DEFINITELY USE CANVAS for:**
    - Interactive elements (counters, buttons, games, demos)
    - Visualizations (charts, graphs, plots, data visualization)
    - Complex layouts or formatting that text cannot represent well
    - Web pages, HTML content, or rich media
    - Documents that need markdown rendering
    - When user explicitly asks for "canvas", "visual", "interactive", "demo", "show me"
    
    **CONSIDER CANVAS for:**
    - Numerical data that could benefit from visualization
    - Complex information that would be clearer visually
    - When the text response mentions creating visual content
    
    **USE TEXT ONLY for:**
    - Simple questions and factual answers
    - Explanations and descriptions
    - When no visual enhancement is needed
    
    **DECISION PROCESS:**
    1. Look for explicit canvas requests in the original prompt
    2. Analyze if the content would benefit from visual representation
    3. Consider if interactive elements would improve user experience
    4. Default to text-only if uncertain
    
    **IMPORTANT OUTPUT RULES:**
    - If the response contains HTML elements like buttons, scripts, or interactive content, ALWAYS use "html" canvas type, never "markdown".
    - Your response will be used to generate a canvas visualization, so be precise in your decision.
    - The canvas content will be displayed in an iframe with limited capabilities, so external libraries must be imported via CDN if needed.
    
    Respond with a JSON object:
    {{
        "use_canvas": true/false,
        "reasoning": "Brief explanation of why canvas is or isn't needed",
        "canvas_type": "html" or "markdown" (only if use_canvas is true),
        "canvas_prompt": "Specific instructions for what canvas content to generate (only if use_canvas is true)"
    }}
    '''
    
    try:
        # Create a simple model for canvas decision
        class CanvasDecision(BaseModel):
            use_canvas: bool
            reasoning: str
            canvas_type: Optional[Literal["html", "markdown"]] = None
            canvas_prompt: Optional[str] = None
        
        response = invoke_llm_with_fallback(primary_llm, fallback_llm, prompt, CanvasDecision)
        
        # Force canvas usage if HTML content is detected
        if contains_html and not response.use_canvas:
            logger.info("CANVAS DETECTION: Forcing canvas usage due to HTML content detection")
            response.use_canvas = True
            response.canvas_type = "html"
            response.reasoning = "HTML content detected, forcing canvas usage"
            response.canvas_prompt = state['original_prompt']
        
        if response.use_canvas:
            logger.info(f"Canvas needed: {response.canvas_type} - {response.reasoning}")
            
            # When canvas is needed, we should generate a concise text response for the chat
            # that references the canvas content without including the actual HTML
            concise_text_prompt = f'''
            The user's original request was:
            "{state['original_prompt']}"
            
            A {response.canvas_type} visualization has been created to display the results.
            Please generate a brief, human-readable message that tells the user to check 
            the {response.canvas_type} visualization for the results, without including any code.
            '''
            
            concise_text_response = invoke_llm_with_fallback(primary_llm, fallback_llm, concise_text_prompt, None).__str__()
            
            result = {
                "final_response": concise_text_response,
                "messages": text_result.get('messages', []),
                "needs_canvas": True,
                "canvas_type": response.canvas_type,
                "canvas_prompt": response.canvas_prompt or state['original_prompt'],
                "has_canvas": False,  # Explicitly set to False initially
                "canvas_content": None  # Explicitly set to None initially
            }
            logger.info(f"Returning with needs_canvas=True, canvas_type={result.get('canvas_type')}")
            return result
        else:
            logger.info(f"Text-only response sufficient: {response.reasoning}")
            result = {
                "final_response": final_response,
                "messages": text_result.get('messages', []),
                "needs_canvas": False,
                "canvas_type": None,
                "canvas_prompt": None,
                "has_canvas": False,
                "canvas_content": None
            }
            logger.info("Returning with needs_canvas=False")
            return result
            
    except Exception as e:
        logger.error(f"Canvas decision failed: {e}. Defaulting to text-only.")
        result = {
            "final_response": final_response,
            "messages": text_result.get('messages', []),
            "needs_canvas": False,
            "canvas_type": None,
            "canvas_prompt": None,
            "has_canvas": False,
            "canvas_content": None
        }
        logger.info("Returning (error case)")
        return result



def load_conversation_history(state: State, config: RunnableConfig):
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return {}

    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")

    if not os.path.exists(history_path):
        return {}

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
                if msg_type == "user":
                    msg = HumanMessage(content=content)
                elif msg_type == "assistant":
                    msg = AIMessage(content=content)
                else:
                    msg = SystemMessage(content=content)

                # Add additional attributes
                if msg_id:
                    msg.id = msg_id
                if metadata:
                    msg.additional_kwargs = metadata
                if timestamp:
                    msg.timestamp = timestamp

                valid_messages.append(msg)

            except Exception as e:
                logger.warning(f"Failed to create message object: {e}")
                continue

        logger.info(f"Successfully loaded {len(valid_messages)} messages for conversation {thread_id}")
        
        return {
            "messages": valid_messages,
            "thread_id": data.get("thread_id"),
            "final_response": data.get("final_response")
        }

    except Exception as e:
        logger.error(f"Failed to load conversation history for {thread_id}: {e}")
        return {"messages": []}

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
            frontend_msg = {
                'id': msg.get('id', str(time.time())),
                'type': 'assistant' if msg_type == 'ai' else 'user' if msg_type == 'human' else 'system',
                'content': content,
                'timestamp': msg_data.get('timestamp', time.time())
            }
            serializable_messages.append(frontend_msg)
            logger.debug(f"Converted message: type={msg_type}, content_length={len(content)}")
            
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

    # Use the serialize_complex_object helper for other potentially complex fields
    # This ensures nested Pydantic models, HttpUrl, etc., are converted correctly.
    logger.info(f"Serialized {len(serializable_messages)} messages for thread {thread_id}")
    logger.info(f"Final response length: {len(state.get('final_response', '')) if state.get('final_response') else 0}")
    
    # Determine status based on orchestration state
    if state.get("waiting_for_continue"):
        status = "orchestration_paused"
    elif state.get("pending_user_input"):
        status = "pending_user_input"
    else:
        status = "completed"
    
    return {
        "thread_id": thread_id,
        "status": status,
        "messages": serializable_messages,
        "task_agent_pairs": serialize_complex_object(state.get("task_agent_pairs", [])),
        "parsed_tasks": serialize_complex_object(state.get("parsed_tasks", [])),
        "final_response": state.get("final_response"),
        "pending_user_input": state.get("pending_user_input", False),
        "question_for_user": state.get("question_for_user"),
        # Metadata for the sidebar
        "metadata": {
            "original_prompt": state.get("original_prompt"),
            "completed_tasks": serialize_complex_object(state.get("completed_tasks", [])),
            "parsed_tasks": serialize_complex_object(state.get("parsed_tasks", [])),
            "currentStage": "paused" if state.get("waiting_for_continue") else "completed",
            "stageMessage": state.get("pause_reason") if state.get("waiting_for_continue") else "Orchestration completed successfully!",
            "progress": 50 if state.get("waiting_for_continue") else 100,
            "orchestrationPaused": state.get("orchestration_paused", False),
            "pauseReason": state.get("pause_reason")
        },
        # Attachments for the sidebar
        "uploaded_files": serialize_complex_object(state.get("uploaded_files", [])),
        # Plan for the sidebar
        "plan": serialize_complex_object(state.get("task_plan", [])),
        # Canvas fields for the sidebar
        "has_canvas": state.get("has_canvas", False),
        "canvas_content": state.get("canvas_content"),
        "canvas_type": state.get("canvas_type"),
        "needs_canvas": state.get("needs_canvas", False),
        "timestamp": time.time(),
    }

def save_conversation_history(state: State, config: RunnableConfig):
    """
    Saves the full, serializable state of the conversation to a JSON file.
    This is the single source of truth for conversation history.
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
                       'uploaded_files', 'task_plan', 'canvas_content', 'canvas_type', 'has_canvas']:
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

        # Write to file in a consistent JSON format
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved conversation history for thread {thread_id}.")

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

def route_after_validation(state: State):
    '''This router acts as the gate after the plan is validated.'''
    if state.get("replan_reason"):
        logger.info("Routing back to plan_execution for a replan.")
        return "plan_execution"
    if state.get("pending_user_input"):
        logger.info("Routing to ask_user due to failed plan validation.")
        return "ask_user"
    else:
        logger.info("Plan is valid. Routing to execute_batch.")
        return "execute_batch"

def analyze_request(state: State):
    """Sophisticated analysis of user request to determine processing approach."""
    logger.info("Performing sophisticated analysis of user request...")
    
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
    Analyze the user's request and determine if it requires complex orchestration or can be handled with a simple response.
    
    Consider the following context:
    {files_context}
    {tasks_context}
    Conversation history:
    {history_context}
    
    User's current request: "{state['original_prompt']}"
    
    Evaluate based on these criteria:
    1. Is this a simple greeting, thanks, or general knowledge question?
    2. Does this require accessing uploaded files or documents?
    3. Does this require multiple agents or complex operations?
    4. Is this a follow-up that builds on previous results?
    5. Does this require external data or complex processing?
    
    Respond with a JSON object containing:
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
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}. Defaulting to complex processing.")
        return {
            "needs_complex_processing": True, 
            "analysis_reasoning": f"Analysis failed: {e}",
            "final_response": ""  # Explicitly clear for error case
        }


def route_after_analysis(state: State):
    """Routes the workflow based on the analysis result."""
    if state.get("needs_complex_processing"):
        logger.info("Request needs complex processing. Routing to preprocess_files or parse_prompt.")
        if state.get("uploaded_files"):
            logger.info("Request has files. Routing to preprocess_files.")
            return "preprocess_files"
        else:
            logger.info("Request has no files. Routing to parse_prompt.")
            return "parse_prompt"
    else:
        logger.info("Request is simple. Routing to final response generation.")
        return "generate_final_response"  # For simple responses, we'll use the generate_final_response node


def route_after_parse(state: State):
    '''If parsing results in no tasks, ask user for clarification. Otherwise, proceed.'''
    if not state.get('parsed_tasks'):
        logger.warning("No tasks were parsed from prompt. Routing to ask_user for clarification.")
        return "ask_user"
    return "agent_directory_search"

def should_continue_or_finish(state: State):
    '''This router runs after execution and evaluation to decide the next step.'''
    if state.get("pending_user_input"):
        # If the evaluation failed and we need user input, go to ask_user
        return "ask_user"
    if not state.get('task_plan'):
        # If the plan is empty and evaluation passed, we are done
        logger.info("Execution plan is complete. Routing to generate_final_response.")
        return "generate_final_response"
    else:
        # If there are more tasks and evaluation passed, continue to next batch
        logger.info("Plan has more batches. Routing back to validation for the next batch.")
        return "validate_plan_for_execution"

# --- Build the State Graph ---
builder = StateGraph(State)

builder.add_node("load_history", load_conversation_history)
builder.add_node("save_history", save_conversation_history)
builder.add_node("analyze_request", analyze_request)
builder.add_node("parse_prompt", parse_prompt)
builder.add_node("preprocess_files", preprocess_files)
builder.add_node("agent_directory_search", agent_directory_search)
builder.add_node("rank_agents", rank_agents)
builder.add_node("pause_for_user_approval", pause_for_user_approval)
builder.add_node("plan_execution", plan_execution)
builder.add_node("validate_plan_for_execution", validate_plan_for_execution)
builder.add_node("execute_batch", execute_batch)
builder.add_node("evaluate_agent_response", evaluate_agent_response)
builder.add_node("ask_user", ask_user)
builder.add_node("generate_final_response", generate_final_response)
builder.add_node("render_canvas_output", render_canvas_output)

builder.add_edge(START, "load_history")
builder.add_edge("load_history", "analyze_request")

builder.add_conditional_edges("analyze_request", route_after_analysis, {
    "preprocess_files": "preprocess_files",
    "parse_prompt": "parse_prompt",
    "generate_final_response": "generate_final_response"
})

builder.add_edge("preprocess_files", "parse_prompt")

builder.add_conditional_edges("parse_prompt", route_after_parse, {
    "ask_user": "ask_user",
    "agent_directory_search": "agent_directory_search"
})

builder.add_edge("agent_directory_search", "rank_agents")
# After rank_agents, always pause for user approval
builder.add_edge("rank_agents", "pause_for_user_approval")
# After pause approval, go to plan execution
builder.add_edge("pause_for_user_approval", "plan_execution")
builder.add_edge("plan_execution", "validate_plan_for_execution")
builder.add_edge("execute_batch", "evaluate_agent_response") 
builder.add_edge("ask_user", "save_history")
builder.add_edge("generate_final_response", "render_canvas_output")
builder.add_edge("render_canvas_output", "save_history")
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
    "ask_user": "ask_user"
})

# Compile the graph
graph = builder.compile()

def create_graph_with_checkpointer(checkpointer):
    """Attaches a checkpointer to the graph for persistent memory."""
    return builder.compile(checkpointer=checkpointer)

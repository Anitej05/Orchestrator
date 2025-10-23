# Project_Agent_Directory/main.py
import uuid
import logging
import json
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pydantic.networks import HttpUrl
from langchain_core.messages import HumanMessage
import shutil
from fastapi import UploadFile, File
from aiofiles import open as aio_open
from typing import List
from pydantic import BaseModel
from typing import Literal

# # --- Add parent directory to path ---
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import subprocess
import sys
import platform
import socket
import re

# --- Third-party Imports ---
from fastapi import FastAPI, HTTPException, Depends, status, Query, Response, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, cast, String, select
from sentence_transformers import SentenceTransformer

# --- Local Application Imports ---
CONVERSATION_HISTORY_DIR = "conversation_history"
from database import SessionLocal
from models import Agent, StatusEnum, AgentCapability, AgentEndpoint, EndpointParameter
from schemas import AgentCard, ProcessRequest, ProcessResponse, PlanResponse, FileObject
from orchestrator.graph import ForceJsonSerializer, graph, create_graph_with_checkpointer, messages_from_dict, messages_to_dict, serialize_complex_object
from orchestrator.state import State
from langgraph.checkpoint.memory import MemorySaver

# --- App Initialization and Configuration ---
app = FastAPI(
    title="Unified Agent Service API",
    version="1.0",
    description="An API for both finding/managing agents and orchestrating tasks."
)

# Configure logging
# Backend logger - only for backend/main.py logs
logger = logging.getLogger("uvicorn.error")

# Configure root logger for backend only (not orchestrator)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up orchestrator logger to write to temp file only (not console)
# This keeps backend console clean and stores orchestrator logs separately
orchestrator_logger = logging.getLogger("AgentOrchestrator")
orchestrator_logger.setLevel(logging.INFO)
orchestrator_logger.propagate = False  # Don't propagate to root logger

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# File handler for orchestrator logs - overwrites on each run (last conversation only)
orchestrator_log_file = "logs/orchestrator_temp.log"
file_handler = logging.FileHandler(orchestrator_log_file, mode='w')  # 'w' mode overwrites
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
orchestrator_logger.addHandler(file_handler)

logger.info(f"Orchestrator logs will be saved to: {orchestrator_log_file}")

def clear_orchestrator_log():
    """Clear the orchestrator log file for a new conversation."""
    try:
        with open(orchestrator_log_file, 'w') as f:
            f.write('')  # Clear the file
        orchestrator_logger.info("=== New Conversation Started ===")
    except Exception as e:
        logger.error(f"Failed to clear orchestrator log: {e}")

# Initialize memory for persistent conversations
checkpointer = MemorySaver()

# Create the graph with the checkpointer
graph = create_graph_with_checkpointer(checkpointer)

# Simple in-memory conversation store as backup
conversation_store: Dict[str, Dict[str, Any]] = {}
from threading import Lock
store_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --- Sentence Transformer Model Loading ---
model = SentenceTransformer('all-mpnet-base-v2')

# --- Interactive Conversation Models ---
class UserResponse(BaseModel):
    """Model for user responses to orchestrator questions"""
    response: str
    thread_id: str
    files: Optional[List[FileObject]] = None

class ConversationStatus(BaseModel):
    """Model for conversation status responses"""
    thread_id: str
    status: str  # "completed", "pending_user_input", "error"
    question_for_user: Optional[str] = None
    final_response: Optional[str] = None
    task_agent_pairs: Optional[List[Dict]] = None
    error_message: Optional[str] = None

# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Agent Server Startup ---
def is_port_in_use(port: int) -> bool:
    """Checks if a local port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_port(port: int, agent_file: str, timeout: int = 15):
    """Waits for a network port to become active."""
    start_time = time.time()
    logger.info(f"Waiting for agent '{agent_file}' to start on port {port}...")
    while time.time() - start_time < timeout:
        if is_port_in_use(port):
            logger.info(f"Agent '{agent_file}' is now running on port {port}.")
            return True
        time.sleep(0.5)
    logger.error(f"Agent '{agent_file}' did not start on port {port} within {timeout} seconds.")
    return False

def start_agent_servers():
    """
    Finds and starts agent servers, with enhanced logging and better error handling
    to track which agents start successfully and which fail.
    """
    # Use absolute path based on the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))  # This gets the backend directory
    agents_dir = os.path.join(project_root, "agents")
    
    if not os.path.isdir(agents_dir):
        logger.warning(f"'{agents_dir}' directory not found. Skipping agent server startup.")
        return

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    logger.info(f"Agent logs will be stored in the '{logs_dir}' directory.")

    agent_files = [f for f in os.listdir(agents_dir) if f.endswith("_agent.py")]
    logger.info(f"Found {len(agent_files)} agent(s) to check: {agent_files}")

    started_agents = []
    failed_agents = []

    for agent_file in agent_files:
        agent_path = os.path.join(agents_dir, agent_file)
        port = None
        process = None
        
        try:
            with open(agent_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for port definition in the agent file
                match = re.search(r'port\s*=\s*int\(os\.getenv\([^,]+,\s*(\d+)\)', content)
                if not match:
                    # Fallback to direct assignment like: port = 8010
                    match = re.search(r"port\s*=\s*(\d+)", content)
                if match:
                    port = int(match.group(1))

            if port is None:
                logger.error(f"Could not find port in {agent_file}. Skipping.")
                failed_agents.append({
                    'agent': agent_file,
                    'reason': 'Port not found in agent file'
                })
                continue

            if is_port_in_use(port):
                logger.info(f"Agent '{agent_file}' is already running on port {port}.")
                started_agents.append({
                    'agent': agent_file,
                    'port': port,
                    'status': 'already_running'
                })
                continue

            logger.info(f"Attempting to start '{agent_file}' on port {port}...")
            
            log_path = os.path.join(logs_dir, f"{agent_file}.log")

            # Create the subprocess differently based on the OS
            if platform.system() == "Windows":
                # For Windows, use subprocess.Popen with proper parameters to run in background
                try:
                    with open(log_path, 'w') as log_file:
                        process = subprocess.Popen(
                            [sys.executable, agent_path],
                            stdout=log_file,
                            stderr=log_file,
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.environ.get('DEBUG_AGENT_STARTUP') else subprocess.CREATE_NEW_PROCESS_GROUP
                        )
                except Exception as e:
                    logger.error(f"Failed to start {agent_file} using Popen: {e}")
                    # Fallback to the start command
                    command = f'start /B "Agent: {agent_file}" /D "{os.getcwd()}" {sys.executable} {agent_path} > "{log_path}" 2>&1'
                    subprocess.run(command, shell=True, check=True)
            else:
                # For Unix-like systems
                with open(log_path, 'w') as log_file:
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file
                    )

            # Wait for the port to be in use with a timeout
            # With lazy imports and reload=False, agents should start quickly
            agent_timeout = 15  # Reasonable timeout for fast startup
            
            if wait_for_port(port, agent_file, timeout=agent_timeout):
                logger.info(f"Successfully started agent '{agent_file}' on port {port}")
                started_agents.append({
                    'agent': agent_file,
                    'port': port,
                    'status': 'started',
                    'process': process
                })
            else:
                logger.error(f"Timed out waiting for agent '{agent_file}' to start on port {port}")
                failed_agents.append({
                    'agent': agent_file,
                    'reason': f'Timed out waiting for port {port} to become available',
                    'port': port
                })
                if process:
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start agent {agent_file} using shell command: {e}")
            failed_agents.append({
                'agent': agent_file,
                'reason': f'Shell command failed: {str(e)}'
            })
        except Exception as e:
            logger.error(f"Unexpected error while starting agent {agent_file}: {e}")
            logger.exception("Full traceback:")  # Log the full traceback
            failed_agents.append({
                'agent': agent_file,
                'reason': f'Unexpected error: {str(e)}'
            })

    # Log summary of agent startup results
    logger.info(f"Agent startup completed. Started: {len(started_agents)}, Failed: {len(failed_agents)}")
    
    if started_agents:
        logger.info("Successfully started agents:")
        for agent_info in started_agents:
            logger.info(f"  - {agent_info['agent']} on port {agent_info['port']} ({agent_info['status']})")
    
    if failed_agents:
        logger.error("Failed to start agents:")
        for agent_info in failed_agents:
            logger.error(f"  - {agent_info['agent']}: {agent_info['reason']}")
        
        # Provide detailed instructions for manual startup
        logger.info("To start agents manually, run these commands in separate terminals:")
        for agent_info in failed_agents:
            agent_file = agent_info['agent']
            agent_path = os.path.join(agents_dir, agent_file)
            logger.info(f"  - python {agent_path}")
    
    logger.info("Agent startup check completed.")

os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/documents", exist_ok=True)

@app.post("/api/upload", response_model=List[FileObject])
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Handles file uploads, saves them to the appropriate storage directory,
    and returns their metadata.
    """
    file_objects = []
    for file in files:
        # **FIX 1: Handle potential None for filename**
        if not file.filename:
            continue  # Or raise an HTTPException for files without names

        # **FIX 2: Handle potential None for content_type**
        file_type = 'image' if file.content_type and file.content_type.startswith('image/') else 'document'
        save_dir = f"storage/{file_type}s"  # Path relative to project root
        file_path = os.path.join(save_dir, file.filename)

        # Save the file asynchronously
        try:
            async with aio_open(file_path, 'wb') as out_file:
                while content := await file.read(1024):  # Read in chunks
                    await out_file.write(content)
        except Exception as e:
            # Handle potential file-saving errors
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

        file_objects.append(FileObject(
            file_name=file.filename,
            file_path=file_path,
            file_type=file_type
        ))
    return file_objects

@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    """
    Serves uploaded files (images, documents) from the storage directory.
    """
    # Decode the file path
    from urllib.parse import unquote
    file_path = unquote(file_path)
    
    # Security: ensure the path doesn't escape the storage directory
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    from mimetypes import guess_type
    media_type, _ = guess_type(file_path)
    
    # Return the file
    from fastapi.responses import FileResponse
    return FileResponse(file_path, media_type=media_type)

# --- Unified Orchestration Service ---
async def execute_orchestration(
    prompt: Optional[str],
    thread_id: str,
    user_response: Optional[str] = None,
    files: Optional[List[FileObject]] = None,
    stream_callback=None
):
    """
    Unified orchestration logic that correctly persists and merges file context
    across all turns in a conversation. Simplified and more robust version.
    """
    logger.info(f"Starting orchestration for thread_id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    # Get the current state of the conversation from the checkpointer
    current_checkpoint = checkpointer.get(config)
    # Extract the state from the checkpoint if it exists
    # The checkpoint structure is { "values": State, "next": List[str], "config": RunnableConfig }
    current_conversation = current_checkpoint.get("values", {}) if current_checkpoint else {}

    # --- State Initialization ---
    if user_response:
        # Continuing an interactive workflow where the user answered a question
        logger.info(f"Resuming conversation for thread_id: {thread_id} with user response.")
        initial_state = dict(current_conversation)  # Convert to dict if it's a State object
        initial_state["user_response"] = user_response
        initial_state["pending_user_input"] = False
        initial_state["question_for_user"] = None
        initial_state["parse_retry_count"] = 0
        if "original_prompt" in initial_state:
            initial_state["original_prompt"] = f"{initial_state['original_prompt']}\n\nAdditional context from user: {user_response}"
        else:
            initial_state["original_prompt"] = user_response
            
        # Clear any previous final response to avoid confusion
        initial_state["final_response"] = None

    elif prompt and current_conversation:
        # A new prompt is sent in an existing conversation thread
        logger.info(f"Continuing conversation for thread_id: {thread_id} with new prompt.")
        initial_state = {
            # Carry over essential long-term memory from the previous turn
            "messages": current_conversation.get("messages", []) + [HumanMessage(content=prompt)],
            "completed_tasks": [],  # Start fresh for new prompt, but keep conversation history
            "uploaded_files": current_conversation.get("uploaded_files", []), # Persist files

            # Reset short-term memory for the new task
            "original_prompt": prompt,
            "parsed_tasks": [],
            "user_expectations": {},
            "candidate_agents": {},
            "task_agent_pairs": [],
            "task_plan": [],
            "final_response": None,
            "pending_user_input": False,
            "question_for_user": None,
            "user_response": None,
            "parsing_error_feedback": None,
            "parse_retry_count": 0,
            "needs_complex_processing": None,  # Let analyze_request determine this
            "analysis_reasoning": None,
        }
    
    elif current_conversation:
        # Resuming without a new prompt or user response (e.g., status check)
        initial_state = dict(current_conversation) # Convert to dict if it's a State object
        logger.info(f"Checking status for thread_id: {thread_id}")

    else:
        # A brand new conversation
        if not prompt:
            raise ValueError("Prompt is required for new conversations")
        logger.info(f"Starting new conversation for thread_id: {thread_id}")
        
        # Clear orchestrator log for new conversation
        clear_orchestrator_log()
        
        initial_state = {
            "original_prompt": prompt,
            "messages": [HumanMessage(content=prompt)],
            "uploaded_files": [], # Start with an empty file list
            "parsed_tasks": [],
            "user_expectations": {},
            "candidate_agents": {},
            "task_agent_pairs": [],
            "task_plan": [],
            "completed_tasks": [],
            "final_response": None,
            "pending_user_input": False,
            "question_for_user": None,
            "user_response": None,
            "parsing_error_feedback": None,
            "parse_retry_count": 0,
            "needs_complex_processing": None,  # Let analyze_request determine this
            "analysis_reasoning": None,
        }

    # --- File Merging Logic ---
    # This block runs for EVERY turn, ensuring new files are always added to the state
    if files:
        # Use a dictionary keyed by file_path to merge lists and avoid duplicates
        file_map = {f['file_path']: f for f in initial_state.get("uploaded_files", [])}
        for new_file in files:
            file_map[new_file.file_path] = new_file.model_dump()
        
        initial_state["uploaded_files"] = list(file_map.values())
        logger.info(f"File context updated. Total unique files in state: {len(initial_state['uploaded_files'])}")
    
    # Store the prepared initial state before running the graph
    with store_lock:
        conversation_store[thread_id] = initial_state.copy()

    final_state = None
    try:
        if stream_callback:
            # Streaming mode for WebSocket
            node_count = 0
            expected_nodes = ["analyze_request", "parse_prompt", "agent_directory_search", "rank_agents", "plan_execution", "execute_batch", "aggregate_responses"]

            async for event in graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    node_count += 1
                    # More accurate progress calculation
                    progress = min((node_count / len(expected_nodes)) * 100, 100) if expected_nodes else 50
                    if isinstance(node_output, dict):
                        final_state = {**final_state, **node_output} if final_state else node_output
                    await stream_callback(node_name, node_output, progress, node_count, thread_id)
                    if isinstance(node_output, dict) and node_output.get("pending_user_input"):
                        logger.info(f"Workflow paused for user input in thread_id: {thread_id}")
                        break
            if not final_state:
                final_state = await graph.ainvoke(initial_state, config=config)
        else:
            # Single response mode for HTTP
            final_state = await graph.ainvoke(initial_state, config=config)

        # Store the final state after the graph run
        with store_lock:
            conversation_store[thread_id] = final_state.copy()
            
        # Save conversation history using orchestrator's save routine
        try:
            from orchestrator.graph import save_conversation_history
            save_conversation_history(final_state, {"configurable": {"thread_id": thread_id}})
        except Exception as e:
            logger.error(f"Failed to save conversation history for {thread_id}: {e}")

        # Ensure a plan file is saved for every conversation
        try:
            from orchestrator.graph import save_plan_to_file
            save_plan_to_file({**final_state, "thread_id": thread_id})
        except Exception as e:
            logger.error(f"Failed to save plan file for thread {thread_id}: {e}")

        logger.info(f"Orchestration completed for thread_id: {thread_id}")
        return final_state

    except Exception as e:
        error_msg = str(e)
        if "No valid tasks to process" in error_msg or "No tasks to rank" in error_msg or "Halting: No agents found for task ''" in error_msg:
            logger.warning(f"No valid tasks could be parsed from prompt for thread_id {thread_id}. Original prompt: '{prompt}'")
            error_state = {
                "final_response": f"I couldn't identify any specific tasks from your message: '{prompt}'. Could you please be more specific?",
                "pending_user_input": False,
                "question_for_user": None,
            }
            with store_lock:
                conversation_store[thread_id] = {**initial_state, **error_state}
            return conversation_store[thread_id]

        logger.error(f"Error during orchestration for thread_id {thread_id}: {e}", exc_info=True)
        raise

def process_node_data(node_name: str, node_output, progress: float, node_count: int, thread_id: str = None):
    """Extract meaningful data from node output consistently"""
    serializable_data = {}
    node_specific_data = {
        "progress_percentage": round(progress, 1),
        "node_sequence": node_count
    }

    if isinstance(node_output, dict):
        for key, value in node_output.items():
            serializable_data[key] = serialize_complex_object(value)

            # Extract node-specific meaningful data
            if node_name == "parse_prompt" and key == "parsed_tasks":
                if value:
                    # Handle both Task objects and dictionaries
                    task_names = []
                    logger.debug(f"Processing {len(value)} parsed tasks for thread_id {thread_id}")
                    for i, task in enumerate(value):
                        if hasattr(task, 'task_name'):
                            task_name = task.task_name
                            task_names.append(task_name)
                            logger.debug(f"Task {i}: {task_name} (Task object)")
                        elif isinstance(task, dict) and 'task_name' in task:
                            task_name = task['task_name']
                            task_names.append(task_name)
                            logger.debug(f"Task {i}: {task_name} (dict)")
                        else:
                            task_str = str(task)
                            task_names.append(task_str)
                            logger.debug(f"Task {i}: {task_str} (fallback)")

                        # Check for empty task names
                        if not task_names[-1] or task_names[-1].strip() == '':
                            logger.warning(f"Empty task name detected at index {i} for thread_id {thread_id}")

                    node_specific_data["tasks_identified"] = len(value)
                    node_specific_data["task_names"] = task_names
                    node_specific_data["description"] = "Identified and parsed user tasks"

                    # Filter out empty task names for logging
                    non_empty_tasks = [name for name in task_names if name and name.strip()]
                    logger.info(f"Successfully parsed {len(non_empty_tasks)} non-empty tasks: {non_empty_tasks}")
                else:
                    node_specific_data["tasks_identified"] = 0
                    node_specific_data["task_names"] = []
                    node_specific_data["description"] = "No tasks identified from prompt"
                    logger.warning(f"No tasks were parsed from prompt for thread_id {thread_id}")

            elif node_name == "agent_directory_search" and key == "candidate_agents":
                node_specific_data["agents_found"] = sum(len(agents) for agents in value.values()) if value else 0
                node_specific_data["tasks_with_agents"] = list(value.keys()) if value else []
                node_specific_data["description"] = "Found candidate agents for tasks"

            elif node_name == "rank_agents" and key == "task_agent_pairs":
                node_specific_data["pairs_created"] = len(value) if value else 0
                node_specific_data["description"] = "Ranked and paired agents with tasks"
                if value:
                    pairs_data = [serialize_complex_object(pair) for pair in value]
                    node_specific_data["task_agent_pairs"] = pairs_data

            elif node_name == "plan_execution" and key == "task_plan":
                node_specific_data["execution_plan_ready"] = True
                node_specific_data["planned_tasks"] = len(value) if value else 0
                node_specific_data["description"] = "Created execution plan"

            elif node_name == "execute_batch" and key in ["completed_tasks", "final_response"]:
                if key == "completed_tasks":
                    node_specific_data["tasks_completed"] = len(value) if value else 0
                    node_specific_data["description"] = "Executed task batch"
                elif key == "final_response":
                    node_specific_data["has_final_response"] = bool(value)
    else:
        serializable_data = {"raw_output": str(node_output)}
        node_specific_data["description"] = f"Node {node_name} completed"

    return {**serializable_data, **node_specific_data}

# --- API Endpoints ---
@app.post("/api/chat", response_model=ProcessResponse)
async def find_agents(request: ProcessRequest):
    """
    Receives a prompt, runs it through the agent-finding graph,
    and returns the selected primary and fallback agents for each task.
    Now supports interactive workflows that may require user input.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info(f"Starting agent search with thread_id: {thread_id}")

    try:
        final_state = await execute_orchestration(
            prompt=request.prompt,
            thread_id=thread_id,
            files=request.files,  # Pass the files to the orchestrator
            stream_callback=None
        )

        # Check if workflow is paused for user input
        if final_state.get("pending_user_input"):
            logger.info(f"Workflow paused for user input in thread_id: {thread_id}")
            return ProcessResponse(
                message="Additional information required to complete your request.",
                thread_id=thread_id,
                task_agent_pairs=[],
                final_response=None,
                pending_user_input=True,
                question_for_user=final_state.get("question_for_user")
            )

        task_agent_pairs = final_state.get("task_agent_pairs", [])
        final_response_str = final_state.get("final_response")

        # Check for a valid outcome
        if not task_agent_pairs and not final_response_str and not final_state.get("pending_user_input"):
            logger.warning(f"Could not parse any tasks or generate a response for thread_id: {thread_id}. Prompt: '{request.prompt}'")
            raise HTTPException(
                status_code=404,
                detail=f"I couldn't identify any specific tasks or generate a response from your message: '{request.prompt}'. Could you please be more specific?"
            )

        logger.info(f"Successfully processed request for thread_id: {thread_id}")

        return ProcessResponse(
            message="Successfully processed the request.",
            thread_id=thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly to preserve the status code
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during graph execution for thread_id {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/api/chat/continue", response_model=ProcessResponse)
async def continue_conversation(user_response: UserResponse):
    """
    Continue a paused conversation by providing user response to a question.
    """
    logger.info(f"Continuing conversation for thread_id: {user_response.thread_id} with response: {user_response.response[:100]}...")

    # Check if conversation exists
    with store_lock:
        existing_conversation = conversation_store.get(user_response.thread_id)

    if not existing_conversation:
        logger.warning(f"No existing conversation found for thread_id: {user_response.thread_id}")
        raise HTTPException(status_code=404, detail="Conversation thread not found. Please start a new conversation.")

    logger.info(f"Found existing conversation for thread_id: {user_response.thread_id}, pending_input: {existing_conversation.get('pending_user_input', False)}")

    try:
        final_state = await execute_orchestration(
            prompt=None, # No new prompt needed
            thread_id=user_response.thread_id,
            user_response=user_response.response,
            files=user_response.files,  # Pass files if provided
            stream_callback=None
        )

        # Check if workflow is paused again for more user input
        if final_state.get("pending_user_input"):
            logger.info(f"Workflow paused again for user input in thread_id: {user_response.thread_id}")
            return ProcessResponse(
                message="Additional information required to complete your request.",
                thread_id=user_response.thread_id,
                task_agent_pairs=[],
                final_response=None,
                pending_user_input=True,
                question_for_user=final_state.get("question_for_user")
            )

        task_agent_pairs = final_state.get("task_agent_pairs", [])
        final_response_str = final_state.get("final_response")

        logger.info(f"Successfully continued conversation for thread_id: {user_response.thread_id}")

        return ProcessResponse(
            message="Successfully processed the continued conversation.",
            thread_id=user_response.thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred during conversation continuation for thread_id {user_response.thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/api/chat/status/{thread_id}", response_model=ConversationStatus)
async def get_conversation_status(thread_id: str):
    """
    Get the current status of a conversation thread.
    """
    try:
        # Get conversation from our store
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

@app.delete("/api/chat/{thread_id}")
async def clear_conversation(thread_id: str):
    """
    Clear a conversation thread from memory.
    """
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

@app.get("/api/chat/debug/conversations")
async def debug_conversations():
    """
    Debug endpoint to see all active conversations (remove in production).
    """
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

@app.get("/api/conversations", response_model=List[str])
async def get_all_conversations():
    """
    Retrieves a list of all conversation thread_ids that have history.
    """
    if not os.path.isdir(CONVERSATION_HISTORY_DIR):
        return []
    
    files = os.listdir(CONVERSATION_HISTORY_DIR)
    # Return filenames without the .json extension, sorted by modification time (newest first)
    files_with_path = [os.path.join(CONVERSATION_HISTORY_DIR, f) for f in files if f.endswith(".json")]
    files_with_path.sort(key=os.path.getmtime, reverse=True)
    return [os.path.splitext(os.path.basename(f))[0] for f in files_with_path]

@app.get("/api/conversations/{thread_id}")
async def get_conversation_history(thread_id: str):
    """
    Retrieves the full, standardized conversation state from its JSON file.
    This is the single source of truth for a conversation's history.
    """
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
    
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="Conversation history not found.")
        
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            # The file already contains the standardized, serializable state.
            # No further processing is needed.
            history_data = json.load(f)
        return history_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse conversation history for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse conversation history file.")
    except Exception as e:
        logger.error(f"Error loading conversation history for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while loading the conversation: {str(e)}")

@app.get("/api/plan/{thread_id}", response_model=PlanResponse)
async def get_agent_plan(thread_id: str):
    """
    Retrieves the markdown execution plan for a given conversation thread.
    """
    # Check both possible locations for plan files
    plan_dirs = ["agent_plans", "backend/agent_plans"]  # Check root first, then backend/
    file_path = None
    
    for plan_dir in plan_dirs:
        temp_path = os.path.join(plan_dir, f"{thread_id}-plan.md")
        if os.path.exists(temp_path):
            file_path = temp_path
            break

    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"Plan file not found for thread_id: {thread_id} in any location: {plan_dirs}"
        )

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return PlanResponse(thread_id=thread_id, content=content)
    except Exception as e:
        logger.error(f"Error reading plan file for thread_id {thread_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while reading the plan file."
        )

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent orchestration updates.
    Uses the unified orchestration service with streaming enabled.
    Now supports interactive workflows with user input.
    """
    await websocket.accept()
    thread_id = None  # Initialize thread_id to None
    
    try:
        while True:  # Keep the connection open for multiple messages
            # Wait for message from client
            try:
                data = await websocket.receive_json()
            except Exception as e:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Failed to receive message",
                    "thread_id": thread_id or "unknown",
                    "timestamp": time.time()
                })
                continue  # Continue waiting for messages

            # Get thread_id from client, or generate a new one if not provided
            thread_id = data.get("thread_id") or str(uuid.uuid4())
            prompt = data.get("prompt")
            user_response = data.get("user_response")  # For continuing conversations
            files_data = data.get("files", [])  # Get files from WebSocket message

            logger.info(f"WebSocket received message with thread_id: {thread_id}")

            if not prompt and not user_response:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Missing 'prompt' field for new conversation or 'user_response' for continuing",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                })
                continue  # Continue waiting for messages

            logger.info(f"Received {'prompt' if prompt else 'user response'} for thread_id {thread_id}")

            # Send acknowledgment
            await websocket.send_json({
                "node": "__start__",
                "thread_id": thread_id,
                "message": "Starting agent orchestration..." if prompt else "Continuing conversation...",
                "data": {
                    "original_prompt": prompt,
                    "user_response": user_response,
                    "status": "initializing",
                    "timestamp": time.time()
                }
            })

            # Define stream callback for WebSocket updates
            async def stream_callback(node_name: str, node_output, progress: float, node_count: int, thread_id: str):
                try:
                    # Process node data using unified helper
                    final_data = process_node_data(node_name, node_output, progress, node_count, thread_id)

                    # Send enhanced node update
                    await websocket.send_json({
                        "node": node_name,
                        "data": final_data,
                        "thread_id": thread_id,
                        "status": "completed",
                        "timestamp": time.time()
                    })
                    logger.info(f"Streamed update from node '{node_name}' (#{node_count}) for thread_id {thread_id} - Progress: {progress:.1f}%")

                except Exception as process_error:
                    logger.warning(f"Failed to process data from node '{node_name}': {process_error}")
                    # Send a simplified message instead
                    try:
                        await websocket.send_json({
                            "node": node_name,
                            "data": {
                                "status": "completed",
                                "message": f"Node {node_name} completed successfully",
                                "warning": f"Data processing failed: {str(process_error)}",
                                "progress_percentage": round(progress, 1),
                                "node_sequence": node_count,
                                "description": f"Node {node_name} processed with warning"
                            },
                            "thread_id": thread_id,
                            "timestamp": time.time()
                        })
                    except Exception as send_error:
                        logger.error(f"Failed to send fallback message for node '{node_name}': {send_error}")

            # Convert files data to FileObject instances
            file_objects = []
            if files_data:
                logger.info(f"Processing {len(files_data)} files from WebSocket message")
                for file_data in files_data:
                    if isinstance(file_data, dict) and 'file_name' in file_data:
                        try:
                            file_objects.append(FileObject(
                                file_name=file_data['file_name'],
                                file_path=file_data['file_path'],
                                file_type=file_data['file_type']
                            ))
                            logger.info(f"Added file: {file_data['file_name']} at {file_data['file_path']}")
                        except Exception as e:
                            logger.error(f"Failed to create FileObject from {file_data}: {e}")
            
            # Use unified orchestration service with streaming
            final_state = await execute_orchestration(
                prompt=prompt,
                thread_id=thread_id,
                user_response=user_response,
                files=file_objects if file_objects else None,
                stream_callback=stream_callback
            )

            # Check if workflow is paused for user input
            if final_state.get("pending_user_input"):
                await websocket.send_json({
                    "node": "__user_input_required__",
                    "thread_id": thread_id,
                    "data": {
                        "question_for_user": final_state.get("question_for_user")
                    },
                    "message": "Additional information required to complete your request.",
                    "status": "pending_user_input",
                    "timestamp": time.time()
                })
                logger.info(f"WebSocket workflow paused for user input in thread_id {thread_id}")
                continue  # Continue waiting for user response message

            # --- **NEW**: Send the complete, standardized state object ---
            # This ensures the frontend has all information needed to update the UI,
            # including plan, metadata, and attachments, without needing a separate API call.
            from orchestrator.graph import get_serializable_state
            serializable_state = get_serializable_state(final_state, thread_id)

            await websocket.send_json({
                "node": "__end__",
                "thread_id": thread_id,
                "data": serializable_state, # Send the entire state object
                "message": "Agent orchestration completed successfully.",
                "status": "completed",
                "timestamp": time.time()
            })

            logger.info(f"WebSocket stream completed successfully for thread_id {thread_id}")
            
            # For single-turn conversations, we might want to close the connection
            # But for multi-turn conversations, we keep it open
            # The frontend can send a special message to close the connection if needed

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread_id {thread_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket stream for thread_id {thread_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "node": "__error__",
                "thread_id": thread_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"An error occurred during orchestration: {str(e)}",
                "status": "error",
                "timestamp": time.time()
            })
        except:
            # If we can't send the error, the connection is likely already closed
            logger.error(f"Could not send error message to WebSocket for thread_id {thread_id}")
    finally:
        # Close the WebSocket connection
        try:
            await websocket.close()
        except:
            pass  # Connection might already be closed
        logger.info(f"WebSocket connection closed for thread_id {thread_id}")

@app.post("/api/agents/register", response_model=AgentCard)
def register_or_update_agent(agent_data: AgentCard, response: Response, db: Session = Depends(get_db)):
    db_agent = db.query(Agent).options(
        joinedload(Agent.capability_vectors),
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters) # Eager load parameters
    ).get(agent_data.id)

    agent_dict = agent_data.model_dump(
        mode='json',
        exclude={"endpoints"},
        exclude_none=True,
        exclude_unset=True
    )

    if db_agent:
        for key, value in agent_dict.items():
            setattr(db_agent, key, value)

        # Clear old related data
        db_agent.capability_vectors.clear()
        db_agent.endpoints.clear()
        response.status_code = status.HTTP_200_OK
    else:
        db_agent = Agent(**agent_dict)
        db.add(db_agent)
        response.status_code = status.HTTP_201_CREATED

    if agent_data.capabilities:
        for cap_text in agent_data.capabilities:
            embedding_vector = model.encode(cap_text)
            new_capability = AgentCapability(
                agent=db_agent,
                capability_text=cap_text,
                embedding=embedding_vector
            )
            db.add(new_capability)

    # *** START: CORRECTED ENDPOINT AND PARAMETER LOGIC ***
    if agent_data.endpoints:
        for endpoint_data in agent_data.endpoints:
            # Create the main endpoint record
            new_endpoint = AgentEndpoint(
                agent=db_agent,
                endpoint=str(endpoint_data.endpoint),
                http_method=endpoint_data.http_method,
                description=endpoint_data.description
            )
            db.add(new_endpoint)

            # Create and associate its parameters in a nested loop
            if endpoint_data.parameters:
                for param_data in endpoint_data.parameters:
                    new_param = EndpointParameter(
                        endpoint=new_endpoint,  # Link to the endpoint being created
                        name=param_data.name,
                        description=param_data.description,
                        param_type=param_data.param_type,
                        required=param_data.required,
                        default_value=param_data.default_value
                    )
                    db.add(new_param)
    # *** END: CORRECTED ENDPOINT AND PARAMETER LOGIC ***

    db.commit()
    db.refresh(db_agent)

    return AgentCard.model_validate(db_agent)

@app.get("/api/agents/search", response_model=List[AgentCard])
def search_agents(
    db: Session = Depends(get_db),
    capabilities: List[str] = Query(..., description="A list of task names to find capable agents for."),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None),
    similarity_threshold: float = Query(0.5, description="Cosine distance threshold (lower is stricter).")
):
    """
    Finds active agents that match ANY of the specified capabilities using vector search.
    """
    if not capabilities:
        return []

    conditions = []
    for task_name in capabilities:
        query_vector = model.encode(task_name)
        # Subquery to find agent_ids for this one task
        subquery = select(AgentCapability.agent_id).where(
            AgentCapability.embedding.cosine_distance(query_vector) < similarity_threshold
        )
        conditions.append(Agent.id.in_(subquery))

    # Combine conditions with OR logic
    query = db.query(Agent).options(
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters) # Eager load parameters
    ).filter(Agent.status == 'active').filter(or_(*conditions))

    # Apply optional price and rating filters
    if max_price is not None:
        query = query.filter(Agent.price_per_call_usd <= max_price)
    if min_rating is not None:
        query = query.filter(Agent.rating >= min_rating)

    return query.all()

@app.get("/api/agents/all", response_model=List[AgentCard])
def get_all_agents(db: Session = Depends(get_db)):
    """
    Returns all agents in the agents table as a JSON list.
    """
    return db.query(Agent).options(
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters) # Eager load parameters
    ).all()

@app.get("/api/agents/{agent_id}", response_model=AgentCard)
def get_agent(agent_id: str, db: Session = Depends(get_db)):
    db_agent = db.query(Agent).options(
        joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters) # Eager load parameters
    ).get(agent_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    return db_agent

@app.post("/api/agents/{agent_id}/rate", response_model=AgentCard)
def rate_agent(agent_id: str, rating: float = Body(..., embed=True), db: Session = Depends(get_db)):
    """
    Update the agent's rating as the mean of the current rating and the new user rating.
    """
    db_agent = db.get(Agent, agent_id)
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5.")
    # Calculate new mean rating
    current_rating = db_agent.rating if db_agent.rating is not None else 0.0
    count = db_agent.rating_count if db_agent.rating_count is not None else 0
    new_rating = ((current_rating * count) + rating) / (count + 1) if count > 0 else rating
    db_agent.rating = float(new_rating)
    db_agent.rating_count = int(count + 1)
    db.commit()
    db.refresh(db_agent)
    return AgentCard.model_validate(db_agent)

@app.post("/api/agents/by-name/{agent_name}/rate", response_model=AgentCard)
def rate_agent_by_name(agent_name: str, rating: float = Body(..., embed=True), db: Session = Depends(get_db)):
    """
    Update the agent's rating using the agent's name as a fallback.
    """
    db_agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail="Agent not found!")
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5.")
    # Calculate new mean rating
    current_rating = db_agent.rating if db_agent.rating is not None else 0.0
    count = db_agent.rating_count if db_agent.rating_count is not None else 0
    new_rating = ((current_rating * count) + rating) / (count + 1) if count > 0 else rating
    db_agent.rating = float(new_rating)
    db_agent.rating_count = int(count + 1)
    db.commit()
    db.refresh(db_agent)
    return AgentCard.model_validate(db_agent)

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    # Start all agent servers in separate terminals
    start_agent_servers()

    # Run the main FastAPI app
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

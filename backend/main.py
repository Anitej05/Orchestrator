# Project_Agent_Directory/main.py
import uuid
import logging
import json
import asyncio
import time
import asyncio
from datetime import datetime
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
from fastapi import FastAPI, HTTPException, Depends, status, Query, Response, WebSocket, WebSocketDisconnect, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, cast, String, select
from sentence_transformers import SentenceTransformer

# --- Local Application Imports ---
CONVERSATION_HISTORY_DIR = "conversation_history"
from database import SessionLocal
from models import Agent, StatusEnum, AgentCapability, AgentEndpoint, EndpointParameter, Workflow, WorkflowExecution, UserThread, WorkflowSchedule, WorkflowWebhook
from schemas import AgentCard, ProcessRequest, ProcessResponse, PlanResponse, FileObject
from orchestrator.graph import ForceJsonSerializer, graph, create_graph_with_checkpointer, create_execution_subgraph, messages_from_dict, messages_to_dict, serialize_complex_object
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
# Use UTF-8 encoding for console output to handle emojis
import io
console_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
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
file_handler = logging.FileHandler(orchestrator_log_file, mode='w', encoding='utf-8')  # UTF-8 encoding for emojis
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

# Create the main graph and execution subgraph with the checkpointer
graph = create_graph_with_checkpointer(checkpointer)
execution_subgraph = create_execution_subgraph(checkpointer)

# Simple in-memory conversation store as backup
conversation_store: Dict[str, Dict[str, Any]] = {}
from threading import Lock
store_lock = Lock()

# Global store for live canvas updates during browser execution
live_canvas_updates: Dict[str, Dict[str, Any]] = {}
canvas_lock = Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

# --- Static Files for Screenshots ---
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Create storage directory if it doesn't exist
storage_path = Path("storage").absolute()
storage_path.mkdir(exist_ok=True)
(storage_path / "images").mkdir(exist_ok=True)

# Mount storage directory for serving screenshots
app.mount("/storage", StaticFiles(directory=str(storage_path)), name="storage")
logger.info(f"Mounted /storage for serving screenshot files from {storage_path}")

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
    global agent_processes
    
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
            process = None
            if platform.system() == "Windows":
                # For Windows, use subprocess.Popen with proper parameters to run in background
                try:
                    with open(log_path, 'w') as log_file:
                        process = subprocess.Popen(
                            [sys.executable, agent_path],
                            stdout=log_file,
                            stderr=log_file,
                            creationflags=subprocess.CREATE_NO_WINDOW  # Run without console window
                        )
                        # Track the process globally
                        agent_processes.append(process)
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
                    # Track the process globally
                    agent_processes.append(process)

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
    stream_callback=None,
    task_event_callback=None,
    planning_mode: bool = False
):
    """
    Unified orchestration logic that correctly persists and merges file context
    across all turns in a conversation. Simplified and more robust version.
    """
    logger.info(f"Starting orchestration for thread_id: {thread_id}, planning_mode: {planning_mode}")

    # Build config with task_event_callback if provided
    config = {"configurable": {"thread_id": thread_id}}
    if task_event_callback:
        config["configurable"]["task_event_callback"] = task_event_callback
        logger.info(f"✅ Task event callback registered for real-time streaming")

    # Get the current state of the conversation from the in-memory store first (most recent)
    # Fall back to checkpointer if not in memory
    with store_lock:
        current_conversation = conversation_store.get(thread_id)
    
    if not current_conversation:
        # If not in memory, try checkpointer
        current_checkpoint = checkpointer.get(config)
        # Extract the state from the checkpoint if it exists
        # The checkpoint structure is { "values": State, "next": List[str], "config": RunnableConfig }
        current_conversation = current_checkpoint.get("values", {}) if current_checkpoint else {}

    # --- State Initialization ---
    if user_response:
        # Continuing an interactive workflow where the user answered a question
        logger.info(f"Resuming conversation for thread_id: {thread_id} with user response.")
        logger.info(f"USER RESPONSE BRANCH: user_response='{user_response}', planning_mode={planning_mode}")
        initial_state = dict(current_conversation)  # Convert to dict if it's a State object
        initial_state["user_response"] = user_response
        initial_state["pending_user_input"] = False
        initial_state["question_for_user"] = None
        initial_state["parse_retry_count"] = 0
        
        # Handle plan approval responses
        needs_approval = initial_state.get("needs_approval", False)
        print(f"!!! USER RESPONSE: needs_approval={needs_approval}, response='{user_response}' !!!")
        logger.info(f"Checking approval state: needs_approval={needs_approval}, user_response='{user_response}'")
        
        if needs_approval:
            user_response_lower = user_response.lower().strip()
            logger.info(f"Processing approval response: '{user_response_lower}'")
            
            if user_response_lower in ["approve", "yes", "proceed", "continue", "execute", "go", "ok"]:
                print(f"!!! USER APPROVED - Setting planning_mode=False and plan_approved=True !!!")
                logger.info(f"User APPROVED execution plan for thread_id: {thread_id}")
                logger.info(f"Clearing approval flags and continuing execution")
                # Simply turn off planning mode and clear approval flags
                initial_state["needs_approval"] = False
                initial_state["approval_required"] = False
                initial_state["planning_mode"] = False  # This is the key - let it run normally now
                initial_state["pending_user_input"] = False
                initial_state["question_for_user"] = None
                initial_state["plan_approved"] = True  # NEW: Flag to skip validation and go straight to execution
                logger.info(f"State after approval: needs_approval={initial_state['needs_approval']}, planning_mode={initial_state['planning_mode']}, pending_user_input={initial_state['pending_user_input']}, plan_approved={initial_state.get('plan_approved')}")
                # Don't modify original_prompt - keep everything as is
            elif user_response_lower in ["cancel", "no", "stop", "abort", "reject"]:
                logger.info(f"User CANCELLED execution plan for thread_id: {thread_id}")
                print(f"!!! USER CANCELLED - Stopping execution !!!")
                initial_state["final_response"] = "Execution cancelled by user."
                return initial_state
            else:
                # Invalid response
                logger.warning(f"Invalid approval response: '{user_response_lower}'")
                print(f"!!! INVALID APPROVAL RESPONSE: '{user_response_lower}' !!!")
                initial_state["pending_user_input"] = True
                initial_state["question_for_user"] = "Please respond with 'approve' to proceed or 'cancel' to stop."
                return initial_state
        else:
            # Regular user response - add to context
            if "original_prompt" in initial_state:
                initial_state["original_prompt"] = f"{initial_state['original_prompt']}\n\nAdditional context: {user_response}"
            else:
                initial_state["original_prompt"] = user_response
            
        # Clear any previous final response to avoid confusion
        initial_state["final_response"] = None

    elif prompt and current_conversation:
        # A new prompt is sent in an existing conversation thread
        logger.info(f"NEW PROMPT IN EXISTING CONVERSATION BRANCH")
        logger.info(f"Continuing conversation for thread_id: {thread_id} with new prompt, planning_mode: {planning_mode}")
        logger.info(f"Prompt content: '{prompt[:100]}'")
        
        # Use MessageManager to add new message without duplicates
        from orchestrator.message_manager import MessageManager
        existing_messages = current_conversation.get("messages", [])
        # Create message with metadata to preserve timestamp and ID
        import hashlib
        timestamp = time.time()
        unique_string = f"human:{prompt}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        new_user_message = HumanMessage(
            content=prompt,
            additional_kwargs={"timestamp": timestamp, "id": msg_id}
        )
        updated_messages = MessageManager.add_message(existing_messages, new_user_message)
        logger.info(f"Continuing conversation. Total messages: {len(updated_messages)}")
        
        initial_state = {
            # Carry over essential long-term memory from the previous turn
            "messages": updated_messages,
            "completed_tasks": current_conversation.get("completed_tasks", []),  # Preserve completed tasks for context
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
            "planning_mode": planning_mode,  # Set planning mode from parameter
            "plan_approved": False,  # Reset plan_approved for new request
        }
    
    elif current_conversation:
        # Resuming without a new prompt or user response (e.g., status check)
        initial_state = dict(current_conversation) # Convert to dict if it's a State object
        logger.info(f"Checking status for thread_id: {thread_id}")

    else:
        # A brand new conversation
        if not prompt:
            raise ValueError("Prompt is required for new conversations")
        logger.info(f"NEW CONVERSATION BRANCH")
        logger.info(f"Starting new conversation for thread_id: {thread_id}, planning_mode: {planning_mode}")
        
        # Clear orchestrator log for new conversation
        clear_orchestrator_log()
        
        # Create message with metadata to preserve timestamp and ID
        import hashlib
        timestamp = time.time()
        unique_string = f"human:{prompt}:{timestamp}"
        msg_id = hashlib.md5(unique_string.encode()).hexdigest()[:16]
        
        initial_state = {
            "original_prompt": prompt,
            "messages": [HumanMessage(
                content=prompt,
                additional_kwargs={"timestamp": timestamp, "id": msg_id}
            )],
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
            "planning_mode": planning_mode,  # Set planning mode from parameter
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
        # Determine if we should use execution subgraph or main graph
        is_post_approval = user_response is not None and initial_state.get("plan_approved") == True
        
        # Select the appropriate graph
        if is_post_approval:
            logger.info(f"Graph execution mode: POST-APPROVAL EXECUTION using subgraph")
            print(f"!!! GRAPH EXECUTION: POST-APPROVAL - Using execution subgraph !!!")
            selected_graph = execution_subgraph
            graph_input = initial_state
        else:
            logger.info(f"Graph execution mode: NORMAL execution using main graph")
            print(f"!!! GRAPH EXECUTION: NORMAL - Using main graph !!!")
            selected_graph = graph
            graph_input = initial_state
        
        if stream_callback:
            # Streaming mode for WebSocket
            node_count = 0
            expected_nodes = ["load_history", "execute_batch", "evaluate_agent_response", "generate_final_response"] if is_post_approval else ["analyze_request", "parse_prompt", "agent_directory_search", "rank_agents", "plan_execution", "execute_batch", "aggregate_responses"]

            async for event in selected_graph.astream(graph_input, config=config, stream_mode="updates"):
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
            
            # After streaming, get the actual state if not available
            if not final_state:
                # If no state from streaming, invoke to get final state
                final_state = await selected_graph.ainvoke(graph_input, config=config)
        else:
            # Single response mode for HTTP
            final_state = await selected_graph.ainvoke(graph_input, config=config)

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

        # Check if there's canvas data from browser agent
        canvas_data = {}
        with canvas_lock:
            if thread_id in live_canvas_updates:
                canvas_data = live_canvas_updates[thread_id]
                logger.info(f"📊 Including canvas data in response for thread {thread_id}")

        return ProcessResponse(
            message="Successfully processed the request.",
            thread_id=thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None,
            has_canvas=canvas_data.get('has_canvas', False),
            canvas_content=canvas_data.get('canvas_content'),
            canvas_type=canvas_data.get('canvas_type'),
            browser_view=canvas_data.get('browser_view'),
            plan_view=canvas_data.get('plan_view'),
            current_view=canvas_data.get('current_view', 'browser')
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

        # Check if there's canvas data from browser agent
        canvas_data = {}
        with canvas_lock:
            if user_response.thread_id in live_canvas_updates:
                canvas_data = live_canvas_updates[user_response.thread_id]
                logger.info(f"📊 Including canvas data in continue response for thread {user_response.thread_id}")

        return ProcessResponse(
            message="Successfully processed the continued conversation.",
            thread_id=user_response.thread_id,
            task_agent_pairs=task_agent_pairs,
            final_response=final_response_str,
            pending_user_input=False,
            question_for_user=None,
            has_canvas=canvas_data.get('has_canvas', False),
            canvas_content=canvas_data.get('canvas_content'),
            canvas_type=canvas_data.get('canvas_type'),
            browser_view=canvas_data.get('browser_view'),
            plan_view=canvas_data.get('plan_view'),
            current_view=canvas_data.get('current_view', 'browser')
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred during conversation continuation for thread_id {user_response.thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/api/canvas/update")
async def update_canvas(update_data: Dict[str, Any] = Body(...)):
    """
    Receive canvas updates from browser agent and store for WebSocket streaming
    Supports both browser view (screenshots) and plan view (task plan)
    """
    try:
        thread_id = update_data.get("thread_id")
        if not thread_id:
            return {"status": "error", "message": "thread_id required"}
        
        logger.info(f"📥 Received canvas update for thread {thread_id} (step {update_data.get('step', 0)})")
        
        screenshot_data = update_data.get("screenshot_data", "")
        url = update_data.get("url", "")
        step = update_data.get("step", 0)
        task = update_data.get("task", "")
        task_plan = update_data.get("task_plan", [])  # New: task plan for plan view
        current_action = update_data.get("current_action", "")  # New: current action
        
        # Create browser view HTML with embedded base64 image
        browser_view_html = f'''
        <div style="text-align: center;">
            <img src="data:image/png;base64,{screenshot_data}" alt="Browser live view" style="width: 100%; max-width: 1200px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" />
            <div style="margin-top: 10px; color: #666; font-size: 14px;">
                <strong>🔴 Live Browser View</strong> | Step {step} | {url[:60] if url else 'Loading...'}
            </div>
        </div>
        '''
        
        # Create plan view HTML with task progress
        # Calculate progress
        completed_count = sum(1 for t in task_plan if t.get('status') == 'completed')
        total_count = len(task_plan)
        progress_percent = (completed_count / total_count * 100) if total_count > 0 else 0
        
        plan_view_html = '''
        <div style="padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;">
            <div style="max-width: 800px; margin: 0 auto;">
                <!-- Header -->
                <div style="background: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                        <h2 style="margin: 0; color: #1a202c; font-size: 24px; font-weight: 600;">📋 Task Execution Plan</h2>
                        <div style="background: #667eea; color: white; padding: 6px 12px; border-radius: 20px; font-size: 14px; font-weight: 600;">
                            Step ''' + str(step) + '''
                        </div>
                    </div>
                    <div style="color: #4a5568; font-size: 14px; line-height: 1.5;">''' + task[:150] + ('...' if len(task) > 150 else '') + '''</div>
                </div>
        '''
        
        if task_plan:
            # Progress bar
            plan_view_html += f'''
                <div style="background: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 14px; font-weight: 600; color: #4a5568;">Progress</span>
                        <span style="font-size: 14px; font-weight: 600; color: #667eea;">{completed_count}/{total_count} completed</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {progress_percent}%; transition: width 0.3s ease;"></div>
                    </div>
                </div>
            '''
            
            # Current action
            if current_action:
                plan_view_html += f'''
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">▶️</span>
                        <div>
                            <div style="font-size: 12px; opacity: 0.9; margin-bottom: 4px;">CURRENT ACTION</div>
                            <div style="font-size: 15px; font-weight: 500;">{current_action}</div>
                        </div>
                    </div>
                </div>
                '''
            
            # Task list
            plan_view_html += '<div style="display: flex; flex-direction: column; gap: 12px;">'
            for i, subtask in enumerate(task_plan, 1):
                status = subtask.get('status', 'pending')
                subtask_text = subtask.get('subtask', 'Unknown')
                
                if status == 'completed':
                    icon = '✅'
                    color = '#10b981'
                    bg_gradient = 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)'
                    border_color = '#10b981'
                    status_text = 'COMPLETED'
                elif status == 'failed':
                    icon = '❌'
                    color = '#ef4444'
                    bg_gradient = 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)'
                    border_color = '#ef4444'
                    status_text = 'FAILED'
                else:  # pending
                    icon = '⏳'
                    color = '#6b7280'
                    bg_gradient = 'linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)'
                    border_color = '#d1d5db'
                    status_text = 'PENDING'
                
                plan_view_html += f'''
                <div style="background: white; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid {border_color}; transition: transform 0.2s ease, box-shadow 0.2s ease;">
                    <div style="display: flex; align-items: start; gap: 14px;">
                        <div style="background: {bg_gradient}; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 20px;">
                            {icon}
                        </div>
                        <div style="flex: 1; min-width: 0;">
                            <div style="font-weight: 600; color: #1a202c; font-size: 15px; margin-bottom: 6px; line-height: 1.4;">{i}. {subtask_text}</div>
                            <div style="display: inline-block; background: {bg_gradient}; color: {color}; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">
                                {status_text}
                            </div>
                        </div>
                    </div>
                </div>
                '''
            plan_view_html += '</div>'
        else:
            plan_view_html += '''
            <div style="background: white; border-radius: 12px; padding: 60px 24px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
                <div style="color: #9ca3af; font-size: 16px;">No task plan available</div>
            </div>
            '''
        
        plan_view_html += '</div></div>'
        
        # Store both views in global canvas updates
        with canvas_lock:
            live_canvas_updates[thread_id] = {
                'has_canvas': True,
                'canvas_type': 'html',
                'canvas_content': browser_view_html,  # Default to browser view
                'browser_view': browser_view_html,
                'plan_view': plan_view_html,
                'timestamp': time.time(),
                'url': url,
                'step': step,
                'task_plan': task_plan,
                'current_action': current_action
            }
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error updating canvas: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/canvas/toggle-view")
async def toggle_canvas_view(data: Dict[str, Any] = Body(...)):
    """
    Toggle between browser view and plan view for a thread
    """
    try:
        thread_id = data.get("thread_id")
        view_type = data.get("view_type", "browser")  # "browser" or "plan"
        
        if not thread_id:
            return {"status": "error", "message": "thread_id required"}
        
        with canvas_lock:
            if thread_id in live_canvas_updates:
                canvas_data = live_canvas_updates[thread_id]
                
                # Switch the canvas_content based on view_type
                if view_type == "plan" and 'plan_view' in canvas_data:
                    canvas_data['canvas_content'] = canvas_data['plan_view']
                    canvas_data['current_view'] = 'plan'
                elif view_type == "browser" and 'browser_view' in canvas_data:
                    canvas_data['canvas_content'] = canvas_data['browser_view']
                    canvas_data['current_view'] = 'browser'
                
                canvas_data['timestamp'] = time.time()  # Update timestamp to trigger refresh
                
                return {"status": "success", "view_type": view_type}
        
        return {"status": "error", "message": "No canvas data found for thread"}
        
    except Exception as e:
        logger.error(f"Error toggling canvas view: {e}")
        return {"status": "error", "message": str(e)}

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

@app.get("/api/chat/history/{thread_id}")
async def get_conversation_history(thread_id: str):
    """
    Load the full conversation history from the saved JSON file.
    Returns all messages, metadata, plan, and uploaded files.
    """
    try:
        # Use absolute path relative to backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
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

@app.get("/api/conversations")
async def get_all_conversations(request: Request, db: Session = Depends(get_db)):
    """
    Retrieves a list of conversations for the authenticated user.
    Returns conversation objects with metadata (id, title, created_at, last_message).
    """
    try:
        # Get authenticated user
        from auth import get_user_from_request
        user = get_user_from_request(request)
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Unable to determine user identity")
        
        logger.info(f"Fetching conversations for user: {user_id}")
        
        # Query user_threads table for this user's conversations
        from models import UserThread
        user_threads = db.query(UserThread).filter_by(user_id=user_id).order_by(
            UserThread.updated_at.desc()
        ).all()
        
        logger.info(f"Found {len(user_threads)} conversations for user {user_id}")
        
        # Build response with conversation metadata
        conversations = []
        for ut in user_threads:
            # Get last message from conversation history file if it exists
            history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{ut.thread_id}.json")
            last_message = None
            
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        history_data = json.load(f)
                        # Extract last message from messages array
                        messages = history_data.get("messages", [])
                        if messages and len(messages) > 0:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict):
                                last_message = last_msg.get("content", "")[:100]  # First 100 chars
                except Exception as e:
                    logger.warning(f"Failed to read history for {ut.thread_id}: {e}")
            
            # Handle title: check for None, empty string, or the literal string "None"
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

@app.get("/api/conversations/{thread_id}")
async def get_conversation_history(thread_id: str, request: Request, db: Session = Depends(get_db)):
    """
    Retrieves the full, standardized conversation state from its JSON file.
    This is the single source of truth for a conversation's history.
    Ensures user can only access their own conversations.
    """
    try:
        # Get authenticated user
        from auth import get_user_from_request
        user = get_user_from_request(request)
        user_id = user.get("sub") or user.get("user_id") or user.get("id")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Unable to determine user identity")
        
        # Verify ownership - check if this thread belongs to this user
        from models import UserThread
        user_thread = db.query(UserThread).filter_by(
            thread_id=thread_id,
            user_id=user_id
        ).first()
        
        if not user_thread:
            logger.warning(f"User {user_id} attempted to access thread {thread_id} they don't own")
            raise HTTPException(status_code=403, detail="You don't have permission to access this conversation")
        
        # Load the conversation history
        history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
        
        if not os.path.exists(history_path):
            raise HTTPException(status_code=404, detail="Conversation history not found.")
            
        with open(history_path, "r", encoding="utf-8") as f:
            # The file already contains the standardized, serializable state.
            # No further processing is needed.
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

# Workflow Endpoints
@app.post("/api/workflows", tags=["Workflows"])
async def save_workflow(request: Request, thread_id: str, name: str, description: str = "", db: Session = Depends(get_db)):
    """Save conversation as reusable workflow"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    # Load conversation
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    workflow_id = str(uuid.uuid4())
    
    # Extract comprehensive blueprint from conversation state
    state = history.get("state", {})
    blueprint = {
        "workflow_id": workflow_id,
        "thread_id": thread_id,
        "original_prompt": state.get("original_prompt", ""),
        "task_agent_pairs": state.get("task_agent_pairs", []),
        "task_plan": state.get("task_plan", []),
        "parsed_tasks": state.get("parsed_tasks", []),
        "candidate_agents": state.get("candidate_agents", {}),
        "user_expectations": state.get("user_expectations"),
        "completed_tasks": state.get("completed_tasks", []),
        "final_response": state.get("final_response"),
        "created_at": datetime.utcnow().isoformat()
    }
    
    workflow = Workflow(
        workflow_id=workflow_id,
        user_id=user_id,
        name=name,
        description=description or state.get("original_prompt", "")[:200],  # Use prompt as fallback description
        blueprint=blueprint
    )
    db.add(workflow)
    db.commit()
    
    logger.info(f"Workflow '{name}' ({workflow_id}) saved by user {user_id}")
    return {
        "workflow_id": workflow_id,
        "name": name,
        "description": workflow.description,
        "status": "saved",
        "task_count": len(blueprint.get("task_agent_pairs", [])),
        "created_at": blueprint["created_at"]
    }

@app.get("/api/workflows", tags=["Workflows"])
async def list_workflows(request: Request, db: Session = Depends(get_db)):
    """List user's workflows"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflows = db.query(Workflow).filter_by(user_id=user_id, status='active').all()
    return [{"workflow_id": w.workflow_id, "name": w.name, "description": w.description, "created_at": w.created_at.isoformat()} for w in workflows]

@app.get("/api/workflows/{workflow_id}", tags=["Workflows"])
async def get_workflow(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Get workflow details"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {"workflow_id": workflow.workflow_id, "name": workflow.name, "description": workflow.description, "blueprint": workflow.blueprint, "created_at": workflow.created_at.isoformat()}

@app.post("/api/workflows/{workflow_id}/execute", tags=["Workflows"])
async def execute_workflow(workflow_id: str, inputs: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    """Execute workflow with new inputs"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    new_thread_id = str(uuid.uuid4())
    
    # Create execution record
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        user_id=user_id,
        inputs=inputs,
        status='queued'
    )
    db.add(execution)
    db.commit()
    
    # Use orchestrator with modified prompt
    blueprint = workflow.blueprint
    original_prompt = blueprint.get("original_prompt", "")
    
    # Replace placeholders in prompt
    modified_prompt = original_prompt
    for key, value in inputs.items():
        modified_prompt = modified_prompt.replace(f"{{{key}}}", str(value))
    
    logger.info(f"Executing workflow {workflow_id} as thread {new_thread_id}")
    return {"execution_id": execution_id, "thread_id": new_thread_id, "status": "queued", "message": "Use /ws/chat with this thread_id to execute"}

@app.post("/api/workflows/{workflow_id}/schedule", tags=["Workflows"])
async def schedule_workflow(workflow_id: str, cron_expression: str, input_template: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    """Schedule workflow execution with cron expression"""
    from auth import get_user_from_request
    from services.workflow_scheduler import get_scheduler
    from database import SessionLocal
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    schedule_id = str(uuid.uuid4())
    schedule = WorkflowSchedule(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        user_id=user_id,
        cron_expression=cron_expression,
        input_template=input_template
    )
    db.add(schedule)
    db.commit()
    
    # Add to scheduler
    try:
        scheduler = get_scheduler()
        scheduler.add_schedule(
            schedule_id=schedule_id,
            workflow_id=workflow_id,
            cron_expression=cron_expression,
            input_template=input_template,
            user_id=user_id,
            db_session_factory=SessionLocal
        )
        logger.info(f"Scheduled workflow {workflow_id} with cron: {cron_expression}")
        return {"schedule_id": schedule_id, "status": "scheduled", "cron": cron_expression}
    except Exception as e:
        # Rollback if scheduling fails
        db.delete(schedule)
        db.commit()
        logger.error(f"Failed to schedule workflow: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid cron expression or scheduling error: {str(e)}")

@app.post("/api/workflows/{workflow_id}/webhook", tags=["Workflows"])
async def create_webhook(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Create webhook trigger for workflow"""
    from auth import get_user_from_request
    import secrets
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    webhook_id = str(uuid.uuid4())
    webhook_token = secrets.token_urlsafe(32)
    
    webhook = WorkflowWebhook(
        webhook_id=webhook_id,
        workflow_id=workflow_id,
        user_id=user_id,
        webhook_token=webhook_token
    )
    db.add(webhook)
    db.commit()
    
    logger.info(f"Created webhook {webhook_id} for workflow {workflow_id}")
    return {"webhook_id": webhook_id, "webhook_url": f"/webhooks/{webhook_id}", "webhook_token": webhook_token}

@app.delete("/api/workflows/{workflow_id}/schedule/{schedule_id}", tags=["Workflows"])
async def delete_schedule(workflow_id: str, schedule_id: str, request: Request, db: Session = Depends(get_db)):
    """Delete a workflow schedule"""
    from auth import get_user_from_request
    from services.workflow_scheduler import get_scheduler
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        user_id=user_id
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    # Remove from scheduler
    try:
        scheduler = get_scheduler()
        scheduler.remove_schedule(schedule_id)
    except Exception as e:
        logger.error(f"Failed to remove schedule from scheduler: {str(e)}")
    
    # Delete from database
    db.delete(schedule)
    db.commit()
    
    logger.info(f"Deleted schedule {schedule_id} for workflow {workflow_id}")
    return {"status": "deleted"}

@app.post("/webhooks/{webhook_id}", tags=["Webhooks"])
async def trigger_webhook(webhook_id: str, payload: Dict[str, Any], webhook_token: str = Query(...), db: Session = Depends(get_db)):
    """Trigger workflow via webhook - executes asynchronously"""
    webhook = db.query(WorkflowWebhook).filter_by(webhook_id=webhook_id, is_active=True).first()
    
    if not webhook or webhook.webhook_token != webhook_token:
        raise HTTPException(status_code=404, detail="Invalid webhook")
    
    workflow = db.query(Workflow).filter_by(workflow_id=webhook.workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=webhook.workflow_id,
        user_id=webhook.user_id,
        inputs=payload,
        status='queued',
        started_at=datetime.utcnow()
    )
    db.add(execution)
    db.commit()
    
    # Execute workflow in background
    from services.workflow_scheduler import get_scheduler
    scheduler = get_scheduler()
    asyncio.create_task(
        scheduler._async_execute_workflow(
            execution_id, workflow.workflow_id, workflow.blueprint, payload, webhook.user_id
        )
    )
    
    logger.info(f"Webhook {webhook_id} triggered workflow {webhook.workflow_id} (execution: {execution_id})")
    return {"execution_id": execution_id, "status": "running", "message": "Workflow execution started"}

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

@app.websocket("/ws/workflow/{workflow_id}/execute")
async def websocket_workflow_execute(websocket: WebSocket, workflow_id: str):
    """Execute workflow via WebSocket with streaming - uses WorkflowExecutor"""
    await websocket.accept()
    thread_id = None
    execution_id = None
    db = None
    
    try:
        from orchestrator.workflow_executor import WorkflowExecutor
        
        data = await websocket.receive_json()
        inputs = data.get("inputs", {})
        owner = data.get("owner")
        
        if not owner:
            await websocket.send_json({"node": "__error__", "error": "Owner required"})
            return
        
        # Get workflow
        db = SessionLocal()
        user_id = owner.get("user_id") or owner.get("sub") or owner
        workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
        
        if not workflow:
            await websocket.send_json({"node": "__error__", "error": "Workflow not found"})
            return
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            user_id=user_id,
            inputs=inputs,
            status='running'
        )
        db.add(execution)
        db.commit()
        
        # Use WorkflowExecutor to prepare execution
        executor = WorkflowExecutor(workflow.blueprint)
        execution_data = await executor.execute(inputs, owner)
        
        thread_id = execution_data["thread_id"]
        prompt = execution_data["prompt"]
        
        logger.info(f"Executing workflow {workflow_id} as thread {thread_id}")
        
        # Stream via orchestrator
        async for event in graph.astream_events(
            {"original_prompt": prompt, "owner": owner},
            config={"configurable": {"thread_id": thread_id}},
            version="v2"
        ):
            if event["event"] == "on_chain_stream":
                node_name = event.get("name", "unknown")
                event_data = event.get("data", {})
                
                await websocket.send_json({
                    "node": node_name,
                    "data": serialize_complex_object(event_data),
                    "thread_id": thread_id,
                    "execution_id": execution_id,
                    "workflow_id": workflow_id
                })
        
        # Update execution status
        execution.status = 'completed'
        execution.completed_at = datetime.utcnow()
        db.commit()
        
        await websocket.send_json({
            "node": "__complete__",
            "thread_id": thread_id,
            "execution_id": execution_id,
            "workflow_id": workflow_id
        })
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}", exc_info=True)
        
        if db and execution_id:
            try:
                execution = db.query(WorkflowExecution).filter_by(execution_id=execution_id).first()
                if execution:
                    execution.status = 'failed'
                    execution.error = str(e)
                    execution.completed_at = datetime.utcnow()
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update execution status: {db_error}")
        
        await websocket.send_json({"node": "__error__", "error": str(e)})
    
    finally:
        if db:
            db.close()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent orchestration updates.
    Uses the unified orchestration service with streaming enabled.
    Now supports interactive workflows with user input.
    Enhanced with comprehensive error handling and logging.
    """
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted from {websocket.client}")
    except Exception as e:
        logger.error(f"Failed to accept WebSocket connection: {e}")
        return
    
    thread_id = None  # Initialize thread_id to None
    
    try:
        while True:  # Keep the connection open for multiple messages
            # Wait for message from client
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError as je:
                # Invalid JSON from client
                logger.error(f"Invalid JSON received from WebSocket client: {je}")
                try:
                    await websocket.send_json({
                        "node": "__error__",
                        "error": "Invalid message format. Expected valid JSON.",
                        "error_type": "JSONDecodeError",
                        "timestamp": time.time()
                    })
                except Exception as send_err:
                    logger.error(f"Failed to send JSON error response: {send_err}")
                continue  # Wait for next message
            except Exception as e:
                # Connection closed or other receive error
                error_type = type(e).__name__
                # WebSocketDisconnect and RuntimeError(websocket.client_state) are expected on close
                if "closed" not in str(e).lower() and error_type != "WebSocketDisconnect":
                    logger.warning(f"WebSocket receive error ({error_type}): {e}")
                break  # Exit the loop, connection is closed

            # Get thread_id from client, or generate a new one if not provided
            thread_id = data.get("thread_id") or str(uuid.uuid4())
            prompt = data.get("prompt")
            user_response = data.get("user_response")  # For continuing conversations
            files_data = data.get("files", [])  # Get files from WebSocket message
            # Pass through owner info if frontend sends Clerk-verified identity
            owner = data.get("owner")  # Expected shape: { user_id, email }
            planning_mode = data.get("planning_mode", False)  # Get planning mode flag

            logger.info(f"WebSocket received message with thread_id: {thread_id}, planning_mode: {planning_mode}")
            logger.info(f"Message details: has_prompt={bool(prompt)}, has_user_response={bool(user_response)}, prompt_value='{prompt[:50] if prompt else None}', user_response_value='{user_response[:50] if user_response else None}'")
            logger.info(f"Owner info: owner={owner}, type={type(owner)}")

            # Determine if this is a new thread
            is_new_thread = "thread_id" not in data or not data.get("thread_id")
            
            # For new threads, owner is required
            if is_new_thread and not owner:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Owner information is required for new conversations",
                    "error_type": "ValidationError",
                    "thread_id": thread_id,
                    "timestamp": time.time()
                })
                continue  # Continue waiting for messages
            
            # For existing threads, if owner not provided, try to get from database
            if not is_new_thread and not owner:
                try:
                    from database import SessionLocal
                    from models import UserThread
                    
                    db = SessionLocal()
                    try:
                        user_thread = db.query(UserThread).filter_by(thread_id=thread_id).first()
                        if user_thread:
                            owner = user_thread.user_id
                            logger.info(f"Retrieved owner from database for existing thread {thread_id}: {owner}")
                        else:
                            logger.warning(f"No user-thread relationship found for thread {thread_id}")
                    finally:
                        db.close()
                except Exception as db_err:
                    logger.error(f"Failed to retrieve owner for thread {thread_id}: {db_err}")
                    # Continue anyway - proceed without owner for existing thread

            if not prompt and not user_response:
                await websocket.send_json({
                    "node": "__error__",
                    "error": "Missing 'prompt' field for new conversation or 'user_response' for continuing",
                    "error_type": "ValidationError",
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
                    
                    # Emit task status events for real-time UI updates
                    if node_name == "execute_batch" and isinstance(node_output, dict):
                        task_events = node_output.get("task_events", [])
                        if task_events:
                            logger.info(f"🔄 Emitting {len(task_events)} task status events")
                            for event in task_events:
                                event_type = event.get("event_type")
                                task_name = event.get("task_name")
                                
                                if event_type == "task_started":
                                    await websocket.send_json({
                                        "node": "task_started",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "task_description": event.get("task_description"),
                                        "agent_name": event.get("agent_name"),
                                        "timestamp": event.get("timestamp", time.time())
                                    })
                                    logger.debug(f"🚀 Emitted task_started for '{task_name}'")
                                    
                                elif event_type == "task_completed":
                                    await websocket.send_json({
                                        "node": "task_completed",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "agent_name": event.get("agent_name"),
                                        "execution_time": event.get("execution_time", 0),
                                        "timestamp": event.get("timestamp", time.time())
                                    })
                                    logger.debug(f"✅ Emitted task_completed for '{task_name}'")
                                    
                                elif event_type == "task_failed":
                                    await websocket.send_json({
                                        "node": "task_failed",
                                        "thread_id": thread_id,
                                        "task_name": task_name,
                                        "error": event.get("error"),
                                        "execution_time": event.get("execution_time", 0),
                                        "timestamp": event.get("timestamp", time.time())
                                    })
                except Exception as e:
                    logger.error(f"Error in stream_callback: {e}", exc_info=True)
            
            # Define task event callback for REAL-TIME task status streaming
            async def task_event_callback(event: dict):
                """Stream task events in real-time as tasks start/complete"""
                try:
                    event_type = event.get("event_type")
                    task_name = event.get("task_name")
                    
                    if event_type == "task_started":
                        await websocket.send_json({
                            "node": "task_started",
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "task_description": event.get("task_description"),
                            "agent_name": event.get("agent_name"),
                            "timestamp": event.get("timestamp", time.time())
                        })
                        logger.info(f"📡 REAL-TIME: Task started - '{task_name}'")
                        
                    elif event_type == "task_completed":
                        await websocket.send_json({
                            "node": "task_completed",
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "agent_name": event.get("agent_name"),
                            "execution_time": event.get("execution_time", 0),
                            "timestamp": event.get("timestamp", time.time())
                        })
                        logger.info(f"📡 REAL-TIME: Task completed - '{task_name}' ({event.get('execution_time', 0):.2f}s)")
                        
                    elif event_type == "task_failed":
                        await websocket.send_json({
                            "node": "task_failed",
                            "thread_id": thread_id,
                            "task_name": task_name,
                            "error": event.get("error"),
                            "execution_time": event.get("execution_time", 0),
                            "timestamp": event.get("timestamp", time.time())
                        })
                        logger.warning(f"📡 REAL-TIME: Task failed - '{task_name}': {event.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Error in task_event_callback: {e}", exc_info=True)

            # Convert files data to FileObject instances with enhanced error handling
            file_objects = []
            if files_data:
                logger.info(f"Processing {len(files_data)} files from WebSocket message")
                for idx, file_data in enumerate(files_data):
                    try:
                        if isinstance(file_data, dict) and 'file_name' in file_data:
                            # Validate required fields
                            required_fields = ['file_name', 'file_path', 'file_type']
                            missing_fields = [f for f in required_fields if f not in file_data]
                            if missing_fields:
                                logger.warning(f"File {idx} missing fields: {missing_fields}. Skipping.")
                                continue
                            
                            file_objects.append(FileObject(
                                file_name=file_data['file_name'],
                                file_path=file_data['file_path'],
                                file_type=file_data['file_type']
                            ))
                            logger.info(f"Added file {idx}: {file_data['file_name']} at {file_data['file_path']}")
                        else:
                            logger.warning(f"File {idx} not a dict or missing 'file_name' field. Type: {type(file_data)}")
                    except Exception as file_err:
                        logger.error(f"Failed to create FileObject from file {idx} ({file_data.get('file_name', 'unknown')}): {file_err}")
                        # Continue processing other files
            
            # Start background task to poll for live canvas updates
            polling_active = True
            async def poll_live_canvas():
                last_update_time = 0
                while polling_active:
                    await asyncio.sleep(0.5)  # Poll every 500ms
                    try:
                        # Check if there's a live canvas update from browser agent
                        with canvas_lock:
                            if thread_id in live_canvas_updates:
                                live_update = live_canvas_updates[thread_id]
                                if live_update.get('timestamp', 0) > last_update_time:
                                    # New canvas update available
                                    await websocket.send_json({
                                        "node": "__live_canvas__",
                                        "thread_id": thread_id,
                                        "data": {
                                            "has_canvas": live_update.get('has_canvas', True),
                                            "canvas_type": live_update.get('canvas_type', 'html'),
                                            "canvas_content": live_update.get('canvas_content', ''),
                                            "browser_view": live_update.get('browser_view', ''),
                                            "plan_view": live_update.get('plan_view', ''),
                                            "current_view": live_update.get('current_view', 'browser'),
                                            "screenshot_count": live_update.get('step', 0)
                                        },
                                        "timestamp": time.time()
                                    })
                                    last_update_time = live_update['timestamp']
                                    logger.info(f"📡 Sent live canvas update for thread {thread_id} (step {live_update.get('step', 0)})")
                    except Exception as e:
                        logger.debug(f"Error polling live canvas: {e}")
            
            # Start polling task
            polling_task = asyncio.create_task(poll_live_canvas())
            
            try:
                # Use unified orchestration service with streaming and enhanced error handling
                try:
                    final_state = await execute_orchestration(
                        prompt=prompt,
                        thread_id=thread_id,
                        user_response=user_response,
                        files=file_objects if file_objects else None,
                        stream_callback=stream_callback,
                        task_event_callback=task_event_callback,
                        planning_mode=planning_mode
                    )
                except asyncio.TimeoutError as timeout_err:
                    logger.error(f"Orchestration execution timed out for thread_id {thread_id}")
                    await websocket.send_json({
                        "node": "__error__",
                        "thread_id": thread_id,
                        "error": "Request processing timed out. Please try with a simpler request.",
                        "error_type": "TimeoutError",
                        "message": "An error occurred during orchestration: Request processing timed out",
                        "status": "error",
                        "timestamp": time.time()
                    })
                    continue
                except Exception as orch_err:
                    logger.error(f"Error executing orchestration for thread_id {thread_id}: {orch_err}", exc_info=True)
                    error_type = type(orch_err).__name__
                    user_friendly_message = "An unexpected error occurred during orchestration"
                    if "permission" in str(orch_err).lower():
                        user_friendly_message = "Permission denied. Please check your authentication."
                    elif "not found" in str(orch_err).lower():
                        user_friendly_message = "Required resource not found."
                    elif "invalid" in str(orch_err).lower():
                        user_friendly_message = "Invalid request parameters."
                    
                    await websocket.send_json({
                        "node": "__error__",
                        "thread_id": thread_id,
                        "error": str(orch_err)[:200],  # Limit error message length
                        "error_type": error_type,
                        "message": f"An error occurred during orchestration: {user_friendly_message}",
                        "status": "error",
                        "timestamp": time.time()
                    })
                    continue
            finally:
                # Stop polling - always cleanup regardless of success/failure
                polling_active = False
                try:
                    await polling_task
                except:
                    pass

            # Check if workflow is paused for user input
            if final_state.get("pending_user_input"):
                # Check if this is an approval request
                needs_approval = final_state.get("needs_approval", False)
                
                # Calculate cost if approval is needed
                estimated_cost = 0.0
                task_count = 0
                if needs_approval:
                    task_plan = final_state.get("task_plan", [])
                    for batch in task_plan:
                        for task_dict in batch:
                            task_count += 1
                            if isinstance(task_dict, dict):
                                primary_agent = task_dict.get('primary', {})
                                cost = primary_agent.get('price_per_call_usd', 0.0)
                                if cost:
                                    estimated_cost += cost
                
                await websocket.send_json({
                    "node": "__user_input_required__",
                    "thread_id": thread_id,
                    "data": {
                        "question_for_user": final_state.get("question_for_user"),
                        "approval_required": needs_approval,
                        "estimated_cost": estimated_cost,
                        "task_count": task_count,
                        "task_plan": final_state.get("task_plan", []),
                        "task_agent_pairs": final_state.get("task_agent_pairs", [])
                    },
                    "message": "Additional information required to complete your request.",
                    "status": "pending_user_input",
                    "timestamp": time.time()
                })
                logger.info(f"WebSocket workflow paused for user input in thread_id {thread_id}, needs_approval: {needs_approval}")
                continue  # Continue waiting for user response message

            # Save conversation history with owner enforcement and error handling
            owner_id = None
            try:
                if owner:
                    if isinstance(owner, str):
                        owner_id = owner
                    else:
                        owner_id = owner.get("user_id") or owner.get("sub") or owner.get("id")
                if not owner_id:
                    logger.error(f"Missing owner_id for thread {thread_id}. Conversation will NOT be saved.")
                    # Don't raise error - allow workflow to continue, just log the issue
                else:
                    from orchestrator.graph import save_conversation_history
                    
                    owner_obj = {"user_id": owner_id}
                    final_state["owner"] = owner_obj
                    save_conversation_history(final_state, {"configurable": {"thread_id": thread_id, "owner": owner_obj}})
                    logger.info(f"Conversation history saved for thread_id {thread_id} with owner_id {owner_id}")
            except Exception as save_err:
                logger.error(f"Failed to save conversation history for thread {thread_id}: {save_err}")
                # Continue anyway - don't fail the request if history saving fails
            
            # Get serializable state with error handling
            try:
                from orchestrator.graph import get_serializable_state
                serializable_state = get_serializable_state(final_state, thread_id)
            except Exception as serialize_err:
                logger.warning(f"Failed to serialize complete state for thread {thread_id}: {serialize_err}")
                # Fallback: send a minimal but valid response
                serializable_state = {
                    "status": "completed",
                    "thread_id": thread_id,
                    "message": "Orchestration completed successfully",
                    "warning": f"Some data could not be processed: {str(serialize_err)[:100]}"
                }

            await websocket.send_json({
                "node": "__end__",
                "thread_id": thread_id,
                "data": serializable_state, # Send the entire state object
                "message": "Agent orchestration completed successfully.",
                "status": "completed",
                "timestamp": time.time()
            })

            logger.info(f"WebSocket stream completed successfully for thread_id {thread_id}")
            
            # Keep the connection open for multi-turn conversations
            # The frontend will continue to send messages or close the connection when done

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for thread_id {thread_id}")
    except Exception as e:
        # Enhanced error handling with categorization and logging
        error_type = type(e).__name__
        error_message = str(e)
        
        # Categorize error for better logging
        error_category = "unknown"
        if "database" in error_message.lower() or "db" in error_message.lower():
            error_category = "database"
        elif "permission" in error_message.lower() or "unauthorized" in error_message.lower():
            error_category = "authorization"
        elif "timeout" in error_message.lower():
            error_category = "timeout"
        elif "resource" in error_message.lower() or "not found" in error_message.lower():
            error_category = "resource_not_found"
        elif "invalid" in error_message.lower():
            error_category = "validation"
        
        logger.error(f"WebSocket error for thread_id {thread_id} [Category: {error_category}]: {error_type} - {error_message}", exc_info=True)
        
        # Prepare user-friendly error message
        user_message = "An error occurred during orchestration"
        if error_category == "database":
            user_message = "Database connection error. Please try again later."
        elif error_category == "authorization":
            user_message = "You do not have permission to perform this action."
        elif error_category == "timeout":
            user_message = "Request took too long. Please try with a simpler request."
        elif error_category == "resource_not_found":
            user_message = "A required resource was not found."
        
        # Truncate error details for security
        error_details = error_message[:150] if len(error_message) > 150 else error_message
        
        try:
            await websocket.send_json({
                "node": "__error__",
                "thread_id": thread_id or "unknown",
                "error": error_details,
                "error_type": error_type,
                "error_category": error_category,
                "message": user_message,
                "status": "error",
                "timestamp": time.time()
            })
            logger.info(f"Error response sent to client for thread_id {thread_id}")
        except Exception as send_err:
            # If we can't send the error, the connection is likely already closed
            logger.error(f"Could not send error message to WebSocket for thread_id {thread_id}: {send_err}")

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
    Falls back to text search if vector search fails.
    """
    if not capabilities:
        return []

    try:
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
    
    except Exception as e:
        logger.warning(f"Vector search failed, falling back to text search: {e}")
        # Fallback: text-based search on capabilities
        query = db.query(Agent).options(
            joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
        ).filter(Agent.status == 'active')
        
        # Apply optional filters
        if max_price is not None:
            query = query.filter(Agent.price_per_call_usd <= max_price)
        if min_rating is not None:
            query = query.filter(Agent.rating >= min_rating)
        
        # Return all active agents as fallback
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

@app.get("/api/agent-servers/status")
async def get_agent_servers_status():
    """Get the status of all agent servers"""
    async with agent_status_lock:
        status_copy = {
            name: {
                'port': info['port'],
                'status': info['status'],
                'pid': info['process'].pid if info['process'] else None
            }
            for name, info in agent_status.items()
        }
    return status_copy

# Global list to track agent processes and their status
agent_processes = []
agent_status = {}  # {agent_name: {'port': int, 'process': subprocess.Popen, 'status': 'starting'|'ready'|'failed'}}
agent_status_lock = asyncio.Lock()

def cleanup_agents():
    """Stop all agent processes"""
    global agent_processes
    for process in agent_processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass
    agent_processes = []
    agent_status.clear()

async def wait_for_agent_ready(agent_name: str, port: int, timeout: float = 30.0) -> bool:
    """
    Wait for a specific agent to be ready by checking its health endpoint.
    Returns True if agent is ready, False if timeout or failed.
    """
    import httpx
    import time
    
    start_time = time.time()
    health_url = f"http://localhost:{port}/"
    
    while time.time() - start_time < timeout:
        async with agent_status_lock:
            status = agent_status.get(agent_name, {}).get('status')
            if status == 'ready':
                return True
            elif status == 'failed':
                return False
        
        # Try to connect to the agent
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(health_url)
                if response.status_code == 200:
                    async with agent_status_lock:
                        if agent_name in agent_status:
                            agent_status[agent_name]['status'] = 'ready'
                    return True
        except:
            pass
        
        await asyncio.sleep(0.5)
    
    async with agent_status_lock:
        if agent_name in agent_status:
            agent_status[agent_name]['status'] = 'failed'
    return False

async def check_agent_health_background():
    """Background task to check agent health and update status"""
    import httpx
    
    while True:
        await asyncio.sleep(2)  # Check every 2 seconds
        
        async with agent_status_lock:
            agents_to_check = list(agent_status.items())
        
        for agent_name, info in agents_to_check:
            if info['status'] == 'starting':
                port = info['port']
                health_url = f"http://localhost:{port}/"
                
                try:
                    async with httpx.AsyncClient(timeout=1.0) as client:
                        response = await client.get(health_url)
                        if response.status_code == 200:
                            async with agent_status_lock:
                                agent_status[agent_name]['status'] = 'ready'
                except:
                    pass

def start_agents_async():
    """Start agents asynchronously without blocking main.py startup"""
    global agent_processes, agent_status
    
    # Clean up any existing agents first
    cleanup_agents()
    
    # Use absolute path based on the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(project_root, "agents")
    
    if not os.path.isdir(agents_dir):
        logger.warning(f"'{agents_dir}' directory not found. Skipping agent server startup.")
        return

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    agent_files = [f for f in os.listdir(agents_dir) if f.endswith("_agent.py")]
    logger.info(f"Starting {len(agent_files)} agent server(s)...")

    for agent_file in agent_files:
        agent_path = os.path.join(agents_dir, agent_file)
        agent_name = agent_file.replace('.py', '')
        port = None
        
        try:
            # Extract port from agent file
            with open(agent_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'port\s*=\s*int\(os\.getenv\([^,]+,\s*(\d+)\)', content)
                if not match:
                    match = re.search(r"port\s*=\s*(\d+)", content)
                if match:
                    port = int(match.group(1))

            if port is None:
                continue

            log_path = os.path.join(logs_dir, f"{agent_file}.log")

            # Start the agent process
            with open(log_path, 'w') as log_file:
                if platform.system() == "Windows":
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    process = subprocess.Popen(
                        [sys.executable, agent_path],
                        stdout=log_file,
                        stderr=log_file,
                        start_new_session=True
                    )
                
                agent_processes.append(process)
                agent_status[agent_name] = {
                    'port': port,
                    'process': process,
                    'status': 'starting'
                }
        
        except Exception as e:
            agent_status[agent_name] = {
                'port': port,
                'process': None,
                'status': 'failed'
            }
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup_agents)
    
    logger.info(f"Agent servers started. Ready for requests.")

@app.on_event("startup")
async def startup_event():
    """Start agents, background health checker, and workflow scheduler on app startup"""
    # Start agents in background
    start_agents_async()
    
    # Start background health checker
    asyncio.create_task(check_agent_health_background())
    
    # Initialize workflow scheduler and load active schedules
    try:
        from services.workflow_scheduler import init_scheduler
        from database import SessionLocal
        db = SessionLocal()
        try:
            init_scheduler(db)
            logger.info("Workflow scheduler initialized successfully")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to initialize workflow scheduler: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Agents will be started automatically via @app.on_event("startup")
    # Run the main FastAPI app
    import uvicorn
    # Use 0.0.0.0 to bind to all interfaces (fixes IPv4/IPv6 issues)
    # Add ws_ping_interval and ws_ping_timeout for better WebSocket stability
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",  # Changed from 127.0.0.1 to support both IPv4 and IPv6
        port=8000, 
        reload=True,
        reload_includes=["*.py"],  # Watch all Python files including agents
        ws_ping_interval=20,  # Send ping every 20 seconds
        ws_ping_timeout=20,   # Wait 20 seconds for pong
        log_level="info"
    )

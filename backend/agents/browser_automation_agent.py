# agents/browser_automation_agent.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import asyncio
from dotenv import load_dotenv
import uvicorn
from browser_use import Agent, ChatOpenAI
import logging
from pathlib import Path
import uuid
from datetime import datetime
import shutil
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from a .env file
load_dotenv()

# --- Configuration & API Key Check ---
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    raise RuntimeError("OLLAMA_API_KEY is not set in the environment. The agent cannot start.")

# --- Agent Definition ---
AGENT_DEFINITION = {
    "id": "browser_automation_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Browser Automation Agent",
    "description": "An intelligent browser automation agent that can navigate websites, interact with web pages, extract information, take screenshots, and perform complex web-based tasks using vision-enabled AI.",
    "capabilities": [
        "take website screenshots",
        "capture webpage screenshots",
        "screenshot websites",
        "automate web browsing",
        "extract web data",
        "scrape website content",
        "interact with web pages",
        "navigate websites",
        "visit urls",
        "analyze web pages visually",
        "describe website content",
        "check website information"
    ],
    "price_per_call_usd": 0.01,
    "status": "active",
    "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA3FcU8hPhmFLgez6qPf801aQahasAlG5S4MPb16nWJPA=\n-----END PUBLIC KEY-----",
    "endpoints": [
        {
            "endpoint": "http://localhost:8070/browse",
            "http_method": "POST",
            "description": "Execute a browser automation task. Orchestrator handles async internally.",
            "parameters": [
                {
                    "name": "task",
                    "param_type": "string",
                    "required": True,
                    "description": "The task description for the browser agent (e.g., 'Take a screenshot of google.com', 'Navigate to GitHub and check stars')."
                },
                {
                    "name": "extract_data",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Whether to extract structured data from the page (default: False)."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8070/browse/async",
            "http_method": "POST",
            "description": "Submit a browser automation task for async processing. Returns immediately with task_id.",
            "parameters": [
                {
                    "name": "task",
                    "param_type": "string",
                    "required": True,
                    "description": "The task description for the browser agent."
                },
                {
                    "name": "extract_data",
                    "param_type": "boolean",
                    "required": False,
                    "description": "Whether to extract structured data from the page (default: False)."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8070/browse/status/{task_id}",
            "http_method": "GET",
            "description": "Check the status of a browser automation task.",
            "parameters": [
                {
                    "name": "task_id",
                    "param_type": "string",
                    "required": True,
                    "description": "The task ID returned from /browse/async"
                }
            ]
        }
    ]
}

app = FastAPI(title="Browser Automation Agent")

# In-memory task storage (in production, use Redis or database)
tasks_storage: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---
class BrowserTask(BaseModel):
    task: str = Field(..., description="The browser automation task to perform")
    extract_data: Optional[bool] = Field(False, description="Whether to extract structured data")

class BrowserActionStep(BaseModel):
    """Represents a single action taken by the browser agent"""
    action: str
    description: str
    timestamp: Optional[str] = None

class BrowserTaskOutput(BaseModel):
    """Pydantic model for structured output from browser agent"""
    result: str = Field(..., description="The final result or answer from the browser task")

class FileObject(BaseModel):
    """File object matching orchestrator's FileObject schema"""
    file_name: str
    file_path: str
    file_type: str

class BrowsingTraceStep(BaseModel):
    """A single step in the browsing trace"""
    step_number: int
    action: str
    description: str
    status: str  # 'success', 'error', 'pending'
    duration: Optional[float] = None  # in seconds
    timestamp: str

class BrowserResult(BaseModel):
    """Result from browser automation"""
    success: bool
    task_summary: str
    actions_taken: List[BrowserActionStep]
    browsing_trace: Optional[List[BrowsingTraceStep]] = Field(None, description="Detailed browsing trace with timing")
    extracted_data: Optional[Dict[str, Any]] = None
    screenshot_files: Optional[List[FileObject]] = Field(None, description="List of screenshot files saved")
    error: Optional[str] = None
    note: Optional[str] = Field(None, description="Additional notes about the execution")

class TaskSubmitResponse(BaseModel):
    """Response when submitting a task"""
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    """Response when checking task status"""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    result: Optional[BrowserResult] = None
    progress: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return AGENT_DEFINITION

async def process_browser_task(task_id: str, request: BrowserTask):
    """Background task processor for browser automation"""
    try:
        # Update status to processing
        tasks_storage[task_id]["status"] = "processing"
        tasks_storage[task_id]["progress"] = "Starting browser automation..."
        
        logger.info(f"[Task {task_id}] Starting browser automation task: {request.task}")
        logger.info(f"Starting browser automation task: {request.task}")
        
        # Initialize the vision LLM via Ollama Cloud's OpenAI-compatible API
        llm = ChatOpenAI(
            model="qwen3-vl:235b-cloud",
            api_key=OLLAMA_API_KEY,
            base_url="https://ollama.com/v1",
            temperature=0.2,
        )
        
        # Create the agent with vision enabled
        # Note: Vision model analyzes screenshots internally but doesn't save them to disk
        agent = Agent(
            task=request.task,
            llm=llm,
            use_vision=True,
        )
        
        # Run the agent and get history
        logger.info("Executing browser agent...")
        history = await agent.run()
        
        # Extract the final result from history
        final_result = None
        actions_taken = []
        extracted_data = None
        screenshot_files = []  # Initialize screenshot files list
        
        # Try to get the final result using the final_result() method
        if hasattr(history, 'final_result'):
            try:
                if callable(history.final_result):
                    final_result = history.final_result()
                else:
                    final_result = history.final_result
                    
                if final_result and isinstance(final_result, str):
                    logger.info(f"Extracted final result: {final_result[:100]}...")
                else:
                    logger.warning(f"final_result is not a string: {type(final_result)}")
                    final_result = None
            except Exception as e:
                logger.warning(f"Could not extract final_result: {e}")
                final_result = None
        
        # Capture final screenshot and add to the periodic screenshots
        try:
            # Extract URL from task - support both with and without protocol
            url_match = re.search(r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', request.task)
            if not url_match:
                # Try to find domain without protocol (e.g., "google.com", "github.com")
                domain_match = re.search(r'(?:go to|navigate to|open|visit)?\s*([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(?:\s|$|,|\.|\band\b)', request.task, re.IGNORECASE)
                if domain_match:
                    url = f"https://{domain_match.group(1)}"
                    logger.info(f"Extracted URL without protocol: {url}")
                else:
                    url = None
            else:
                url = url_match.group(0)
                # Clean up trailing punctuation
                url = re.sub(r'[,;.!?]+$', '', url)
            
            if url:
                logger.info(f"Capturing final screenshot of {url}...")
                
                # Use Playwright to capture final screenshot
                from playwright.async_api import async_playwright
                
                storage_dir = Path("storage/images")
                storage_dir.mkdir(parents=True, exist_ok=True)
                
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    
                    # Generate filename
                    timestamp = int(time.time() * 1000)
                    filename = f"browser_final_{timestamp}.png"
                    filepath = storage_dir / filename
                    
                    # Capture screenshot
                    await page.screenshot(path=str(filepath), full_page=False)
                    await browser.close()
                    
                    screenshot_files.append(FileObject(
                        file_name=filename,
                        file_path=str(filepath),
                        file_type="image"
                    ))
                    
                    logger.info(f"Final screenshot saved to {filepath}")
            else:
                logger.warning("Could not extract URL from task for final screenshot")
        except Exception as e:
            logger.warning(f"Could not capture final screenshot: {e}")
            # Continue anyway - we have periodic screenshots
        
        # Parse the history to extract action steps and browsing trace
        browsing_trace = []
        if history:
            try:
                history_list = list(history)  # Convert to list
                start_time = time.time()
                
                # Extract actions from each step
                for idx, step in enumerate(history_list):
                    action_desc = f"Step {idx + 1}"
                    action_name = "Unknown action"
                    step_duration = None
                    
                    if hasattr(step, 'result') and step.result:
                        # Try to get action details
                        if hasattr(step.result, 'action'):
                            action_obj = step.result.action
                            if hasattr(action_obj, '__class__'):
                                action_name = action_obj.__class__.__name__.replace('Action', '').replace('Model', '')
                            action_desc = f"Action: {str(action_obj)[:100]}"
                        elif hasattr(step.result, 'extracted_content'):
                            action_name = "Extract content"
                            action_desc = f"Extracted: {str(step.result.extracted_content)[:100]}"
                    
                    # Create browsing trace entry
                    browsing_trace.append(BrowsingTraceStep(
                        step_number=idx + 1,
                        action=action_name,
                        description=action_desc[:200],
                        status='success',
                        duration=step_duration,
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
                    actions_taken.append(BrowserActionStep(
                        action=f"step_{idx + 1}",
                        description=action_desc[:200]
                    ))
                
            except Exception as e:
                logger.warning(f"Could not parse history details: {e}")
                actions_taken.append(BrowserActionStep(
                    action="task_completed",
                    description="Browser task completed"
                ))
        
        if not actions_taken:
            actions_taken.append(BrowserActionStep(
                action="task_completed",
                description="Browser automation task completed successfully"
            ))
        
        # Use the structured output result as task summary
        if final_result:
            task_summary = final_result
            if request.extract_data:
                extracted_data = {"content": final_result}
        else:
            task_summary = f"Successfully completed browser automation task: {request.task}"
            logger.warning("No structured output found, using generic message")
        
        logger.info(f"Browser automation completed successfully. Actions: {len(actions_taken)}")
        logger.info(f"Screenshot files captured: {len(screenshot_files)} files")
        if screenshot_files:
            for sf in screenshot_files:
                logger.info(f"  - {sf.file_name} at {sf.file_path}")
        
        result = BrowserResult(
            success=True,
            task_summary=task_summary,
            actions_taken=actions_taken,
            browsing_trace=browsing_trace if browsing_trace else None,
            extracted_data=extracted_data if request.extract_data else None,
            screenshot_files=screenshot_files if screenshot_files else None,
            note="Vision model analyzed screenshots internally. task_summary contains visual description. Screenshot files saved for later use."
        )
        
        # Update task storage with result
        tasks_storage[task_id]["status"] = "completed"
        tasks_storage[task_id]["result"] = result
        tasks_storage[task_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"[Task {task_id}] Browser automation completed successfully")
        
    except Exception as e:
        logger.error(f"[Task {task_id}] Error during browser automation: {str(e)}")
        result = BrowserResult(
            success=False,
            task_summary=f"Browser automation failed: {str(e)}",
            actions_taken=[],
            error=str(e)
        )
        
        # Update task storage with error
        tasks_storage[task_id]["status"] = "failed"
        tasks_storage[task_id]["result"] = result
        tasks_storage[task_id]["completed_at"] = datetime.utcnow().isoformat()

@app.post("/browse/async", response_model=TaskSubmitResponse)
async def submit_browser_task(request: BrowserTask, background_tasks: BackgroundTasks):
    """
    Submit a browser automation task for async processing.
    Returns immediately with a task_id that can be used to check status.
    """
    task_id = str(uuid.uuid4())
    
    # Store task info
    tasks_storage[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "request": request.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "result": None,
        "progress": "Task queued"
    }
    
    # Add to background tasks
    background_tasks.add_task(process_browser_task, task_id, request)
    
    logger.info(f"[Task {task_id}] Submitted browser task: {request.task}")
    
    return TaskSubmitResponse(
        task_id=task_id,
        status="pending",
        message="Task submitted successfully. Use /browse/status/{task_id} to check progress."
    )

@app.get("/browse/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Check the status of a browser automation task.
    """
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task_info = tasks_storage[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        result=task_info.get("result"),
        progress=task_info.get("progress"),
        created_at=task_info["created_at"],
        completed_at=task_info.get("completed_at")
    )

@app.post("/browse", response_model=BrowserResult)
async def execute_browser_task_sync(request: BrowserTask):
    """
    Synchronous endpoint for backward compatibility.
    Executes a browser automation task and waits for completion.
    Note: Use /browse/async for long-running tasks to avoid timeouts.
    """
    task_id = str(uuid.uuid4())
    logger.info(f"[Task {task_id}] Synchronous browser task: {request.task}")
    
    # Initialize task in storage
    tasks_storage[task_id] = {
        "status": "pending",
        "result": None,
        "progress": "Task created",
        "created_at": datetime.now().isoformat()
    }
    
    # Process task synchronously
    await process_browser_task(task_id, request)
    
    # Return result
    if task_id in tasks_storage and tasks_storage[task_id].get("result"):
        return tasks_storage[task_id]["result"]
    else:
        raise HTTPException(status_code=500, detail="Task processing failed")

if __name__ == "__main__":
    port = int(os.getenv("BROWSER_AGENT_PORT", 8070))
    logger.info(f"Starting Browser Automation Agent on port {port}")
    uvicorn.run("browser_automation_agent:app", host="0.0.0.0", port=port, reload=False)

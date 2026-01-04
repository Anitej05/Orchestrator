# agents/browser_automation_agent.py
# SOTA WRAPPER - Delegates to agents/browser_agent/agent.py

# CRITICAL: Fix Windows asyncio subprocess issue with Playwright
# Must be set BEFORE any other asyncio imports



# CRITICAL: Fix Windows asyncio subprocess issue with Playwright
# Must be set BEFORE any other asyncio imports

import sys
import os
import logging

from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add parent directory to path for imports when running as standalone
CURRENT_DIR = Path(__file__).parent
BACKEND_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Import standardized file manager 
try:
    from agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    try:
        from utils.agent_file_manager import AgentFileManager, FileType, FileStatus
    except ImportError:
        logger.error("Failed to import agent_file_manager from any location")
        raise

# Import New SOTA Agent
from agents.browser_agent.agent import BrowserAgent as SotaBrowserAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Configurations ---
AGENT_DEFINITION = {
    "id": "custom_browser_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Custom Browser Automation Agent",
    "description": "A powerful custom browser automation agent with full control over web interactions",
    "capabilities": [
        "web browsing", "data extraction", "form filling", "screenshot capture",
        "web scraping", "page navigation", "element interaction"
    ],
    "price_per_call_usd": 0.01,
    "status": "active",
    "endpoints": [{
        "endpoint": "http://localhost:8090/browse",
        "http_method": "POST",
        "description": "Execute browser automation task",
        "parameters": [{
            "name": "task",
            "param_type": "string",
            "required": True,
            "description": "Task description"
        }]
    }]
}

app = FastAPI(title="Custom Browser Automation Agent")

# Storage Setup (kept for compatibility)
STORAGE_DIR = Path("storage/browser_agent/screenshots")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path("storage/browser_agent/downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path("storage/browser_agent/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

screenshot_file_manager = AgentFileManager(
    agent_id="browser_agent_screenshots",
    storage_dir=str(STORAGE_DIR),
    default_ttl_hours=24,
    auto_cleanup=True,
    cleanup_interval_hours=6
)

download_file_manager = AgentFileManager(
    agent_id="browser_agent_downloads",
    storage_dir=str(DOWNLOADS_DIR),
    default_ttl_hours=72,
    auto_cleanup=True,
    cleanup_interval_hours=12
)

# Request/Response Models
class BrowseRequest(BaseModel):
    task: str
    extract_data: Optional[bool] = False
    max_steps: Optional[int] = 10

class BrowseResponse(BaseModel):
    success: bool
    task_summary: str
    actions_taken: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]] = None
    screenshot_files: Optional[List[str]] = None
    downloaded_files: Optional[List[str]] = None
    error: Optional[str] = None

class FileListResponse(BaseModel):
    success: bool
    files: List[Dict[str, Any]]
    count: int
    error: Optional[str] = None

class FileStatsResponse(BaseModel):
    success: bool
    screenshots: Dict[str, Any]
    downloads: Dict[str, Any]
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "healthy", "agent": "custom_browser_agent", "message": "SOTA Browser Automation Agent is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/browse")
async def browse(request: BrowseRequest, headless: bool = False, enable_streaming: bool = True, thread_id: Optional[str] = None):
    """Execute browser automation task via SOTA Agent"""
    logger.info(f"📥 Received task: {request.task} | Streaming: {enable_streaming}")
    
    try:
        # Initialize SOTA Agent
        # Uses default backend_url=http://localhost:8000. Update if configurable.
        # Note: max_steps is accepted in request but not used by current BrowserAgent
        agent = SotaBrowserAgent(
            task=request.task, 
            headless=headless,
            thread_id=thread_id if enable_streaming else None
        )
        
        # Run SOTA Agent
        result = await agent.run()
        
        # Extract downloads and screenshots from result data if present
        downloads = []
        screenshots = []
        if result.success:
            # Check history for download/screenshot items
            for item in getattr(result, 'extracted_data', {}).values():
                 if isinstance(item, dict):
                    if 'downloaded_files' in item:
                        downloads.extend(item['downloaded_files'])
                    if 'screenshot_path' in item:
                        screenshots.append(item['screenshot_path'])

            # Also check raw data in actions_taken for direct action results
            for action in result.actions_taken:
                if action.get('action') == 'save_screenshot' and action.get('data'):
                    path = action['data'].get('screenshot_path')
                    if path:
                        screenshots.append(path)
        
        return BrowseResponse(
            success=result.success,
            task_summary=result.task_summary,
            actions_taken=result.actions_taken or [], # Ensure list even if None
            extracted_data=result.extracted_data,
            screenshot_files=screenshots, 
            downloaded_files=downloads,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"❌ Error in SOTA wrapper: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "task_summary": f"Error: {str(e)}",
            "actions_taken": [],
            "screenshot_files": None,
            "extracted_data": None,
            "error": str(e)
        }

# File Management Endpoints (Preserved)
@app.get("/files/screenshots", response_model=FileListResponse)
async def list_screenshots(status: Optional[str] = None, thread_id: Optional[str] = None):
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = screenshot_file_manager.list_files(status=file_status, thread_id=thread_id)
        return FileListResponse(success=True, files=[f.to_orchestrator_format() for f in files], count=len(files))
    except Exception as e:
        return FileListResponse(success=False, files=[], count=0, error=str(e))

@app.get("/files/downloads", response_model=FileListResponse)
async def list_downloads(status: Optional[str] = None, thread_id: Optional[str] = None):
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = download_file_manager.list_files(status=file_status, thread_id=thread_id)
        return FileListResponse(success=True, files=[f.to_orchestrator_format() for f in files], count=len(files))
    except Exception as e:
        return FileListResponse(success=False, files=[], count=0, error=str(e))

@app.get("/files/{file_type}/{file_id}")
async def get_file_info(file_type: str, file_id: str):
    try:
        if file_type == "screenshots":
            metadata = screenshot_file_manager.get_file(file_id)
        elif file_type == "downloads":
            metadata = download_file_manager.get_file(file_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type")
        
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
            
        return {"success": True, "file": metadata.to_orchestrator_format()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/stats", response_model=FileStatsResponse)
async def get_file_stats():
    try:
        return FileStatsResponse(
            success=True,
            screenshots=screenshot_file_manager.get_stats(),
            downloads=download_file_manager.get_stats()
        )
    except Exception as e:
        return FileStatsResponse(success=False, screenshots={}, downloads={}, error=str(e))

if __name__ == "__main__":
    import uvicorn
    # Read port from AGENT_DEFINITION or default to 8090
    port = 8090
    endpoints = AGENT_DEFINITION.get("endpoints", [])
    if endpoints:
        url = endpoints[0].get("endpoint", "")
        if url and ":" in url:
            try:
                port = int(url.split(":")[-1].split("/")[0])
            except:
                pass
    
    uvicorn.run(app, host="0.0.0.0", port=port)

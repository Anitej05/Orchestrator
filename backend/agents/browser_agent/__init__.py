"""
Browser Agent - FastAPI Entry Point

UAP-compliant browser automation agent with /execute and /continue endpoints.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from .agent import BrowserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import standardized schemas
import sys
from pathlib import Path
backend_root = Path(__file__).resolve().parents[3]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus


# =============================================================================
# UAP SCHEMAS
# =============================================================================

class UAPExecuteRequest(BaseModel):
    """UAP standard request format for /execute endpoint."""
    prompt: str = Field(..., description="Natural language task description")
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional: headless, max_steps, thread_id"
    )
    task_id: Optional[str] = Field(default=None)
    thread_id: Optional[str] = Field(default=None)


class UAPContinueRequest(BaseModel):
    """UAP standard request format for /continue endpoint."""
    task_id: str = Field(..., description="Task ID from previous /execute call")
    answer: str = Field(..., description="User's answer to the agent's question")
    thread_id: Optional[str] = Field(default=None)


class UAPResponse(BaseModel):
    """UAP standard response format."""
    success: bool
    result: Any
    status: str  # "completed", "needs_input", "in_progress", "error"
    task_id: Optional[str] = None
    question: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager"""
    logger.info("🚀 Browser Agent starting...")
    yield
    logger.info("👋 Browser Agent shutting down...")


app = FastAPI(
    title="Browser Automation Agent",
    description="UAP-compliant browser automation",
    version="3.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# UAP ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint for protocol compliance."""
    return {
        "status": "active",
        "agent": "browser-automation-agent",
        "version": "3.0.0",
        "endpoints": ["/execute", "/health"]
    }


@app.get("/health")
async def health():
    """UAP health check endpoint."""
    return {
        "status": "healthy",
        "agent_id": "browser_automation_agent",
        "agent_name": "Browser Automation Agent",
        "version": "3.0.0",
        "capabilities": ["browse", "click", "type", "extract", "screenshot"]
    }


@app.post("/execute", response_model=AgentResponse)
async def execute(request: UAPExecuteRequest):
    """
    UAP unified execution endpoint.
    
    Accepts natural language prompts and executes browser automation tasks.
    """
    try:
        task_id = request.task_id or str(uuid.uuid4())
        payload = request.payload or {}
        thread_id = request.thread_id or payload.get("thread_id")
        
        logger.info(f"📥 Execute: {request.prompt[:100]}...")
        
        agent = BrowserAgent(
            task=request.prompt,
            headless=payload.get("headless", True),
            thread_id=thread_id
        )
        
        result = await agent.run()
        
        logger.info(f"✅ Task complete: {result.task_summary}")
        
        return AgentResponse(
            status=AgentResponseStatus.COMPLETE if result.success else AgentResponseStatus.ERROR,
            result={
                "task_summary": result.task_summary,
                "extracted_data": result.extracted_data,
                "actions_taken": result.actions_taken,
                "metrics": result.metrics
            },
            standard_response=StandardAgentResponse(
                status="success" if result.success else "error",
                summary=result.task_summary,
                data={
                     "extracted_data": result.extracted_data,
                     "metrics": result.metrics
                },
                error_message=result.error if not result.success else None
            ),
            error=result.error if not result.success else None
        )
        
    except Exception as e:
        logger.error(f"❌ Task failed: {e}")
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=str(e),
            standard_response=StandardAgentResponse(
                status="error",
                summary="Browser task failed exception",
                error_message=str(e)
            )
        )


@app.post("/continue", response_model=UAPResponse)
async def continue_task(request: UAPContinueRequest):
    """
    UAP continue endpoint for multi-turn conversations.
    
    Browser agent runs tasks to completion; this endpoint exists for protocol compliance.
    """
    logger.info(f"📥 Continue: task_id={request.task_id}")
    
    return UAPResponse(
        success=False,
        result=None,
        status="error",
        task_id=request.task_id,
        error="Browser agent runs tasks to completion. Use /execute for new tasks."
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)



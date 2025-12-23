"""
Browser Agent - FastAPI Entry Point

Modular, simple browser automation agent.
"""

import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .agent import BrowserAgent
from .schemas import BrowserTask, BrowserResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager"""
    logger.info("🚀 Browser Agent starting...")
    yield
    logger.info("👋 Browser Agent shutting down...")


app = FastAPI(
    title="Browser Automation Agent",
    description="Simple, reliable browser automation",
    version="2.0.0",
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


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "agent": "browser", "version": "2.0.0"}


@app.post("/browse", response_model=BrowserResult)
async def browse(request: BrowserTask):
    """
    Execute a browser automation task.
    
    Args:
        request: BrowserTask with task description and options
        
    Returns:
        BrowserResult with success status, actions taken, and extracted data
    """
    try:
        logger.info(f"📥 Received task: {request.task}")
        
        agent = BrowserAgent(
            task=request.task,
            headless=request.headless
        )
        
        result = await agent.run()
        
        logger.info(f"✅ Task complete: {result.task_summary}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)

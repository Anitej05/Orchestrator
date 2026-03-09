# agents/integrations_agent/agent.py
"""
Integrations Agent

Universal fallback agent with in-chat OAuth and session persistence.
Handles requests where no dedicated agent exists, and manages Composio connections.

Key Features:
- In-chat OAuth: returns auth URL when connection is missing
- App detection: identifies which Composio app a task requires
- Session persistence: preserves context across conversations
- Tool execution caching (TTL: 5 minutes)
- Full UAP compliance
- Graceful error handling
"""

import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import sys
import os

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from orchestrator.uap_schemas import UAPExecuteRequest, UAPResponse
from .service import IntegrationsAgentService
from .tool_cache import ToolCache

logger = logging.getLogger("integrations_agent")

# Create FastAPI app
app = FastAPI(
    title="Integrations Agent",
    description="Universal fallback agent with in-chat OAuth and 100+ Composio integrations.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tool cache (shared across all users)
tool_cache = ToolCache(ttl_seconds=300)  # 5 minute TTL

# Service instances (per-user)
_service_cache: Dict[str, IntegrationsAgentService] = {}

def get_service(user_id: str) -> IntegrationsAgentService:
    """Get or create service instance for user"""
    if user_id not in _service_cache:
        _service_cache[user_id] = IntegrationsAgentService(user_id, tool_cache)
    return _service_cache[user_id]

# === Health & Info Endpoints ===

@app.get("/")
async def root():
    """Root endpoint - agent info"""
    return {
        "agent": "Integrations Agent",
        "version": "1.0.0",
        "description": "Universal agent for Composio integrations (Slack, Notion, GitHub, etc.)",
        "status": "operational",
        "supports": [
            "Slack",
            "Notion",
            "GitHub",
            "Linear",
            "Jira",
            "Asana",
            "Trello",
            "Figma",
            "And 100+ other Composio apps"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "integrations_agent"}

# === UAP Execute Endpoint ===

@app.post("/execute", response_model=UAPResponse)
async def execute(request: UAPExecuteRequest) -> UAPResponse:
    """
    UAP Execute endpoint - main entry point for all tasks.
    
    Workflow:
    1. Extract user_id from payload
    2. Check if user has required app connection
    3. If not connected, return needs_input with connect URL
    4. If connected, discover and execute tools
    5. Return result in UAP format
    
    Example Request:
    ```json
    {
        "prompt": "Send a Slack message to #general saying 'Hello!'",
        "payload": {
            "user_id": "user_123"
        }
    }
    ```
    
    Example Response (Connected):
    ```json
    {
        "success": true,
        "result": {"message": "sent", "channel": "#general"},
        "status": "completed",
        "execution_time_ms": 1234
    }
    ```
    
    Example Response (Not Connected):
    ```json
    {
        "success": false,
        "result": null,
        "status": "needs_input",
        "question": "Please connect your Slack account to continue.",
        "error": "No Slack connection found. Connect at: https://app.orbimesh.com/connections/slack"
    }
    ```
    """
    start_time = time.time()
    
    try:
        # Extract user_id from payload
        if not request.payload or "user_id" not in request.payload:
            return UAPResponse(
                success=False,
                result=None,
                status="error",
                error="Missing 'user_id' in payload. Cannot execute without user context."
            )
        
        user_id = request.payload["user_id"]
        service = get_service(user_id)
        
        # Execute the prompt using the service
        result = await service.execute(
            prompt=request.prompt,
            payload=request.payload,
            task_id=request.task_id,
            thread_id=request.thread_id
        )
        
        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000
        result["execution_time_ms"] = execution_time_ms
        
        return UAPResponse(**result)
        
    except ValueError as e:
        # User-friendly errors (connection issues, invalid input)
        return UAPResponse(
            success=False,
            result=None,
            status="error",
            error=str(e),
            execution_time_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"[Execute] Unexpected error: {e}", exc_info=True)
        return UAPResponse(
            success=False,
            result=None,
            status="error",
            error=f"Internal error: {str(e)}",
            execution_time_ms=(time.time() - start_time) * 1000
        )

# === Cache Management Endpoints ===

@app.post("/cache/invalidate/{user_id}")
async def invalidate_cache(user_id: str):
    """
    Invalidate tool cache for a user.
    
    Call this when:
    - User disconnects an app
    - User connects a new app
    - Tool execution fails due to stale cache
    """
    try:
        tool_cache.invalidate_user(user_id)
        logger.info(f"Invalidated cache for user {user_id}")
        return {
            "success": True,
            "message": f"Cache invalidated for user {user_id}"
        }
    except Exception as e:
        logger.error(f"Cache invalidation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return tool_cache.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)

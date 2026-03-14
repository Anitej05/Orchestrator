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

from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus
from orchestrator.uap_schemas import UAPExecuteRequest
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

@app.post("/execute", response_model=AgentResponse)
async def execute(request: UAPExecuteRequest) -> AgentResponse:
    """
    UAP Execute endpoint - main entry point for all tasks.
    
    Returns StandardAgentResponse format for consistency with all other agents.

    Workflow:
    1. Extract user_id from payload
    2. Check if user has required app connection
    3. If not connected, return needs_input with connect URL
    4. If connected, discover and execute tools
    5. Return result in StandardAgentResponse format
    """
    start_time = time.time()

    try:
        # Extract user_id from payload
        if not request.payload or "user_id" not in request.payload:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                success=False,
                error_message="Missing 'user_id' in payload",
                standard_response=StandardAgentResponse(
                    status=AgentResponseStatus.ERROR,
                    success=False,
                    summary="Missing user_id in payload",
                )
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
        
        # Determine status
        is_success = result.get('success', False)
        status = AgentResponseStatus.COMPLETE if is_success else AgentResponseStatus.ERROR
        
        # Check if needs_input (for OAuth)
        if 'needs_input' in result or result.get('status') == 'needs_input':
            status = AgentResponseStatus.NEEDS_INPUT
        
        return AgentResponse(
            status=status,
            success=is_success,
            summary=result.get('message', 'Integration operation completed'),
            standard_response=StandardAgentResponse(
                status=status,
                success=is_success,
                summary=result.get('message', 'Integration operation completed'),
                data=result.get('data', result),
                canvas_display=result.get('canvas_display'),
                question=result.get('question'),
                question_type=result.get('question_type'),
                execution_time_ms=execution_time_ms,
            )
        )

    except ValueError as e:
        # User-friendly errors (connection issues, invalid input)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            success=False,
            error_message=str(e),
            standard_response=StandardAgentResponse(
                status=AgentResponseStatus.ERROR,
                success=False,
                summary=f"Error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"[Execute] Unexpected error: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            success=False,
            error_message=f"Internal error: {str(e)}",
            standard_response=StandardAgentResponse(
                status=AgentResponseStatus.ERROR,
                success=False,
                summary=f"Internal error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
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

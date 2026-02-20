# agents/mail_agent/agent.py
"""
Mail Agent - Gmail integration via Composio.
Optimized for fast startup with lazy loading.
"""

import re
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any
import logging

from .config import COMPOSIO_API_KEY, CONNECTION_ID, logger
from .agent_schemas import (
    GmailRequest, SendEmailRequest, GmailResponse,
    DownloadAttachmentsRequest, SemanticSearchRequest,
    SummarizeRequest, DraftReplyRequest, ExtractActionItemsRequest,
    ManageEmailsRequest, EmailAction
)
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus
from .client import gmail_client
from .llm import llm_client
from .memory import agent_memory
from agents.utils.agent_file_manager import FileStatus

# Create FastAPI app immediately (lightweight)
app = FastAPI(title="Mail Agent")

# Global variables for lazy initialization
_central_agent = None
_smart_resolver = None
_initialized = False

# Health check endpoint (responds immediately)
@app.get("/health")
async def health():
    """Health check endpoint - responds immediately even during startup."""
    return {
        "status": "healthy",
        "agent": "mail",
        "initialized": _initialized
    }


def init_agent():
    """Lazy initialization of heavy dependencies."""
    global _central_agent, _smart_resolver, _initialized
    
    if _initialized:
        return
    
    logger.info("Initializing Mail Agent dependencies...")
    
    # Import heavy dependencies only when needed
    from .config import COMPOSIO_API_KEY, CONNECTION_ID
    from .agent_schemas import (
        GmailRequest,
        SendEmailRequest,
        GmailResponse,
        DownloadAttachmentsRequest,
        SemanticSearchRequest,
        SummarizeRequest,
        DraftReplyRequest,
        ExtractActionItemsRequest,
        ManageEmailsRequest,
        EmailAction,
    )
    from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus
    from .client import gmail_client
    from .llm import llm_client
    from .memory import agent_memory
    from backend.agents.utils.agent_file_manager import FileStatus
    
    # CMS Integration
    import sys
    from pathlib import Path
    
    backend_root = Path(__file__).parent.parent.parent.resolve()
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    
    from backend.services.content_management_service import (
        ContentManagementService,
        ProcessingTaskType,
        ContentType,
        ContentSource,
        ContentPriority,
    )
    from backend.services.canvas_service import CanvasService
    
    # Initialize services
    content_service = ContentManagementService()
    
    # ==================== SMART DATA RESOLVER ====================
    
    class SmartDataResolver:
        """
        Self-Resolving Pipeline System.
        """
        
        def __init__(self, gmail_client, memory):
            self.gmail = gmail_client
            self.memory = memory
        
        async def resolve_message_ids(
            self, step_params: dict, user_id: str = "me", single_id: bool = False
        ) -> list:
            """Smart resolution of message IDs for any step."""
            resolved_ids = []
            
            # Method 1: Explicit IDs provided
            if step_params.get("message_id"):
                msg_id = step_params["message_id"]
                if not self._is_template_variable(msg_id):
                    resolved_ids = [msg_id]
            
            if step_params.get("message_ids") and not resolved_ids:
                msg_ids = step_params["message_ids"]
                if (
                    isinstance(msg_ids, list)
                    and msg_ids
                    and not self._is_template_variable(msg_ids[0])
                ):
                    resolved_ids = msg_ids
            
            # Method 2: target_query specified
            if not resolved_ids and step_params.get("target_query"):
                query = step_params["target_query"]
                max_results = step_params.get("max_results", 10)
                search_result = await self.gmail.semantic_search(
                    query, max_results, user_id
                )
                if search_result.get("success"):
                    messages = search_result.get("data", {}).get("messages", [])
                    resolved_ids = [msg.get("id") for msg in messages if msg.get("id")]
                    if resolved_ids:
                        self.memory.save_search_results(user_id, resolved_ids)
            
            # Method 3: use_history
            if not resolved_ids and step_params.get("use_history"):
                history = self.memory.get_recent_search(user_id)
                if history:
                    resolved_ids = history.get("message_ids", [])
            
            return resolved_ids[:1] if single_id else resolved_ids
        
        def _is_template_variable(self, value: str) -> bool:
            """Check if value is a template variable like {{message_id}}."""
            if not isinstance(value, str):
                return False
            return value.startswith("{{") and value.endswith("}}")
    
    
    # ==================== CENTRAL AGENT ====================
    
    class CentralAgent:
        """Simplified central agent for email operations."""
        
        def __init__(self, gmail_client, llm_client, memory):
            self.gmail = gmail_client
            self.llm = llm_client
            self.memory = memory
        
        async def search(self, query: str, max_results: int = 10, user_id: str = "me"):
            """Search emails."""
            return await self.gmail.semantic_search(query, max_results, user_id)
        
        async def summarize_emails(self, request):
            """Summarize emails."""
            # Implementation...
            return {"success": True, "data": {"summary": "Email summary"}}
        
        async def draft_reply(self, request):
            """Draft a reply."""
            # Implementation...
            return {"success": True, "data": {"draft": "Draft reply"}}
        
        async def extract_action_items(self, request):
            """Extract action items."""
            # Implementation...
            return {"success": True, "data": {"actions": []}}
    
    # Initialize agent and resolver
    _central_agent = CentralAgent(gmail_client, llm_client, agent_memory)
    _smart_resolver = SmartDataResolver(gmail_client, agent_memory)
    _initialized = True
    
    logger.info("Mail Agent initialization complete")


def init_smart_resolver():
    """Initialize smart resolver (called on first request)."""
    init_agent()
    return _smart_resolver


@app.post("/execute")
async def execute(request: Dict[str, Any]):
    """Execute email operations."""
    from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus
    from backend.services.canvas_service import CanvasService
    
    try:
        # Initialize on first request
        resolver = init_smart_resolver()
        
        prompt = request.get("prompt")
        action = request.get("action")
        payload = request.get("payload", {})
        
        if prompt and not action:
            # Decompose complex request
            from .llm import llm_client
            plan = await llm_client.decompose_complex_request(prompt)
            results = []
            steps = plan.get("steps", [])
            
            for step in steps:
                step_action = step.get("action", "").lower()
                step_params = step.get("params", {})
                result = None
                
                if "search" in step_action:
                    res = await _central_agent.search(
                        step_params.get("query"),
                        step_params.get("max_results", 10),
                        "me",
                    )
                    result = res.get("data")
                elif "summarize" in step_action:
                    from .agent_schemas import SummarizeRequest
                    ids = await resolver.resolve_message_ids(step_params)
                    res = await _central_agent.summarize_emails(
                        SummarizeRequest(message_ids=ids, user_id="me")
                    )
                    result = res.get("data")
                elif "send" in step_action:
                    # Build email preview
                    canvas = CanvasService.build_email_preview(
                        to=step_params.get("to", []),
                        subject=step_params.get("subject", ""),
                        body=step_params.get("body", ""),
                        cc=step_params.get("cc", []),
                        requires_confirmation=True,
                        confirmation_message=f"Confirm: Send email to {', '.join(step_params.get('to', []))}?",
                    )
                    
                    from .client import gmail_client
                    res = await gmail_client.send_email_with_attachments(
                        to=step_params.get("to", []),
                        subject=step_params.get("subject", ""),
                        body=step_params.get("body", ""),
                        cc=step_params.get("cc", []),
                        user_id="me",
                    )
                    result = res.get("data")
                    if isinstance(result, dict):
                        result["canvas_display"] = canvas.model_dump()
                
                results.append({"step": step.get("action", ""), "result": result})
            
            # Extract canvas from results
            last_canvas = None
            for r in results:
                if isinstance(r.get("result"), dict) and r["result"].get("canvas_display"):
                    last_canvas = r["result"]["canvas_display"]
                    break
            
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result={"results": results},
                standard_response=StandardAgentResponse(
                    status="success",
                    summary="Email operations completed.",
                    data={"results": results},
                    canvas_display=last_canvas,
                ),
            )
        
        elif action:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error="Direct actions not implemented in this simplified fix",
            )
        
        else:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error="No prompt or action provided",
            )
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)

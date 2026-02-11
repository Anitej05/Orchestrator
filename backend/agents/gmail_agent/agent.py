# agents/gmail_agent/agent.py
import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional
import sys
import os
from cachetools import TTLCache

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from schemas import AgentResponse, AgentResponseStatus
from .agent_schemas import (
    SearchRequest, SendEmailRequest, ReplyRequest,
    CreateDraftRequest, SummarizeRequest, DraftReplyRequest,
    ExtractActionsRequest, AddLabelsRequest, DownloadAttachmentsRequest,
    ExecuteRequest, GmailResponse
)
from .service import GmailService
from .memory import agent_memory

logger = logging.getLogger("gmail_agent")

# Create FastAPI app
app = FastAPI(
    title="Gmail Agent",
    description="Clean Composio-native Gmail agent using official SDK",
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

# Cache for service instances (per-user) with TTL of 30 minutes
_service_cache: TTLCache[str, GmailService] = TTLCache(maxsize=100, ttl=1800)

def get_service(user_id: str) -> GmailService:
    """Get or create Gmail service for user (with TTL cache)"""
    if user_id not in _service_cache:
        _service_cache[user_id] = GmailService(user_id)
    return _service_cache[user_id]

# === Health & Info Endpoints ===

@app.get("/")
async def root():
    """Root endpoint - agent info"""
    return {
        "agent": "Gmail Agent",
        "version": "1.0.0",
        "description": "Composio-native Gmail agent with 23 tools",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "gmail_agent"}

# === Email Operations ===

@app.post("/search")
async def search_emails(request: SearchRequest):
    """Search emails with optional LLM optimization"""
    try:
        service = get_service(request.user_id)
        result = await service.search_emails(
            query=request.query,
            max_results=request.max_results,
            include_payload=request.include_payload
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Search] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/send")
async def send_email(request: SendEmailRequest):
    """Send email"""
    try:
        service = get_service(request.user_id)
        result = await service.send_email(
            to=request.to,
            subject=request.subject,
            body=request.body,
            cc=request.cc,
            bcc=request.bcc,
            is_html=request.is_html,
            attachments=request.attachments
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Send] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reply")
async def reply_to_email(request: ReplyRequest):
    """Reply to email/thread"""
    try:
        service = get_service(request.user_id)
        result = await service.reply_to_email(
            thread_id=request.thread_id,
            message_id=request.message_id,
            body=request.body,
            to=request.to,
            cc=request.cc
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Reply] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/message/{user_id}/{message_id}")
async def get_message(user_id: str, message_id: str):
    """Get single email"""
    try:
        service = get_service(user_id)
        result = await service.get_email(message_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[GetMessage] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/message/{user_id}/{message_id}")
async def delete_message(user_id: str, message_id: str, permanent: bool = False):
    """Delete email"""
    try:
        service = get_service(user_id)
        result = await service.delete_email(message_id, permanent)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[DeleteMessage] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trash/{user_id}/{message_id}")
async def trash_message(user_id: str, message_id: str):
    """Move email to trash"""
    try:
        service = get_service(user_id)
        result = await service.delete_email(message_id, permanent=False)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[TrashMessage] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Draft Operations ===

@app.post("/draft/create")
async def create_draft(request: CreateDraftRequest):
    """Create email draft"""
    try:
        service = get_service(request.user_id)
        result = await service.create_draft(
            to=request.to,
            subject=request.subject,
            body=request.body,
            cc=request.cc,
            thread_id=request.thread_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[CreateDraft] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drafts/{user_id}")
async def list_drafts(user_id: str, max_results: int = 10):
    """List drafts"""
    try:
        service = get_service(user_id)
        result = await service.list_drafts(max_results)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ListDrafts] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/draft/{user_id}/{draft_id}/send")
async def send_draft(user_id: str, draft_id: str):
    """Send draft"""
    try:
        service = get_service(user_id)
        result = await service.send_draft(draft_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[SendDraft] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/draft/{user_id}/{draft_id}")
async def delete_draft(user_id: str, draft_id: str):
    """Delete draft"""
    try:
        service = get_service(user_id)
        result = await service.delete_draft(draft_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[DeleteDraft] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Label Operations ===

@app.post("/labels/add")
async def add_labels(request: AddLabelsRequest):
    """Add labels to email"""
    try:
        service = get_service(request.user_id)
        result = await service.add_labels(
            message_id=request.message_id,
            label_ids=request.label_ids
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[AddLabels] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/labels/{user_id}")
async def list_labels(user_id: str):
    """List all labels"""
    try:
        service = get_service(user_id)
        result = await service.list_labels()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ListLabels] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/labels/create/{user_id}")
async def create_label(user_id: str, name: str = Body(..., embed=True)):
    """Create custom label"""
    try:
        service = get_service(user_id)
        result = await service.create_label(name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[CreateLabel] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Attachment Operations ===

@app.post("/attachments/download")
async def download_attachments(request: DownloadAttachmentsRequest):
    """Download attachments"""
    try:
        service = get_service(request.user_id)
        
        if request.attachment_ids:
            # Download specific attachments
            results = []
            for att_id in request.attachment_ids:
                result = await service.download_attachment(
                    message_id=request.message_id,
                    attachment_id=att_id,
                    file_name=f"attachment_{att_id}"
                )
                results.append(result)
            return {"success": True, "files": results}
        else:
            # Download all attachments
            result = await service.download_all_attachments(request.message_id)
            return result
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[DownloadAttachments] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === LLM-Enhanced Operations ===

@app.post("/summarize")
async def summarize_emails(request: SummarizeRequest):
    """Summarize emails"""
    try:
        service = get_service(request.user_id)
        result = await service.summarize_emails(request.message_ids)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[Summarize] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/draft-reply")
async def draft_smart_reply(request: DraftReplyRequest):
    """AI-generated reply draft"""
    try:
        service = get_service(request.user_id)
        result = await service.draft_smart_reply(
            message_id=request.message_id,
            user_instructions=request.user_instructions
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[DraftReply] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-actions")
async def extract_action_items(request: ExtractActionsRequest):
    """Extract action items"""
    try:
        service = get_service(request.user_id)
        result = await service.extract_action_items(request.message_ids)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ExtractActions] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Contact Operations ===

@app.get("/contacts/{user_id}")
async def list_contacts(user_id: str, max_results: int = 100):
    """List contacts"""
    try:
        service = get_service(user_id)
        result = await service.list_contacts(max_results)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ListContacts] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contacts/search/{user_id}")
async def search_contacts(user_id: str, query: str = Body(..., embed=True)):
    """Search contacts"""
    try:
        service = get_service(user_id)
        result = await service.search_contacts(query)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[SearchContacts] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Thread Operations ===

@app.get("/threads/{user_id}")
async def list_threads(user_id: str, query: str = "", max_results: int = 10):
    """List threads"""
    try:
        service = get_service(user_id)
        result = await service.list_threads(query, max_results)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[ListThreads] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thread/{user_id}/{thread_id}")
async def get_thread(user_id: str, thread_id: str):
    """Get thread messages"""
    try:
        service = get_service(user_id)
        result = await service.get_thread(thread_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[GetThread] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Profile ===

@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    """Get Gmail profile"""
    try:
        service = get_service(user_id)
        result = await service.get_profile()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[GetProfile] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === Orchestrator Integration ===

@app.post("/execute")
async def execute_action(request: ExecuteRequest):
    """
    Execute action from orchestrator.
    Supports natural language requests.
    Returns UAP-compliant AgentResponse.
    """
    try:
        service = get_service(request.user_id)
        prompt = request.prompt.lower()
        context = request.context or {}
        
        # Simple intent routing
        if any(word in prompt for word in ["search", "find", "get"]):
            result = await service.search_emails(query=request.prompt)
        elif any(word in prompt for word in ["summarize", "summary"]):
            message_ids = context.get("message_ids") or agent_memory.get_last_search_results(request.user_id) or []
            if not message_ids:
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    result={"error": "No emails to summarize. Please search first."}
                ).dict()
            result = await service.summarize_emails(message_ids)
        elif any(word in prompt for word in ["send", "compose"]):
            return AgentResponse(
                status=AgentResponseStatus.NEEDS_INPUT,
                result={"error": "Please provide to, subject, and body parameters"}
            ).dict()
        elif any(word in prompt for word in ["reply"]):
            return AgentResponse(
                status=AgentResponseStatus.NEEDS_INPUT,
                result={"error": "Please provide message_id and body parameters"}
            ).dict()
        elif any(word in prompt for word in ["draft"]):
            message_ids = context.get("message_ids") or agent_memory.get_last_search_results(request.user_id) or []
            if not message_ids:
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    result={"error": "No email to reply to. Please specify message_id."}
                ).dict()
            result = await service.draft_smart_reply(message_ids[0])
        elif any(word in prompt for word in ["action", "todo", "extract"]):
            message_ids = context.get("message_ids") or agent_memory.get_last_search_results(request.user_id) or []
            if not message_ids:
                return AgentResponse(
                    status=AgentResponseStatus.ERROR,
                    result={"error": "No emails to analyze. Please search first."}
                ).dict()
            result = await service.extract_action_items(message_ids)
        else:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                result={"error": f"Unknown action: {request.prompt}"}
            ).dict()
        
        # Wrap result in AgentResponse
        if result.get("success"):
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result=result
            ).dict()
        else:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                result=result
            ).dict()
        
    except ValueError as e:
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            result={"error": str(e)}
        ).dict()
    except Exception as e:
        logger.error(f"[Execute] Error: {e}")
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            result={"error": str(e)}
        ).dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

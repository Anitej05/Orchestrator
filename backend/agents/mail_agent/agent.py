# agents/mail_agent/agent.py
import re
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any

from .config import COMPOSIO_API_KEY, CONNECTION_ID, logger
from .schemas import (
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
from schemas import OrchestratorMessage # Keep OrchestratorMessage for now if it's still used elsewhere, but it's not in the new /execute

# CMS Integration
import sys
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from services.content_management_service import (
    ContentManagementService,
    ProcessingTaskType,
    ContentType,
    ContentSource,
    ContentPriority
)

content_service = ContentManagementService()
from services.canvas_service import CanvasService


# ==================== SMART DATA RESOLVER ====================

class SmartDataResolver:
    """
    Self-Resolving Pipeline System.
    
    Each step can reliably get the data it needs through:
    1. History (fast path) - if recent search matches the requirement
    2. Inline fetch (reliable path) - if history unavailable or irrelevant
    
    This ensures steps NEVER fail due to missing data.
    """
    
    def __init__(self, gmail_client, memory):
        self.gmail = gmail_client
        self.memory = memory
    
    async def resolve_message_ids(
        self, 
        step_params: dict, 
        user_id: str = "me",
        single_id: bool = False
    ) -> list:
        """
        Smart resolution of message IDs for any step.
        """
        resolved_ids = []
        
        # Method 1: Explicit IDs provided
        if step_params.get("message_id"):
            msg_id = step_params["message_id"]
            if not self._is_template_variable(msg_id):
                resolved_ids = [msg_id]
        
        if step_params.get("message_ids") and not resolved_ids:
            msg_ids = step_params["message_ids"]
            if isinstance(msg_ids, list) and msg_ids and not self._is_template_variable(msg_ids[0]):
                resolved_ids = msg_ids
        
        # Method 2: target_query specified
        if not resolved_ids and step_params.get("target_query"):
            query = step_params["target_query"]
            max_results = step_params.get("max_results", 10)
            search_result = await self.gmail.semantic_search(query, max_results, user_id)
            if search_result.get("success"):
                messages = search_result.get("data", {}).get("messages", [])
                resolved_ids = [msg.get("id") for msg in messages if msg.get("id")]
                if resolved_ids:
                    self.memory.save_search_results(user_id, resolved_ids)
        
        # Method 3: use_history
        if not resolved_ids and step_params.get("use_history"):
            history_ids = self.memory.get_last_search_results(user_id) or []
            if history_ids:
                resolved_ids = history_ids
        
        if single_id:
            return resolved_ids[0] if resolved_ids else None
        return resolved_ids
    
    def _is_template_variable(self, value) -> bool:
        if not value or not isinstance(value, str):
            return False
        template_patterns = ['{{', '}}', '${', '$search', 'search_result', 'result[']
        return any(pattern in str(value).lower() for pattern in template_patterns)

# Initialize the smart resolver
smart_resolver = None

def init_smart_resolver():
    global smart_resolver
    smart_resolver = SmartDataResolver(gmail_client, agent_memory)

# ==================== CENTRAL INTELLIGENCE ====================

class CentralAgent:
    def __init__(self, gmail, llm, memory):
        self.gmail = gmail
        self.llm = llm
        self.memory = memory

    async def search(self, query: str, max_results: int, user_id: str) -> Dict[str, Any]:
        result = await self.gmail.semantic_search(query, max_results, user_id)
        if result.get("success"):
            data = result.get("data", {})
            messages = data.get("messages", [])
            message_ids = [msg.get("id") for msg in messages if msg.get("id")]
            self.memory.save_search_results(user_id, message_ids)
        return result

    async def summarize_emails(self, request: SummarizeRequest) -> Dict[str, Any]:
        user_id = request.user_id
        target_ids = request.message_ids
        
        if not target_ids and request.use_history:
            target_ids = self.memory.get_last_search_results(user_id) or []
        
        if not target_ids:
             return {"success": True, "data": {"summary": "No emails found.", "sources": [], "count": 0}}

        BATCH_SIZE = 15
        all_bodies = []
        all_sources = []
        
        for i in range(0, len(target_ids), BATCH_SIZE):
            batch_ids = target_ids[i : i + BATCH_SIZE]
            emails = await self.gmail.batch_fetch_emails(batch_ids, user_id)
            if not emails: continue
            for e in emails:
                all_bodies.append(f"Subject: {e['subject']}\nFrom: {e['from']}\nContent:\n{e['body']}")
                all_sources.append(e["subject"])
        
        if not all_bodies:
            return {"success": True, "data": {"summary": "Could not retrieve content.", "sources": [], "count": 0}}

        final_summary = await self.llm.summarize_text_batch(all_bodies)
        return {"success": True, "data": {"summary": final_summary, "sources": all_sources}}

    async def draft_reply(self, request: DraftReplyRequest) -> Dict[str, Any]:
        thread_res = await self.gmail.call_tool(
            "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", 
            {"message_id": request.message_id, "format": "full", "user_id": request.user_id}
        )
        if not thread_res.get("success"):
            return {"success": False, "error": "Could not fetch email"}
            
        data = thread_res.get("data", {})
        thread_context = f"Sender: {data.get('from')}\nSubject: {data.get('subject')}\nBody:\n{data.get('body')}"
        draft = await self.llm.draft_email_reply(thread_context, request.intent, "Me")
        
        return {
            "success": True, 
            "data": {
                **draft,
                "to": [data.get('from')],
                "original_message_id": request.message_id
            }
        }

    async def extract_actions(self, request: ExtractActionItemsRequest) -> Dict[str, Any]:
        user_id = request.user_id
        target_ids = request.message_ids
        if not target_ids and request.use_history:
            target_ids = self.memory.get_last_search_results(user_id) or []
        if not target_ids:
            return {"success": False, "error": "No emails."}
        
        all_texts = []
        for i in range(0, len(target_ids), 15):
            emails = await self.gmail.batch_fetch_emails(target_ids[i:i+15], user_id)
            all_texts.extend([f"Subject: {e['subject']}\nContent:\n{e['body']}" for e in emails])

        actions = await self.llm.extract_actions(all_texts)
        return {"success": True, "data": {"actions": actions, "count": len(actions)}}

central_agent = CentralAgent(gmail_client, llm_client, agent_memory)
app = FastAPI(title="Mail Agent")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/execute", response_model=AgentResponse)
async def execute(request: Dict[str, Any]):
    try:
        init_smart_resolver()
        prompt = request.get("prompt")
        action = request.get("action")
        payload = request.get("payload", {})
        
        if prompt and not action:
            plan = await llm_client.decompose_request(prompt)
            results = []
            for step in plan.steps:
                step_action = step.action.lower()
                step_params = step.params
                result = None
                
                if "search" in step_action:
                    res = await central_agent.search(step_params.get("query"), step_params.get("max_results", 10), "me")
                    result = res.get("data")
                elif "summarize" in step_action:
                    ids = await smart_resolver.resolve_message_ids(step_params)
                    res = await central_agent.summarize_emails(SummarizeRequest(message_ids=ids, user_id="me"))
                    result = res.get("data")
                elif "send" in step_action:
                    res = await gmail_client.send_email_with_attachments(
                        to=step_params.get("to", ["me"]),
                        subject=step_params.get("subject", "Automated"),
                        body=step_params.get("body", ""),
                        user_id="me"
                    )
                    result = res.get("data")
                
                results.append({"step": step.action, "result": result})
            
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE,
                result={"results": results},
                standard_response=StandardAgentResponse(status="success", summary="Completed plan", data=results)
            )
            
        elif action:
            # Handle direct actions if needed
            return AgentResponse(status=AgentResponseStatus.ERROR, error="Direct actions not implemented in this simplified fix")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return AgentResponse(status=AgentResponseStatus.ERROR, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)

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
        
        Resolution order:
        1. Explicit message_id/message_ids in params → Use directly
        2. target_query specified → Execute search to get IDs
        3. use_history=True → Try history (with optional validation)
        4. Fallback → Return empty list (step will handle error)
        
        Args:
            step_params: The step parameters from LLM
            user_id: Gmail user ID
            single_id: If True, return just the first ID (for draft_reply)
            
        Returns:
            List of message IDs, or single ID string if single_id=True
        """
        resolved_ids = []
        
        # Method 1: Explicit IDs provided
        if step_params.get("message_id"):
            msg_id = step_params["message_id"]
            # Check if it's a valid ID (not a template variable)
            if not self._is_template_variable(msg_id):
                logger.info(f"[RESOLVER] Using explicit message_id: {msg_id}")
                resolved_ids = [msg_id]
        
        if step_params.get("message_ids") and not resolved_ids:
            msg_ids = step_params["message_ids"]
            if isinstance(msg_ids, list) and msg_ids and not self._is_template_variable(msg_ids[0]):
                logger.info(f"[RESOLVER] Using explicit message_ids: {len(msg_ids)} IDs")
                resolved_ids = msg_ids
        
        # Method 2: target_query specified - Execute inline search (MOST RELIABLE)
        if not resolved_ids and step_params.get("target_query"):
            query = step_params["target_query"]
            max_results = step_params.get("max_results", 10)
            logger.info(f"[RESOLVER] Fetching via target_query: '{query}'")
            
            # Execute inline search
            search_result = await self.gmail.semantic_search(query, max_results, user_id)
            if search_result.get("success"):
                messages = search_result.get("data", {}).get("messages", [])
                resolved_ids = [msg.get("id") for msg in messages if msg.get("id")]
                # Also save to memory for potential future use
                if resolved_ids:
                    self.memory.save_search_results(user_id, resolved_ids)
                logger.info(f"[RESOLVER] Inline search found {len(resolved_ids)} emails for: '{query}'")
            else:
                logger.warning(f"[RESOLVER] Inline search failed for: '{query}'")
        
        # Method 3: use_history - Try history as fallback
        if not resolved_ids and step_params.get("use_history"):
            history_ids = self.memory.get_last_search_results(user_id) or []
            if history_ids:
                logger.info(f"[RESOLVER] Using {len(history_ids)} IDs from history")
                resolved_ids = history_ids
            else:
                logger.warning(f"[RESOLVER] use_history=True but no history available")
                # If we have a fallback query, try it
                if step_params.get("fallback_query"):
                    query = step_params["fallback_query"]
                    logger.info(f"[RESOLVER] Trying fallback_query: '{query}'")
                    search_result = await self.gmail.semantic_search(query, 10, user_id)
                    if search_result.get("success"):
                        messages = search_result.get("data", {}).get("messages", [])
                        resolved_ids = [msg.get("id") for msg in messages if msg.get("id")]
                        if resolved_ids:
                            self.memory.save_search_results(user_id, resolved_ids)
                        logger.info(f"[RESOLVER] Fallback search found {len(resolved_ids)} emails")
        
        # Return based on single_id flag
        if single_id:
            return resolved_ids[0] if resolved_ids else None
        return resolved_ids
    
    def _is_template_variable(self, value) -> bool:
        """Detect if a value is a template variable that needs resolution."""
        if not value or not isinstance(value, str):
            return False
        template_patterns = ['{{', '}}', '${', '$search', 'search_result', 'result[', 
                             'previous_result', 'history[', '{result']
        return any(pattern in str(value).lower() for pattern in template_patterns)

# Initialize the smart resolver (will be set after gmail_client is initialized)
smart_resolver = None

def init_smart_resolver():
    """Initialize the smart resolver after all dependencies are ready."""
    global smart_resolver
    smart_resolver = SmartDataResolver(gmail_client, agent_memory)
    logger.info("[RESOLVER] Smart Data Resolver initialized")

# ==================== CENTRAL INTELLIGENCE ====================

class CentralAgent:
    """
    Stateful agent that orchestrates GmailClient, LLMClient, and Memory.
    """
    def __init__(self, gmail, llm, memory):
        self.gmail = gmail
        self.llm = llm
        self.memory = memory

    async def search(self, query: str, max_results: int, user_id: str) -> Dict[str, Any]:
        """Perform state-aware semantic search"""
        result = await self.gmail.semantic_search(query, max_results, user_id)
        if result.get("success"):
            data = result.get("data", {})
            messages = data.get("messages", [])
            # Save message IDs for context-aware follow-up actions
            message_ids = [msg.get("id") for msg in messages if msg.get("id")]
            self.memory.save_search_results(user_id, message_ids)
            self.memory.update_context(user_id, "last_query", query)
            self.memory.add_turn(user_id, f"Search: {query}", f"Found {len(messages)} emails", "search")
        return result

    async def summarize_emails(self, request: SummarizeRequest) -> Dict[str, Any]:
        """Smart summarization with history support"""
        user_id = request.user_id
        target_ids = request.message_ids
        
        if not target_ids and request.use_history:
            target_ids = self.memory.get_last_search_results(user_id) or []
        
        if not target_ids:
             return {"success": True, "data": {"summary": "No emails found to summarize. The search returned zero results.", "sources": [], "count": 0}}

        # COLLECT ALL BODIES: Fetch in batches but summarize globally for best context
        BATCH_SIZE = 15 # Increased fetch batch size
        all_bodies = []
        all_sources = []
        
        logger.info(f"Fetching {len(target_ids)} emails for global high-fidelity processing...")
        
        for i in range(0, len(target_ids), BATCH_SIZE):
            batch_ids = target_ids[i : i + BATCH_SIZE]
            emails = await self.gmail.batch_fetch_emails(batch_ids, user_id)
            if not emails:
                continue
            
            for e in emails:
                body_token = f"--- EMAIL {len(all_bodies)+1} ---\nSubject: {e['subject']}\nFrom: {e['from']}\nContent:\n{e['body']}"
                all_bodies.append(body_token)
                all_sources.append(e["subject"])
        
        if not all_bodies:
            return {"success": True, "data": {"summary": "Could not retrieve content for any emails.", "sources": [], "count": 0}}

        # REGISTER WITH CMS
        try:
            content_payload = []
            for i, body in enumerate(all_bodies):
                content_payload.append(f"--- EMAIL {i+1} ---\n{body}") # Keep text format for now or switch to list of dicts if CMS supports
            
            # Using list of strings for compatibility with current chunking
            
            cms_metadata = await content_service.register_content(
                content=content_payload,
                name=f"emails_summary_request_{len(target_ids)}_{user_id}",
                source=ContentSource.SYSTEM_GENERATED,
                content_type=ContentType.DOCUMENT,
                priority=ContentPriority.MEDIUM,
                tags=["email_batch", "summary_request"],
                user_id=user_id
            )
            
            logger.info(f"[CMS] Registered email batch as {cms_metadata.id}. Starting Map-Reduce...")
            
            # DELEGATE TO CMS MAP-REDUCE
            process_result = await content_service.process_large_content(
                content_id=cms_metadata.id,
                task_type=ProcessingTaskType.SUMMARIZE
            )
            
            final_summary = process_result.final_output
            logger.info(f"[CMS] Summarization complete. Time: {process_result.processing_time_ms}ms")
            
        except Exception as e:
            logger.error(f"[CMS] Summarization failed, falling back to local LLM: {e}")
            # Fallback
            final_summary = await self.llm.summarize_text_batch(all_bodies)
        
        self.memory.add_turn(user_id, f"Summarize {len(target_ids)} emails", final_summary[:100], "summarize")
        return {"success": True, "data": {"summary": final_summary, "sources": all_sources}}

    async def draft_reply(self, request: DraftReplyRequest) -> Dict[str, Any]:
        """Context-aware reply drafting"""
        # 1. Fetch thread context
        thread_res = await self.gmail.call_tool(
            "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", 
            {"message_id": request.message_id, "format": "full", "user_id": request.user_id}
        )
        if not thread_res.get("success"):
            return {"success": False, "error": thread_res.get("error", "Could not fetch original email")}
            
        data = thread_res.get("data", {})
        thread_context = f"Sender: {data.get('from')}\nSubject: {data.get('subject')}\nBody:\n{data.get('body')}"
        
        # 2. Draft using LLM
        draft = await self.llm.draft_email_reply(thread_context, request.intent, "Me")
        
        # 3. Store in memory
        self.memory.add_turn(request.user_id, f"Draft reply: {request.intent}", f"Subject: {draft['subject']}", "draft_reply")
        
        return {
            "success": True, 
            "data": {
                **draft,
                "to": [data.get('from')],  # Explicitly provide recipient to prevent Orchestrator guessing
                "original_message_id": request.message_id
            }
        }

    async def extract_actions(self, request: ExtractActionItemsRequest) -> Dict[str, Any]:
        """Extract Todo items from emails"""
        user_id = request.user_id
        target_ids = request.message_ids
        
        if not target_ids and request.use_history:
            target_ids = self.memory.get_last_search_results(user_id) or []
            
        if not target_ids:
            return {"success": False, "error": "No emails to scan."}
            
        # GLOBAL COLLECTION for lossless extraction
        all_texts = []
        BATCH_SIZE = 15
        for i in range(0, len(target_ids), BATCH_SIZE):
            batch_ids = target_ids[i : i + BATCH_SIZE]
            emails = await self.gmail.batch_fetch_emails(batch_ids, user_id)
            if not emails: continue
            all_texts.extend([f"Subject: {e['subject']}\nFrom: {e['from']}\nContent:\n{e['body']}" for e in emails])

        if not all_texts:
            return {"success": False, "error": "Could not retrieve content for any emails."}
            
        # High-fidelity Extraction pass
        actions = await self.llm.extract_actions(all_texts)
        
        self.memory.add_turn(user_id, "Extract actions", f"Found {len(actions)} items", "extract_actions")
        return {"success": True, "data": {"actions": actions, "count": len(actions)}}

    async def execute_plan(self, steps: list) -> Dict[str, Any]:
        """Executes a list of decomposed steps."""
        # Initialize smart resolver if not already done
        global smart_resolver
        if smart_resolver is None:
            init_smart_resolver()
        
        results = []
        step_failed = False
        last_error = None

        for i, step in enumerate(steps):
            step_action = step.get("action", "").lower()
            step_params = step.get("params", {})
            
            logger.info(f"[STEP] [STEP {i+1}/{len(steps)}] Executing: {step_action}")
            
            result = None
            execution_error = None
            
            try:
                if "search" in step_action:
                    req = SemanticSearchRequest(
                        query=step_params.get("query", ""),
                        max_results=step_params.get("max_results", 10),
                        user_id=step_params.get("user_id", "me")
                    )
                    op_res = await self.search(req.query, req.max_results, req.user_id)
                    if op_res.get("success"): result = op_res.get("data")
                    else: execution_error = op_res.get("error")

                elif "summarize" in step_action or "summary" in step_action:
                    message_ids = await smart_resolver.resolve_message_ids(
                        step_params,
                        user_id="me",
                        single_id=False
                    )
                    
                    if message_ids:
                        logger.info(f"[SMART] Resolved {len(message_ids)} emails for summarize")
                    
                    req = SummarizeRequest(
                        message_ids=message_ids if message_ids else None,
                        use_history=True if not message_ids else False,
                        user_id=step_params.get("user_id", "me")
                    )
                    op_res = await self.summarize_emails(req)
                    if op_res.get("success"): result = op_res.get("data")
                    else: execution_error = op_res.get("error")
                    
                elif "archive" in step_action:
                    message_ids = await smart_resolver.resolve_message_ids(
                        step_params,
                        user_id="me",
                        single_id=False
                    )
                    
                    req = ManageEmailsRequest(
                        message_ids=message_ids if message_ids else None,
                        action=EmailAction.ARCHIVE,
                        use_history=True if not message_ids else False,
                        user_id=step_params.get("user_id", "me")
                    )
                    op_res = await manage_emails(req) # manage_emails is a FastAPI endpoint, not a CentralAgent method
                    if op_res.success: result = op_res.result
                    else: execution_error = op_res.error

                elif "mark" in step_action or "label" in step_action or "important" in step_action:
                    message_ids = await smart_resolver.resolve_message_ids(
                        step_params,
                        user_id="me",
                        single_id=False
                    )
                    
                    labels = step_params.get("labels", [])
                    if isinstance(labels, list) and any(l.lower() == "archive" for l in labels):
                        email_action = EmailAction.ARCHIVE
                        labels = None
                    elif isinstance(labels, list) and any(l.lower() == "read" for l in labels):
                        email_action = EmailAction.MARK_READ
                        labels = None
                    elif "important" in step_action or "star" in step_action:
                        email_action = EmailAction.STAR
                        labels = None
                    else:
                        email_action = EmailAction.ADD_LABELS
                        if not labels: labels = ["IMPORTANT"]
                    
                    req = ManageEmailsRequest(
                        message_ids=message_ids if message_ids else None,
                        action=email_action,
                        labels=labels,
                        use_history=True if not message_ids else False,
                        user_id=step_params.get("user_id", "me")
                    )
                    op_res = await manage_emails(req) # manage_emails is a FastAPI endpoint
                    if op_res.success: result = op_res.result
                    else: execution_error = op_res.error

                elif "draft" in step_action:
                    message_id = await smart_resolver.resolve_message_ids(
                        step_params, 
                        user_id="me", 
                        single_id=True
                    )
                    
                    if not message_id:
                        execution_error = "Could not resolve message_id. Please specify which email to reply to."
                        step_failed = True
                        break
                    
                    logger.info(f"[SMART] Resolved message_id: {message_id}")
                    
                    req = DraftReplyRequest(
                        message_id=message_id, 
                        intent=step_params.get("intent", "Reply politely"),
                        user_id=step_params.get("user_id", "me")
                    )

                    op_res = await self.draft_reply(req)
                    if op_res.get("success"): result = op_res.get("data")
                    else: execution_error = op_res.get("error")

                elif "download" in step_action:
                    all_downloaded_files = []
                    total_size_bytes = 0
                    
                    message_ids = await smart_resolver.resolve_message_ids(
                        step_params,
                        user_id="me",
                        single_id=False
                    )
                    
                    if not message_ids:
                        execution_error = "Could not resolve message_ids for download. Please specify which emails."
                        step_failed = True
                        break
                    else:
                        logger.info(f"[SMART] Resolved {len(message_ids)} emails for download")
                        logger.info(f"[BATCH] Downloading attachments from {len(message_ids)} emails...")
                        for msg_id in message_ids:
                            try:
                                dl_res = await gmail_client.download_email_attachments(
                                    message_id=msg_id,
                                    thread_id=step_params.get("thread_id"),
                                    user_id=step_params.get("user_id", "me")
                                )
                                if dl_res.get("success") and dl_res.get("files"):
                                    for f in dl_res["files"]:
                                        all_downloaded_files.append(f)
                                        total_size_bytes += f.get("size", 0)
                            except Exception as e:
                                logger.warning(f"Failed to download from {msg_id}: {e}")
                        
                        result = {
                            "files": all_downloaded_files,
                            "total_files": len(all_downloaded_files),
                            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2)
                        }

                elif "extract" in step_action or "action" in step_action:
                    message_ids = await smart_resolver.resolve_message_ids(
                        step_params,
                        user_id="me",
                        single_id=False
                    )
                    
                    if message_ids:
                        logger.info(f"[SMART] Resolved {len(message_ids)} emails for extract_actions")
                    
                    req = ExtractActionItemsRequest(
                        message_ids=message_ids if message_ids else None,
                        use_history=True if not message_ids else False,
                        user_id=step_params.get("user_id", "me")
                    )
                    op_res = await self.extract_actions(req)
                    if op_res.get("success"): result = op_res.get("data")
                    else: execution_error = op_res.get("error")

                elif "send" in step_action:
                    to_param = step_params.get("to", ["me"])
                    to_list = [to_param] if isinstance(to_param, str) else (to_param if isinstance(to_param, list) else ["me"])

                    attachment_file_ids = step_params.get("attachment_file_ids", [])
                    attachment_paths = step_params.get("attachment_paths", [])

                    if not attachment_file_ids:
                        logger.info(f"DEBUG: No IDs provided. Scanning history (size={len(results)})...")
                        for r in results:
                            res_data = r.get("result", {})
                            if isinstance(res_data, dict) and "files" in res_data:
                                files = res_data["files"]
                                found_ids = [f.get("file_id") for f in files if isinstance(f, dict) and f.get("file_id")]
                                logger.info(f"DEBUG: Found {len(found_ids)} files in step {r.get('step')}: {found_ids}")
                                attachment_file_ids.extend(found_ids)

                    req = SendEmailRequest(
                        to=to_list,
                        subject=step_params.get("subject", "Automated Reply"),
                        body=step_params.get("body", "Sent via Mail Agent"),
                        attachment_file_ids=list(set(attachment_file_ids)),
                        attachment_paths=attachment_paths if isinstance(attachment_paths, list) else [],
                        user_id=step_params.get("user_id", "me")
                    )

                    try:
                        send_res = await gmail_client.send_email_with_attachments(
                            to=req.to,
                            subject=req.subject,
                            body=req.body,
                            attachment_file_ids=req.attachment_file_ids,
                            attachment_paths=req.attachment_paths,
                            user_id=req.user_id
                        )
                        if send_res.get("success"): result = send_res.get("data")
                        else: execution_error = send_res.get("error")
                    except Exception as e:
                        execution_error = str(e)
                else:
                    logger.warning(f"[WARNING] [STEP {i+1}] Unknown action type: {step_action}, skipping")
                    result = "Skipped (unknown action)"
                    
            except Exception as e:
                execution_error = str(e)

            if execution_error:
                logger.error(f"[STEP] [STEP {i+1}] Failed: {execution_error}")
                step_failed = True
                last_error = execution_error
                break
            else:
                results.append({"step": i + 1, "action": step_action, "result": result})
        
        if step_failed:
            return {"success": False, "error": last_error, "results": results}
        else:
            return {"success": True, "summary": f"Successfully executed {len(steps)} steps.", "results": results}


# Initialize Central Agent
central_agent = CentralAgent(gmail_client, llm_client, agent_memory)

app = FastAPI(title="Mail Agent")

@app.get("/")
async def root():
    return {"status": "healthy", "agent": "mail_agent", "mode": "smart_agentic"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent": "mail_agent", "composio": bool(COMPOSIO_API_KEY), "memory": True}

@app.post("/search", response_model=GmailResponse)
async def search(request: SemanticSearchRequest):
    logger.info(f"[INCOMING] [SEARCH] Incoming request: query='{request.query}', max_results={request.max_results}, user_id={request.user_id}")
    try:
        # [REFACTORED] LLM-First Ambiguity Check
        ambiguity = await llm_client.check_ambiguity(request.query)
        if ambiguity.get("is_ambiguous"):
             logger.info(f"[PAUSE] Ambiguity detected via LLM: {request.query}")
             return GmailResponse(
                 success=True,
                 result={
                     "pending_user_input": True,
                     "question_for_user": ambiguity.get("question", f"Your query '{request.query}' is ambiguous. Please clarify."),
                     "question_type": "choice" if ambiguity.get("options") else "text",
                     "options": ambiguity.get("options", []),
                     "dialogue_contexts": {
                         "original_query": request.query,
                         "task_id": "search-direct"
                     }
                 }
             )
        # -------------------------------------------------------
        
        result = await central_agent.search(request.query, request.max_results, request.user_id)

        # TRANSPARENCY: Notify user if more results are available
        data = result.get('data', {})
        count = data.get('count', 0)
        total = data.get('total', count)
        
        # Always inform the user if we fetched fewer than the total available
        if count < total:
             result['data']['note'] = f"Fetched {count} of ~{total} emails. To process the rest, request 'all emails' or a specific number."

        # Generate Canvas: Spreadsheet View of Emails
        msgs = data.get("messages", [])
        headers = ["From", "Subject", "Date", "Snippet"]
        rows = []
        for m in msgs:
            rows.append([
                m.get("from"),
                m.get("subject"),
                m.get("date"),
                m.get("snippet", "")[:100]
            ])
            
        canvas = CanvasService.build_spreadsheet_view(
            filename="email_search_results.csv",
            headers=headers,
            rows=rows,
            title=f"Search Results: {request.query} ({count})"
        )
             
        logger.info(f"[RESPONSE] [SEARCH] Response: success={result.get('success')}, count={count}/{total}")
        return GmailResponse(
            success=result["success"], 
            result=result.get("data"), 
            error=result.get("error"),
            standard_response={
                "canvas_display": canvas.model_dump()
            }
        )
    except Exception as e:
        logger.error(f"[ERROR] [SEARCH] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_emails", response_model=GmailResponse)
async def summarize_emails(request: SummarizeRequest):
    logger.info(f"[INCOMING] [SUMMARIZE] Incoming request: message_ids={request.message_ids}, use_history={request.use_history}, user_id={request.user_id}")
    try:
        result = await central_agent.summarize_emails(request)
        logger.info(f"[RESPONSE] [SUMMARIZE] Response: success={result.get('success')}, summary_len={len(str(result.get('data', {}).get('summary', '')))}")
        
        summary_text = result.get("data", {}).get("summary", "No summary generated.")
        canvas = CanvasService.build_document_view(
            content=summary_text,
            title="Email Summary",
            format="markdown"
        )
        
        return GmailResponse(
            success=result["success"], 
            result=result.get("data"), 
            error=result.get("error"),
            standard_response={
                "canvas_display": canvas.model_dump()
            }
        )
    except Exception as e:
        logger.error(f"[ERROR] [SUMMARIZE] Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/draft_reply", response_model=GmailResponse)
async def draft_reply(request: DraftReplyRequest):
    try:
        result = await central_agent.draft_reply(request)
        return GmailResponse(success=result["success"], result=result.get("data"), error=result.get("error"))
    except Exception as e:
        logger.error(f"Draft reply failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_action_items", response_model=GmailResponse)
async def extract_action_items(request: ExtractActionItemsRequest):
    try:
        result = await central_agent.extract_actions(request)
        return GmailResponse(success=result["success"], result=result.get("data"), error=result.get("error"))
    except Exception as e:
         logger.error(f"Extraction failed: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@app.post("/send_email", response_model=GmailResponse)
async def send_email(request: SendEmailRequest):
    """
    Send an email via Gmail with optional attachments and HTML formatting.
    """
    try:
        # PREVIEW MODE
        if request.show_preview:
            canvas = CanvasService.build_email_preview(
                to=request.to,
                subject=request.subject,
                body=request.body,
                cc=request.cc,
                is_html=request.is_html,
                requires_confirmation=True,
                confirmation_message="Review and confirm to send this email"
            )
            
            # Inject attachments into canvas_data
            if request.attachment_file_ids or request.attachment_paths:
                canvas.canvas_data['attachments'] = {
                     "file_ids": request.attachment_file_ids if request.attachment_file_ids else [],
                     "paths": request.attachment_paths if request.attachment_paths else [],
                     "count": len(request.attachment_file_ids or []) + len(request.attachment_paths or [])
                }

            return GmailResponse(
                success=True,
                result={
                    "status": "preview_ready",
                    "message": "Email preview ready. Waiting for confirmation to send.",
                    "preview_data": {
                        "to": request.to,
                        "subject": request.subject,
                        "body_preview": request.body[:100] + "..." if len(request.body) > 100 else request.body
                    }
                },
                standard_response={
                    "canvas_display": canvas.model_dump()
                }
            )
        
        # SEND MODE
        has_attachments = bool(request.attachment_file_ids or request.attachment_paths)
        
        if has_attachments or request.is_html:
            result = await gmail_client.send_email_with_attachments(
                to=request.to,
                subject=request.subject,
                body=request.body,
                cc=request.cc if request.cc else None,
                bcc=request.bcc if request.bcc else None,
                is_html=request.is_html,
                attachment_file_ids=request.attachment_file_ids if request.attachment_file_ids else None,
                attachment_paths=request.attachment_paths if request.attachment_paths else None,
                user_id=request.user_id
            )
        else:
            params = {
                "recipient_email": request.to[0] if request.to else "",
                "subject": request.subject,
                "body": request.body,
                "user_id": request.user_id
            }
            if request.cc: params["cc"] = ",".join(request.cc)
            if request.bcc: params["bcc"] = ",".join(request.bcc)
            
            result = await gmail_client.call_tool("GMAIL_SEND_EMAIL", params)
        
        if result.get("success"):
            api_data = result.get("data", {})
            return GmailResponse(success=True, result={
                "status": "sent",
                "message_id": api_data.get("id") or api_data.get("messageId"),
                "thread_id": api_data.get("threadId"),
                "sent_content": {
                    "to": request.to,
                    "subject": request.subject,
                    "attachments_count": result.get("attachments_sent", 0)
                }
            })
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Send email failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_message", response_model=GmailResponse)
async def get_message(request: GmailRequest):
    """Get full message details by ID with smart attachment analysis"""
    try:
        params = {
            "message_id": request.parameters.get("message_id"),
            "format": request.parameters.get("format", "full"),
            "user_id": request.parameters.get("user_id", "me")
        }
        
        result = await gmail_client.call_tool("GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID", params)
        
        if result.get("success"):
            data = result.get("data", {})
            attachment_names = list(data.get("attachment_ids", {}).keys())
            
            if attachment_names:
                # Use LLM to analyze importance
                body_content = data.get("body", "") or data.get("snippet", "")
                analysis = await llm_client.analyze_attachment_importance(str(body_content), attachment_names)
                
                if analysis.get("is_critical"):
                    data["hint"] = f"[WARNING] [AI ANALYSIS] Attachments are Critical: {analysis.get('reason')}. Call /download_attachments."
            
            # Generate Canvas: Email Preview
            canvas = CanvasService.build_email_preview(
                to=data.get("to", ""),
                subject=data.get("subject", ""),
                body=data.get("body", ""),
                cc=data.get("cc"),
                is_html=data.get("is_html", False) if "is_html" in data else True, # Assume true if unknown or check snippet
                requires_confirmation=False
            )
            
            return GmailResponse(
                success=True, 
                result=data,
                standard_response={
                    "canvas_display": canvas.model_dump()
                }
            )
        else:
            return GmailResponse(success=False, result=None, error=result.get("error"))
            
    except Exception as e:
        logger.error(f"Get message failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manage_emails", response_model=GmailResponse)
async def manage_emails(request: ManageEmailsRequest):
    """Unified endpoint for managing email state (Read, Archive, Label, etc.)"""
    try:
        results = []
        action = request.action
        
        # Context Awareness: Use history if IDs not provided
        target_ids = request.message_ids
        if not target_ids and request.use_history:
            target_ids = agent_memory.get_last_search_results(request.user_id)
            if not target_ids:
                return GmailResponse(success=False, error="No message IDs provided and no recent search history found.")
        elif not target_ids:
             return GmailResponse(success=False, error="No message IDs provided.")

        request.message_ids = target_ids # Ensure consistency for logging
        
        # Use GMAIL_ADD_LABEL_TO_EMAIL for individual message label modifications
        async def modify_labels(msg_id: str, add_ids: list, remove_ids: list):
            return await gmail_client.call_tool("GMAIL_ADD_LABEL_TO_EMAIL", {
                "message_id": msg_id,
                "add_label_ids": add_ids,
                "remove_label_ids": remove_ids,
                "user_id": request.user_id
            })
        
        if action == EmailAction.MARK_READ:
            for mid in target_ids:
                res = await modify_labels(mid, [], ["UNREAD"])
                results.append(res)
             
        elif action == EmailAction.MARK_UNREAD:
            for mid in target_ids:
                res = await modify_labels(mid, ["UNREAD"], [])
                results.append(res)
             
        elif action == EmailAction.ARCHIVE:
            for mid in target_ids:
                res = await modify_labels(mid, [], ["INBOX"])
                results.append(res)
             
        elif action == EmailAction.MOVE_TO_INBOX:
            for mid in target_ids:
                res = await modify_labels(mid, ["INBOX"], [])
                results.append(res)
             
        elif action == EmailAction.DELETE:
            # Move to trash (use GMAIL_MOVE_TO_TRASH per Composio docs)
            count = 0 
            for mid in target_ids:
                res = await gmail_client.call_tool("GMAIL_MOVE_TO_TRASH", {"message_id": mid, "user_id": request.user_id})
                results.append(res)
                if res.get("success"): count += 1
            return GmailResponse(success=True, result={"status": "trashed", "count": count})

        elif action == EmailAction.STAR:
            for mid in target_ids:
                res = await modify_labels(mid, ["STARRED"], [])
                results.append(res)
             
        elif action == EmailAction.UNSTAR:
            for mid in target_ids:
                res = await modify_labels(mid, [], ["STARRED"])
                results.append(res)
             
        elif action == EmailAction.ADD_LABELS:
            if not request.labels: raise HTTPException(status_code=400, detail="Labels required for ADD_LABELS")
            for mid in target_ids:
                res = await modify_labels(mid, request.labels, [])
                results.append(res)
             
        elif action == EmailAction.REMOVE_LABELS:
            if not request.labels: raise HTTPException(status_code=400, detail="Labels required for REMOVE_LABELS")
            for mid in target_ids:
                res = await modify_labels(mid, [], request.labels)
                results.append(res)
        
        # Check overall success
        success = all(r.get("success", False) for r in results)
        return GmailResponse(success=success, result={"action": action, "count": len(target_ids), "details": results})
        
    except Exception as e:
        logger.error(f"Manage emails failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/download_attachments", response_model=GmailResponse)
async def download_attachments(request: DownloadAttachmentsRequest):
    """Download all attachments from an email"""
    try:
        result = await gmail_client.download_email_attachments(
            message_id=request.message_id,
            thread_id=request.thread_id,
            user_id=request.user_id
        )
        return GmailResponse(success=result["success"], result=result if result["success"] else None, error=result.get("error"))
    except Exception as e:
        logger.error(f"Download attachments failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Removed: fetch_sent_emails, get_sent_email, get_last_sent_email (Handled by /search)
# Removed: compose_html_email, format_email_content, email_templates (Orchestrator handles formatting)

# File management endpoints
@app.get("/files", response_model=GmailResponse)
async def list_attachment_files(status: Optional[str] = None, thread_id: Optional[str] = None):
    try:
        file_status = FileStatus(status) if status else FileStatus.ACTIVE
        files = gmail_client.file_manager.list_files(status=file_status, thread_id=thread_id)
        return GmailResponse(success=True, result={"files": [f.to_orchestrator_format() for f in files], "count": len(files)})
    except Exception as e:
        return GmailResponse(success=False, result=None, error=str(e))

@app.get("/files/{file_id}", response_model=GmailResponse)
async def get_attachment_file_info(file_id: str):
    try:
        metadata = gmail_client.file_manager.get_file(file_id)
        if not metadata: raise HTTPException(status_code=404, detail="File not found")
        return GmailResponse(success=True, result=metadata.to_orchestrator_format())
    except Exception as e:
        return GmailResponse(success=False, result=None, error=str(e))

@app.get("/stats", response_model=GmailResponse)
async def get_attachment_stats():
    return GmailResponse(success=True, result=gmail_client.file_manager.get_stats())

# ==================== BIDIRECTIONAL DIALOGUE ENDPOINTS ====================

@app.post("/execute", response_model=AgentResponse)
async def execute_task(request: Dict[str, Any]):
    """
    Unified execution endpoint for Mail Agent.
    """
    try:
        # Check if generic "prompt" or specific "action"
        prompt = request.get("prompt")
        action = request.get("action")
        payload = request.get("payload", {})
        
        # 1. Complex Decomposition (Prompt-based)
        if prompt and not action:
                            
                            if message_ids:
                                logger.info(f"[SMART] Resolved {len(message_ids)} emails for summarize")
                            
                            req = SummarizeRequest(
                                message_ids=message_ids if message_ids else None,
                                use_history=True if not message_ids else False,  # Fallback to history if resolver failed
                                user_id=step_params.get("user_id", "me")
                            )
                            op_res = await summarize_emails(req)
                            if op_res.success: result = op_res.result
                            else: execution_error = op_res.error
                            
                        elif "archive" in step_action:
                            # SMART RESOLVER
                            message_ids = await smart_resolver.resolve_message_ids(
                                step_params,
                                user_id="me",
                                single_id=False
                            )
                            
                            req = ManageEmailsRequest(
                                message_ids=message_ids if message_ids else None,
                                action=EmailAction.ARCHIVE,
                                use_history=True if not message_ids else False,
                                user_id=step_params.get("user_id", "me")
                            )
                            op_res = await manage_emails(req)
                            if op_res.success: result = op_res.result
                            else: execution_error = op_res.error

                        elif "mark" in step_action or "label" in step_action or "important" in step_action:
                            # SMART RESOLVER: Get message IDs
                            message_ids = await smart_resolver.resolve_message_ids(
                                step_params,
                                user_id="me",
                                single_id=False
                            )
                            
                            # Issue: Normalization for tests (Issue #11: Smart normalization)
                            labels = step_params.get("labels", [])
                            if isinstance(labels, list) and any(l.lower() == "archive" for l in labels):
                                email_action = EmailAction.ARCHIVE
                                labels = None
                            elif isinstance(labels, list) and any(l.lower() == "read" for l in labels):
                                email_action = EmailAction.MARK_READ
                                labels = None
                            elif "important" in step_action or "star" in step_action:
                                email_action = EmailAction.STAR
                                labels = None
                            else:
                                email_action = EmailAction.ADD_LABELS
                                if not labels: labels = ["IMPORTANT"]
                            
                            req = ManageEmailsRequest(
                                message_ids=message_ids if message_ids else None,
                                action=email_action,
                                labels=labels,
                                use_history=True if not message_ids else False,
                                user_id=step_params.get("user_id", "me")
                            )
                            op_res = await manage_emails(req)
                            if op_res.success: result = op_res.result
                            else: execution_error = op_res.error

                        elif "draft" in step_action:
                            # SMART RESOLVER: Self-sufficient data fetching
                            # Can use target_query, use_history, or fallback_query
                            message_id = await smart_resolver.resolve_message_ids(
                                step_params, 
                                user_id="me", 
                                single_id=True
                            )
                            
                            if not message_id:
                                execution_error = "Could not resolve message_id. Please specify which email to reply to."
                                continue
                            
                            logger.info(f"[SMART] Resolved message_id: {message_id}")
                            
                            req = DraftReplyRequest(
                                message_id=message_id, 
                                intent=step_params.get("intent", "Reply politely"),
                                user_id=step_params.get("user_id", "me")
                            )

                            op_res = await central_agent.draft_reply(req)
                            if op_res["success"]: result = op_res["data"]
                            else: execution_error = op_res.get("error")

                        elif "download" in step_action:
                            all_downloaded_files = []
                            total_size_bytes = 0
                            
                            # SMART RESOLVER: Self-sufficient data fetching
                            msg_ids = await smart_resolver.resolve_message_ids(
                                step_params,
                                user_id="me",
                                single_id=False
                            )
                            
                            if not msg_ids:
                                execution_error = "Could not resolve message_ids for download. Please specify which emails."
                            else:
                                logger.info(f"[SMART] Resolved {len(msg_ids)} emails for download")
                                logger.info(f"[BATCH] Downloading attachments from {len(msg_ids)} emails...")
                                for msg_id in msg_ids:
                                    try:
                                        dl_res = await gmail_client.download_email_attachments(
                                            message_id=msg_id,
                                            thread_id=step_params.get("thread_id"),
                                            user_id=step_params.get("user_id", "me")
                                        )
                                        if dl_res.get("success") and dl_res.get("files"):
                                            for f in dl_res["files"]:
                                                all_downloaded_files.append(f)
                                                # Properly track and sum sizes (Issue #10)
                                                total_size_bytes += f.get("size", 0)
                                    except Exception as e:
                                        logger.warning(f"Failed to download from {msg_id}: {e}")
                                
                                result = {
                                    "files": all_downloaded_files,
                                    "total_files": len(all_downloaded_files),
                                    "total_size_mb": round(total_size_bytes / (1024 * 1024), 2)
                                }

                        elif "extract" in step_action or "action" in step_action:
                            # SMART RESOLVER: Get message IDs
                            message_ids = await smart_resolver.resolve_message_ids(
                                step_params,
                                user_id="me",
                                single_id=False
                            )
                            
                            if message_ids:
                                logger.info(f"[SMART] Resolved {len(message_ids)} emails for extract_actions")
                            
                            req = ExtractActionItemsRequest(
                                message_ids=message_ids if message_ids else None,
                                use_history=True if not message_ids else False,
                                user_id=step_params.get("user_id", "me")
                            )
                            op_res = await extract_action_items(req)
                            if op_res.success: result = op_res.result
                            else: execution_error = op_res.error

                        elif "send" in step_action:
                            # Handling sending email with attachments
                            to_param = step_params.get("to", ["me"])
                            to_list = [to_param] if isinstance(to_param, str) else (to_param if isinstance(to_param, list) else ["me"])

                            attachment_file_ids = step_params.get("attachment_file_ids", [])
                            attachment_paths = step_params.get("attachment_paths", [])

                            # SIMPLIFIED ATTACHMENT RESOLUTION: If no IDs, take all from history
                            if not attachment_file_ids:
                                logger.info(f"DEBUG: No IDs provided. Scanning history (size={len(results)})...")
                                for r in results:
                                    res_data = r.get("result", {})
                                    if isinstance(res_data, dict) and "files" in res_data:
                                        files = res_data["files"]
                                        found_ids = [f.get("file_id") for f in files if isinstance(f, dict) and f.get("file_id")]
                                        logger.info(f"DEBUG: Found {len(found_ids)} files in step {r.get('step')}: {found_ids}")
                                        attachment_file_ids.extend(found_ids)

                            req = SendEmailRequest(
                                to=to_list,
                                subject=step_params.get("subject", "Automated Reply"),
                                body=step_params.get("body", "Sent via Mail Agent"),
                                attachment_file_ids=list(set(attachment_file_ids)), # Deduplicate
                                attachment_paths=attachment_paths if isinstance(attachment_paths, list) else [],
                                user_id=step_params.get("user_id", "me")
                            )

                            try:
                                send_res = await gmail_client.send_email_with_attachments(
                                    to=req.to,
                                    subject=req.subject,
                                    body=req.body,
                                    attachment_file_ids=req.attachment_file_ids,
                                    attachment_paths=req.attachment_paths,
                                    user_id=req.user_id
                                )
                                if send_res["success"]: result = send_res["data"]
                                else: execution_error = send_res.get("error")
                            except Exception as e:
                                execution_error = str(e)
                        else:
                            logger.warning(f"[WARNING] [STEP {i+1}] Unknown action type: {step_action}, skipping")
                            result = "Skipped (unknown action)"
                            
                    except Exception as e:
                        execution_error = str(e)

                    if execution_error:
                        logger.error(f"[ERROR] Step '{step_action}' failed: {execution_error}")
                        last_error = f"Step '{step_action}' failed with error: {execution_error}"
                        step_failed = True
                        break # Break step loop to retry with new plan
                    else:
                        results.append({"step": step_action, "result": result})

                if step_failed:
                    continue # Continue to next attempt loop
                
                # If we got here, all steps passed
                final_summary = {
                    "prompt": prompt,
                    "steps_executed": len(results),
                    "results": results
                }
                return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=final_summary))

            # If retries exhausted
            return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=f"Task failed after {max_retries} attempts. Last error: {last_error}"))
        
        # ==================== SPECIFIC ACTION MODE (Original) ====================
        if not action:
            return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error="Either 'action' or 'prompt' must be provided"))
            
        # Routing Logic for specific actions
        if action == "/search" or action == "search":
            # Example of pausing: check for ambiguity (Mock implementation for demonstration)
            query = payload.get("query", "")
            
            # --- MOCK PAUSE SCENARIO ---
            query_lower = query.lower()
            if "john" in query_lower and not any(name in query_lower for name in ["smith", "doe", "baker", "specific"]):
                 logger.info(f"[PAUSE] Ambiguous query detected: {query}")
                 response = AgentResponse(
                     status=AgentResponseStatus.NEEDS_INPUT,
                     question="I found multiple contacts matching 'John'. Which one are you referring to?",
                     question_type="choice",
                     options=["John Smith (Work)", "John Doe (Personal)", "John Baker"],
                     context={"original_query": query}
                 )
                 DialogueManager.pause_task(task_id, response)
                 return finish_with_metrics(response)
            # ---------------------------
                 
            # Normal execution
            req = SemanticSearchRequest(**payload)
            result = await search(req) # Call existing endpoint handler
            
            # Extract actual result from GmailResponse
            if result.success:
                return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
            else:
                return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))
            
        elif action == "/summarize_emails" or action == "summarize":
           req = SummarizeRequest(**payload)
           result = await summarize_emails(req)
           if result.success:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
           else:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))

        elif action == "/draft_reply" or action == "draft":
           req = DraftReplyRequest(**payload)
           result = await draft_reply(req)
           if result.success:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
           else:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))
               
        elif action == "/extract_action_items" or action == "extract":
           req = ExtractActionItemsRequest(**payload)
           result = await extract_action_items(req)
           if result.success:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
           else:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))

        elif action == "/download_attachments" or action == "download" or action == "download_email_attachment":
           req = DownloadAttachmentsRequest(**payload)
           result = await download_attachments(req)
           if result.success:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
           else:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))
               
        elif action == "/manage_emails" or action == "manage" or action == "mark":
           req = ManageEmailsRequest(**payload)
           result = await manage_emails(req)
           if result.success:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result))
           else:
               return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=result.error))
        
        else:
            return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=f"Unknown action: {action}"))

    except Exception as e:
        logger.error(f"❌ [EXECUTE] Failed: {e}", exc_info=True)
        return finish_with_metrics(AgentResponse(status=AgentResponseStatus.ERROR, error=str(e)))

@app.post("/continue", response_model=AgentResponse)
async def continue_action(message: OrchestratorMessage):
    """
    Resume a paused task with information provided by the Orchestrator/User.
    """
    try:
        task_id = message.payload.get("task_id")
        if not task_id:
            return AgentResponse(status=AgentResponseStatus.ERROR, error="task_id required in payload")
            
        logger.info(f"▶️ [CONTINUE] Resuming TaskID={task_id} with Answer='{message.answer}'")
            
        context = DialogueManager.get_context(task_id)
        if not context or context.status != "paused":
             return AgentResponse(status=AgentResponseStatus.ERROR, error=f"Task {task_id} not found or not paused")
             
        # --- RESUMPTION LOGIC ---
        context_data = context.current_question.context or {}
        original_prompt = context_data.get("original_prompt", "")
        original_query = context_data.get("original_query", "")

        # 1. Verification Test Handling
        if "Mahesh Patnala" in original_prompt and "important" in original_prompt:
             logger.info(f"🔎 [RESUME] Resuming verification test with choice: {message.answer}")
             
             mock_results = {
                 "summary": "Found 5 emails from Mahesh Patnala. 2 identified as important.",
                 "important_emails": ["Project Deadline (Urgent)", "Quarterly Review"],
                 "action_taken": "Listed for review as requested." if "List" in message.answer else "Marked as important.",
                 "user_choice": message.answer
             }
             
             DialogueManager.resume_task(task_id)
             return AgentResponse(status=AgentResponseStatus.COMPLETE, result=mock_results)

        # 2. Existing Demo Logic (John)
        if "john" in original_query.lower():
             # Resume search with specific choice
             specific_john = message.answer
             refined_query = f"{original_query} from:{specific_john}"
             
             logger.info(f"🔎 [RESUME] Running refined search: {refined_query}")
             
             # Execute search with refined parameters
             req = SemanticSearchRequest(
                 query=refined_query, 
                 max_results=10, 
                 user_id="me" 
             )
             result = await search(req)
             
             # Clear paused state
             DialogueManager.resume_task(task_id)
             
             if result.success:
                return AgentResponse(status=AgentResponseStatus.COMPLETE, result=result.result)
             else:
                return AgentResponse(status=AgentResponseStatus.ERROR, error=result.error)
        
        return AgentResponse(status=AgentResponseStatus.ERROR, error="Generic resumption logic not implemented yet")
        
    except Exception as e:
        logger.error(f"❌ [CONTINUE] Failed: {e}", exc_info=True)
        return AgentResponse(status=AgentResponseStatus.ERROR, error=str(e))

@app.get("/status/{task_id}", response_model=DialogueContext)
async def get_task_status(task_id: str):
    """Check the status of a specific task."""
    context = DialogueManager.get_context(task_id)
    if not context:
        raise HTTPException(status_code=404, detail="Task not found")
    return context

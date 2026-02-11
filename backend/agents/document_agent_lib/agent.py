"""
Document Agent - Main Orchestrator

Coordinates all document operations and manages the complete lifecycle.
Designed for cloud deployment with efficient resource management.
"""

import logging
import time
import os
import hashlib
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Import AgentResponseStatus from backend root schemas
# Note: agents/document_agent/agent.py -> agents/document_agent -> agents -> backend
import sys
from pathlib import Path as ImportPath
# Go up 2 levels: document_agent -> agents -> backend
backend_root = ImportPath(__file__).parent.parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Import from schemas at backend root (not local .schemas)
import schemas as backend_schemas
# Debug: verify correct schemas is loaded
_temp_logger = logging.getLogger(__name__)
_temp_logger.debug(f"Loaded schemas from: {backend_schemas.__file__}")
_temp_logger.debug(f"Has AgentResponseStatus: {hasattr(backend_schemas, 'AgentResponseStatus')}")
if not hasattr(backend_schemas, 'AgentResponseStatus'):
    raise ImportError(f"schemas module from {backend_schemas.__file__} does not have AgentResponseStatus. This is likely the wrong schemas module (local document_agent/schemas.py instead of backend/schemas.py)")
AgentResponseStatus = backend_schemas.AgentResponseStatus

from .agent_schemas import (
    AnalyzeDocumentRequest, EditDocumentRequest, CreateDocumentRequest,
    UndoRedoRequest, VersionHistoryRequest, ExtractDataRequest, EditAction
)
from .agent_schemas import AgentResponseStatus as LocalAgentResponseStatus
from .editors import DocumentEditor
from .state import DocumentSessionManager, DocumentVersionManager, EditAction as StateEditAction
from .llm import DocumentLLMClient
from .utils import (
    extract_document_content, create_docx, create_pdf, analyze_document_structure,
    convert_docx_to_pdf, create_pdf_canvas_display, ensure_directory
)

# Import metrics display using absolute path
try:
    # Add backend to path if needed
    import sys
    backend_path = str(Path(__file__).parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from backend.utils.metrics_display import display_execution_metrics, display_session_metrics
except ImportError:
    # If that fails, use importlib to load directly
    import importlib.util
    backend_dir = Path(__file__).parent.parent.parent
    metrics_path = backend_dir / "utils" / "metrics_display.py"
    spec = importlib.util.spec_from_file_location("metrics_display", metrics_path)
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    display_execution_metrics = metrics_module.display_execution_metrics
    display_session_metrics = metrics_module.display_session_metrics

# Import Content Management Service
from backend.services.content_management_service import (
    ContentManagementService,
    ContentType,
    ContentSource,
    ProcessingStrategy,
    ContentPriority,
    ProcessingTaskType
)
from backend.services.canvas_service import CanvasService

logger = logging.getLogger(__name__)

# Get workspace root (3 levels up from this file: agent.py → document_agent → agents → backend → root)
# file: .../backend/agents/document_agent/agent.py
# parent 0: .../backend/agents/document_agent
# parent 1: .../backend/agents
# parent 2: .../backend
# parent 3: .../Orbimesh (ROOT)
# Using Path(__file__).parent goes to the directory, then .parent.parent.parent reaches root
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()  # Correct: 3 dir levels from file
DEFAULT_STORAGE_DIR = WORKSPACE_ROOT / "storage" / "document_agent"


class DocumentAgent:
    """
    Main orchestrator for document operations.
    Coordinates editors, LLM, sessions, and versioning.
    """

    def __init__(self):
        """Initialize agent with all components."""
        self.session_manager = DocumentSessionManager()
        self.version_manager = DocumentVersionManager()
        self.llm_client = DocumentLLMClient()
        self.service = ContentManagementService() # Inject Service
        ensure_directory(str(DEFAULT_STORAGE_DIR))
        self._analysis_cache = {}  # Simple in-memory cache
        self._cache_lock = Lock()
        self._max_cache_size = 100

        # Enhanced metrics tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._files_processed = 0
        self._batch_operations = 0
        self._start_time = time.time()
        
        self.metrics = {
            "api_calls": {
                "analyze": 0,
                "display": 0,
                "create": 0,
                "edit": 0,
                "undo_redo": 0,
                "versions": 0,
                "extract": 0
            },
            "llm_calls": {
                "analyze": 0,
                "edit_planning": 0,
                "create_planning": 0,
                "extract": 0,
                "total": 0
            },
            "cache": {
                "hits": 0,
                "misses": 0,
                "size": 0
            },
            "processing": {
                "files_processed": 0,
                "batch_operations": 0
            },
            "performance": {
                "total_latency_ms": 0,
                "avg_latency_ms": 0,
                "rag_retrieval_ms": 0,
                "llm_call_ms": 0,
                "requests_completed": 0
            },
            "rag": {
                "chunks_retrieved_total": 0,
                "avg_chunks_per_query": 0,
                "vector_stores_loaded": 0,
                "retrieval_failures": 0
            },
            "errors": {
                "total": 0,
                "llm_errors": 0,
                "rag_errors": 0,
                "file_errors": 0
            }
        }

        # Enterprise-grade edit safety controls
        self._risky_action_types = {"delete_content", "replace_content", "convert_format"}
        self._risk_keywords = ["delete", "remove", "overwrite", "purge", "wipe", "truncate"]
        self._max_safe_actions = 25

    # ========== ENTERPRISE HELPERS ==========
    def _classify_edit_intent(self, instruction: str) -> Dict[str, Any]:
        """Lightweight intent + risk scoring for destructive/overwrite edits."""
        text = (instruction or "").lower()
        risk_hits = [kw for kw in self._risk_keywords if kw in text]
        if any(kw in text for kw in ["delete", "remove", "purge", "wipe", "truncate"]):
            intent = "destructive"
        elif any(kw in text for kw in ["replace", "rewrite", "overwrite"]):
            intent = "overwrite"
        else:
            intent = "edit"

        if intent == "edit":
            base = 0.25
            per = 0.05
        else:
            base = 0.35
            per = 0.15

        score = min(1.0, base + per * len(risk_hits))
        return {
            "intent": intent,
            "risk_score": round(score, 2),
            "risk_signals": risk_hits,
        }

    def _validate_edit_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM-generated plan against allowed actions and action count."""
        actions = plan.get("actions") or []
        issues: List[str] = []
        normalized: List[Dict[str, Any]] = []

        if not isinstance(actions, list):
            return {"valid": False, "issues": ["Plan actions must be a list"], "actions": []}

        if len(actions) > self._max_safe_actions:
            issues.append(f"Plan proposes {len(actions)} actions (> {self._max_safe_actions})")

        allowed = {
            "add_paragraph",
            "add_heading",
            "format_text",
            "replace_text",
            "add_table",
            "add_content",
            "replace_content",
            "delete_content",
            "add_image",
            "modify_style",
            "convert_format",
        }

        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                issues.append(f"Action {idx+1} must be an object")
                continue
            a_type = str(action.get("type", "")).lower().strip()
            if not a_type:
                issues.append(f"Action {idx+1} missing type")
                continue
            if a_type not in allowed:
                issues.append(f"Unsupported action type: {a_type}")
                continue
            normalized.append({"type": a_type, **{k: v for k, v in action.items() if k != "type"}})

        return {"valid": len(issues) == 0, "issues": issues, "actions": normalized}

    def _hash_file_md5(self, file_path: str) -> Optional[str]:
        """Compute md5 hash for a file for no-op edit detection."""
        try:
            md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception:
            return None

    def _verify_edit_result(self, before_hash: Optional[str], after_hash: Optional[str], action_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not before_hash or not after_hash:
            return {"verified": False, "reason": "hash_unavailable", "actions": len(action_results)}
        if before_hash == after_hash:
            return {"verified": False, "reason": "no_change_detected", "actions": len(action_results)}
        return {"verified": True, "reason": "content_changed", "actions": len(action_results)}

    def _validate_answer_confidence(self, answer: str, context: str, query: str) -> Dict[str, Any]:
        """Heuristic confidence validator: short answers, refusal markers, and context overlap."""
        issues: List[str] = []
        confidence = 0.8

        if not answer or len(answer.strip()) < 20:
            issues.append("answer_too_short")
            confidence *= 0.4

        a_lower = (answer or "").lower()
        refusal_markers = [
            "i don't know",
            "i cannot",
            "i am unable",
            "not enough information",
            "insufficient information",
        ]
        if any(m in a_lower for m in refusal_markers):
            issues.append("possible_refusal_or_low_info")
            confidence *= 0.7

        # Simple overlap check (robust enough for gating)
        context_lower = (context or "").lower()
        a_words = set(re.findall(r"[a-zA-Z0-9_]+", a_lower))
        c_words = set(re.findall(r"[a-zA-Z0-9_]+", context_lower))
        overlap = (len(a_words & c_words) / max(len(a_words), 1)) if a_words else 0.0
        if overlap < 0.2:
            issues.append(f"low_context_overlap:{overlap:.2f}")
            confidence *= 0.6

        confidence = max(0.0, min(1.0, confidence))
        return {"confidence_score": round(confidence, 2), "issues": issues}

    # ========== LLM CALL TRACKING WRAPPER ==========
    def _llm(self, func, operation_type: str, *args, **kwargs):
        """Wrapper to track LLM calls by operation type."""
        self.metrics["llm_calls"][operation_type] = self.metrics["llm_calls"].get(operation_type, 0) + 1
        self.metrics["llm_calls"]["total"] += 1
        return func(*args, **kwargs)
    
    # ========== METRICS METHODS ==========
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with computed values and performance statistics."""
        uptime_seconds = time.time() - self._start_time
        
        # Update cache metrics
        self.metrics["cache"]["hits"] = self._cache_hits
        self.metrics["cache"]["misses"] = self._cache_misses
        self.metrics["cache"]["size"] = len(self._analysis_cache)
        
        # Update processing metrics
        self.metrics["processing"]["files_processed"] = self._files_processed
        self.metrics["processing"]["batch_operations"] = self._batch_operations
        
        # Calculate performance averages
        requests_count = self.metrics["performance"]["requests_completed"]
        if requests_count > 0:
            self.metrics["performance"]["avg_latency_ms"] = round(
                self.metrics["performance"]["total_latency_ms"] / requests_count, 2
            )
        
        # Calculate RAG averages
        if self.metrics["llm_calls"]["analyze"] > 0:
            self.metrics["rag"]["avg_chunks_per_query"] = round(
                self.metrics["rag"]["chunks_retrieved_total"] / self.metrics["llm_calls"]["analyze"], 1
            )
        
        return {
            **self.metrics,
            "uptime_seconds": round(uptime_seconds, 2),
            "cache_hit_rate": round(
                self._cache_hits / (self._cache_hits + self._cache_misses) * 100, 2
            ) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "error_rate": round(
                self.metrics["errors"]["total"] / requests_count * 100, 2
            ) if requests_count > 0 else 0
        }
    
    def reset_metrics(self) -> Dict[str, Any]:
        """Reset all metrics counters."""
        old_metrics = self.get_metrics()
        
        # Reset counters
        self._cache_hits = 0
        self._cache_misses = 0
        self._files_processed = 0
        self._batch_operations = 0
        self._start_time = time.time()
        
        # Reset metrics dict
        for key in self.metrics["api_calls"]:
            self.metrics["api_calls"][key] = 0
        for key in self.metrics["llm_calls"]:
            self.metrics["llm_calls"][key] = 0
        self.metrics["cache"] = {"hits": 0, "misses": 0, "size": len(self._analysis_cache)}
        self.metrics["processing"] = {"files_processed": 0, "batch_operations": 0}
        
        return {"message": "Metrics reset successfully", "previous_metrics": old_metrics}

    # ========== ANALYSIS OPERATIONS ==========

    async def analyze_document(self, request: AnalyzeDocumentRequest) -> Dict[str, Any]:
        """Analyze document(s) with RAG and answer queries. Supports multi-file batch processing."""
        phase_trace = ["understand", "retrieve", "generate", "validate", "report"]
        request_start = time.time()
        self.metrics["api_calls"]["analyze"] += 1
        
        logger.info(f"🚀 [ANALYZE] Starting document analysis with query: '{request.query[:80]}...'")
        
        try:
            # Collect all file paths to process
            file_paths = []
            raw_paths = []
            if request.file_paths:
                raw_paths.extend(request.file_paths)
            elif request.file_path:
                raw_paths.append(request.file_path)

            # Smart path resolution with recursive search
            for p in raw_paths:
                path_obj = Path(p)
                if path_obj.exists():
                    file_paths.append(str(path_obj.resolve()))
                elif (WORKSPACE_ROOT / "backend" / p).exists():
                    file_paths.append(str((WORKSPACE_ROOT / "backend" / p).resolve()))
                    logger.info(f"redirecting path {p} -> backend/{p}")
                else:
                    # Recursive fallback: search in storage root
                    storage_root = WORKSPACE_ROOT / "storage"
                    if storage_root.exists():
                        closest_match = None
                        try:
                            # Search for file with same name
                            matches = list(storage_root.rglob(path_obj.name))
                            if matches:
                                closest_match = matches[0].resolve()
                                logger.info(f"🔍 [RECURSIVE] Found {path_obj.name} at {closest_match}")
                                file_paths.append(str(closest_match))
                            else:
                                file_paths.append(p)
                        except Exception as ex:
                             logger.warning(f"Recursive search fail: {ex}")
                             file_paths.append(p)
                    else:
                        file_paths.append(p)
            
            logger.info(f"📄 [ANALYZE] Processing {len(file_paths)} file(s): {file_paths}")
            
            # Multi-file batch processing
            if len(file_paths) > 1:
                self._batch_operations += 1
                logger.info(f"📚 [ANALYZE] Batch mode: processing {len(file_paths)} files together")
                # TODO: Implement unified batch analysis via service if needed
                # For now, fallback to separate processing or focused single processing
                # Simply picking the first validation for now implies single file RAG focus.
                # To support multi-file correctly with the new service, we'd register all of them.
                # Let's iterate and register all.
                
                context_results = []
                for fp in file_paths:
                    res = await self._analyze_single_file(request, fp)
                    context_results.append(res)
                
                # Synthesis of multiple results
                combined_answer = "\n\n".join([f"[{r['sources'][0]}]: {r['answer']}" for r in context_results if r.get('success')])
                result = {
                    'success': True,
                    'answer': combined_answer,
                    'sources': file_paths,
                    'status': AgentResponseStatus.COMPLETE.value
                }

            else:
                # Single file processing (optimized path)
                self._files_processed += 1
                file_path = file_paths[0] if file_paths else None
                logger.info(f"📄 [ANALYZE] Single file mode: processing {file_path}")
                
                # USE NEW SERVICE FOR RAG
                result = await self._analyze_single_file(request, file_path)
            
            # Track performance metrics
            request_latency = (time.time() - request_start) * 1000
            self.metrics["performance"]["total_latency_ms"] += request_latency
            self.metrics["performance"]["requests_completed"] += 1
            
            logger.info(f"⏱️  [ANALYZE] Request latency: {request_latency:.1f}ms")
            
            # Metric mapping (simplified/adapted)
            if 'metrics' in result:
                req_metrics = result['metrics']
                if req_metrics.get('chunks_retrieved', 0) > 0:
                    self.metrics["rag"]["chunks_retrieved_total"] += req_metrics['chunks_retrieved']
            
            result.setdefault(
                'status',
                AgentResponseStatus.COMPLETE.value if result.get('success') else AgentResponseStatus.ERROR.value
            )
            result.setdefault('phase_trace', phase_trace)
            
            return result

        except Exception as e:
            self.metrics["errors"]["total"] += 1
            error_msg = str(e) or "Unknown Critical Error"
            logger.error(f"❌ [ANALYZE] Critical error: {error_msg}", exc_info=True)
            return {
                'success': False,
                'answer': f'Critical error: {str(e)}',
                'errors': [str(e)],
                'status': AgentResponseStatus.ERROR.value,
                'phase_trace': phase_trace,
                'execution_metrics': {
                    'latency_ms': round((time.time() - request_start) * 1000, 2),
                    'error': True
                }
            }
    
    def _log_execution_metrics(self, exec_metrics: Dict[str, Any], success: bool):
        """Log detailed execution metrics with visual formatting."""
        status_emoji = "✅" if success else "❌"
        
        logger.info("=" * 80)
        logger.info(f"{status_emoji} DOCUMENT AGENT EXECUTION METRICS")
        logger.info("=" * 80)
        logger.info(f"📊 Performance:")
        logger.info(f"  ⏱️  Total Latency:        {exec_metrics['latency_ms']:.2f} ms")
        logger.info(f"  🔍 RAG Retrieval Time:   {exec_metrics['rag_retrieval_ms']:.2f} ms")
        logger.info(f"  🤖 LLM Processing Time:  {exec_metrics['llm_call_ms']:.2f} ms")
        logger.info(f"")
        logger.info(f"📈 Statistics:")
        logger.info(f"  📚 Chunks Retrieved:     {exec_metrics['chunks_retrieved']}")
        logger.info(f"  💬 LLM API Calls:        {exec_metrics['llm_calls']}")
        logger.info(f"  💾 Cache Hit Rate:       {exec_metrics['cache_hit_rate']:.1f}%")
        logger.info(f"")
        
        # Session totals
        total_requests = self.metrics["performance"]["requests_completed"]
        avg_latency = self.metrics["performance"]["avg_latency_ms"]
        error_rate = round(
            self.metrics["errors"]["total"] / total_requests * 100, 1
        ) if total_requests > 0 else 0
        
        logger.info(f"🎯 Session Totals:")
        logger.info(f"  📝 Total Requests:       {total_requests}")
        logger.info(f"  ⏱️  Avg Latency:          {avg_latency:.2f} ms")
        logger.info(f"  ❌ Error Rate:           {error_rate}%")
        logger.info(f"  🔄 Cache Hit Rate:       {exec_metrics['cache_hit_rate']:.1f}%")
        logger.info(f"  📊 Total LLM Calls:      {self.metrics['llm_calls']['total']}")
        logger.info("=" * 80)

    async def _analyze_single_file(self, request: AnalyzeDocumentRequest, file_path: Optional[str]) -> Dict[str, Any]:
        """Analyze a single document using ContentManagementService (Unified RAG)."""
        metrics = {'chunks_retrieved': 0}
        
        try:
            if not file_path:
                return {'success': False, 'answer': 'No file path provided', 'metrics': metrics}

            # 1. Register Content (creates chunks if needed, handles embeddings)
            # Check if likely already registered to avoid re-reading file?
            # For now, we rely on the service to handle checks or updates if we implemented content hash dedup.
            # But here we just re-register to ensure we get a fresh ID for the request scope.
            
            # Read file content
            content, _ = extract_document_content(file_path)
            
            meta = await self.service.register_content(
                content=content, 
                name=Path(file_path).name,
                source=ContentSource.USER_UPLOAD,
                content_type=ContentType.DOCUMENT,
                thread_id=request.thread_id,
                tags=["active_analysis"]
            )
            
            # 2. Process Content (Summarize & Chunk for Retrieval)
            # This runs the 'Context Optimization' strategy
            # strategy=CONTEXT_OPTIMIZATION ensures chunks are searchable
            proc_result = await self.service.process_large_content(
                content_id=meta.id,
                task_type=ProcessingTaskType.SUMMARIZE, # Using SUMMARIZE to drive the context optimization flow
                strategy=ProcessingStrategy.CONTEXT_OPTIMIZATION,
                query=request.query
            )
            
            # 3. Retrieve & Answer
            # The service now has a unified retrieval method
            answer = await self.service.retrieve_relevant_context(request.query, request.thread_id)
            
            return {
                'success': True,
                'answer': answer,
                'sources': [file_path],
                'metrics': metrics,
                'phase_trace': ['register', 'process', 'retrieve', 'answer'],
                'status': AgentResponseStatus.COMPLETE.value
            }

        except Exception as e:
            logger.error(f"Unified analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'answer': f'Error: {str(e)}',
                'status': AgentResponseStatus.ERROR.value
            }
    def _analyze_multiple_files(self, request: AnalyzeDocumentRequest, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple files concurrently with robust error handling."""
        logger.info(f"Starting batch analysis of {len(file_paths)} files with {request.max_workers} workers")
        
        file_results = []
        all_answers = []
        errors = []
        successful_count = 0
        failed_count = 0

        # Concurrent processing with ThreadPoolExecutor
        max_workers = min(request.max_workers or 4, len(file_paths))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file_safe, fp, request.query): fp
                for fp in file_paths
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=60)  # 60s timeout per file
                    file_results.append(result)
                    
                    if result['success']:
                        successful_count += 1
                        if result.get('answer'):
                            all_answers.append(f"[{Path(file_path).name}]: {result['answer']}")
                    else:
                        failed_count += 1
                        if result.get('error'):
                            errors.append(f"{Path(file_path).name}: {result['error']}")
                
                except Exception as e:
                    logger.error(f"Future failed for {file_path}: {e}")
                    failed_count += 1
                    error_msg = f"Processing timeout or error: {str(e)}"
                    errors.append(f"{Path(file_path).name}: {error_msg}")
                    file_results.append({
                        'file_path': file_path,
                        'success': False,
                        'error': error_msg,
                        'processing_time': None
                    })

        # Aggregate results
        if not all_answers and successful_count == 0:
            combined_answer = f"Failed to analyze all {len(file_paths)} files. Errors: {'; '.join(errors[:3])}"
        else:
            combined_answer = "\n\n".join(all_answers) if all_answers else "Analysis completed with errors."

        response = {
            'success': successful_count > 0,
            'answer': combined_answer,
            'sources': file_paths,
            'total_files': len(file_paths),
            'successful_files': successful_count,
            'failed_files': failed_count,
            'errors': errors if errors else None
        }

        if request.include_per_file_results:
            response['file_results'] = file_results

        return response

    async def _process_single_file_safe(self, file_path: str, query: str) -> Dict[str, Any]:
        """Thread-safe processing of a single file (Async Wrapper)."""
        return await self._analyze_single_file(AnalyzeDocumentRequest(query=query), file_path)
        start_time = time.time()
        result = {
            'file_path': file_path,
            'success': False,
            'answer': None,
            'error': None,
            'processing_time': None
        }

        try:
            # Check cache first
            cache_key = f"{file_path}:{query}"
            cached = self._get_cached_result(cache_key)
            if cached and cached.get('success'):
                result.update({
                    'success': True,
                    'answer': cached.get('answer'),
                    'processing_time': time.time() - start_time
                })
                return result

            # Extract and analyze
            content, _ = extract_document_content(file_path)
            self.metrics["llm_calls"]["analyze"] += 1
            self.metrics["llm_calls"]["total"] += 1
            answer = self.llm_client.analyze_document_with_query(content, query)
            
            result.update({
                'success': True,
                'answer': answer,
                'processing_time': time.time() - start_time
            })

            # Cache successful result
            self._cache_result(cache_key, {'success': True, 'answer': answer})

        except Exception as e:
            logger.error(f"File processing error for {file_path}: {e}")
            result.update({
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            })

        return result

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with hit/miss tracking."""
        with self._cache_lock:
            result = self._analysis_cache.get(cache_key)
            if result:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            return result

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Thread-safe cache storage with size limit."""
        with self._cache_lock:
            if len(self._analysis_cache) >= self._max_cache_size:
                # Simple FIFO eviction
                first_key = next(iter(self._analysis_cache))
                del self._analysis_cache[first_key]
            self._analysis_cache[cache_key] = result

    # RAG retrieval is now integrated into _analyze_single_file using LCEL chain

    # ========== DISPLAY OPERATIONS ==========

    async def display_document(self, file_path: str) -> Dict[str, Any]:
        """Display document with canvas."""
        try:
            self.metrics["api_calls"]["display"] += 1
            if not Path(file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {file_path}'
                }

            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.pdf':
                canvas_display = create_pdf_canvas_display(
                    file_path,
                    Path(file_path).name,
                    'pdf'
                )
            elif file_ext == '.docx':
                # Convert to PDF for display
                pdf_path = convert_docx_to_pdf(file_path)
                canvas_display = create_pdf_canvas_display(
                    pdf_path,
                    Path(file_path).name,
                    'docx'
                )
            else:
                content, _ = extract_document_content(file_path)
                canvas_display = {
                    'canvas_type': 'text',
                    'content': content[:5000],
                    'file_name': Path(file_path).name
                }

            return {
                'success': True,
                'message': 'Document displayed',
                'canvas_display': canvas_display,
                'file_type': file_ext
            }

        except Exception as e:
            logger.error(f"Display failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    # ========== CREATION OPERATIONS ==========

    async def create_document(self, request: CreateDocumentRequest) -> Dict[str, Any]:
        """Create a new document."""
        try:
            self.metrics["api_calls"]["create"] += 1
            # Use absolute path from workspace root
            if Path(request.output_dir).is_absolute():
                output_dir = Path(request.output_dir)
            else:
                output_dir = WORKSPACE_ROOT / request.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / request.file_name

            if request.file_type.value == 'docx':
                create_docx(request.content, str(file_path))
            elif request.file_type.value == 'pdf':
                create_pdf(request.content, str(file_path))
            else:
                with open(file_path, 'w') as f:
                    f.write(request.content)

            logger.info(f"Created document: {file_path}")

            # Create initial version
            self.version_manager.save_version(str(file_path), "Initial creation")

            # Generate Canvas Display
            if request.file_type.value in ['pdf', 'docx']:
                display = CanvasService.build_pdf_view(str(file_path), title=f"Created: {request.file_name}")
            else:
                display = CanvasService.build_document_view(
                    content=request.content[:10000],  # Limit content for init load
                    title=f"Created: {request.file_name}",
                    file_path=str(file_path)
                )

            standard_response = {
                'status': 'success',
                'summary': f"Created document {request.file_name}",
                'canvas_display': display.model_dump()
            }

            return {
                'success': True,
                'message': f'Created {request.file_name}',
                'file_path': str(file_path),
                'standard_response': standard_response
            }

        except Exception as e:
            logger.error(f"Creation failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    # ========== EDITING OPERATIONS ==========

    async def edit_document(self, request: EditDocumentRequest) -> Dict[str, Any]:
        """Edit document using natural language instruction with safety gating."""
        try:
            self.metrics["api_calls"]["edit"] += 1
            phase_trace = ["understand", "plan", "validate", "execute", "verify", "report"]
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}',
                    'status': LocalAgentResponseStatus.ERROR.value,
                    'phase_trace': phase_trace
                }

            risk = self._classify_edit_intent(request.instruction)
            before_hash = self._hash_file_md5(request.file_path)

            # Get or create session
            session = self.session_manager.get_or_create_session(
                request.file_path,
                Path(request.file_path).name,
                request.thread_id
            )

            # Analyze document structure
            structure = analyze_document_structure(request.file_path)
            content, _ = extract_document_content(request.file_path)

            # Plan edits using LLM
            self.metrics["llm_calls"]["edit_planning"] += 1
            self.metrics["llm_calls"]["total"] += 1
            
            # Now awaited
            plan = await self.llm_client.interpret_edit_instruction(
                request.instruction,
                content,
                structure
            )

            if not plan.get('success', False):
                return {
                    'success': False,
                    'message': f'Failed to plan edits: {plan.get("error", "Unknown error")}',
                    'status': LocalAgentResponseStatus.ERROR.value,
                    'risk_assessment': risk,
                    'phase_trace': phase_trace
                }

            validation = self._validate_edit_plan(plan)
            risky_actions = [a for a in validation.get('actions', []) if a.get('type') in self._risky_action_types]
            if risky_actions:
                risk['risk_score'] = max(risk.get('risk_score', 0.0), 0.7)
                risk.setdefault('risk_signals', []).append('risky_action_types')

            if not validation.get('valid', False):
                return {
                    'success': False,
                    'message': 'Edit plan validation failed',
                    'status': LocalAgentResponseStatus.ERROR.value,
                    'risk_assessment': risk,
                    'phase_trace': phase_trace,
                    'errors': validation.get('issues', [])
                }

            # Approval gating: pause if risk is high unless auto_approve is set
            if (risk.get('risk_score', 0) >= 0.6 or risk.get('intent') in {'destructive', 'overwrite'}) and not getattr(request, 'auto_approve', False):
                question = f"Approve edit plan with {len(validation.get('actions', []))} actions (risk_score={risk.get('risk_score')})?"
                return {
                    'success': False,
                    'message': 'Approval required',
                    'status': LocalAgentResponseStatus.NEEDS_INPUT.value,
                    'question': question,
                    'question_type': 'confirmation',
                    'pending_plan': {**plan, 'actions': validation.get('actions', [])},
                    'risk_assessment': risk,
                    'phase_trace': phase_trace
                }

            # Execute edits
            editor = DocumentEditor(request.file_path)
            results = []

            for action in validation.get('actions', []):
                result = self._execute_edit_action(editor, action)
                results.append(result)

            # Save document and create version
            editor.save()
            after_hash = self._hash_file_md5(request.file_path)
            verification = self._verify_edit_result(before_hash, after_hash, results)

            # If nothing changed, avoid creating a noisy new version
            if verification.get('verified') is False and verification.get('reason') == 'no_change_detected':
                return {
                    'success': True,
                    'message': f'No changes applied (no-op) – planned {len(results)} actions',
                    'file_path': request.file_path,
                    'can_undo': len(self.version_manager.get_versions(request.file_path)) > 1,
                    'can_redo': False,
                    'edit_summary': verification,
                    'risk_assessment': risk,
                    'status': LocalAgentResponseStatus.COMPLETE.value,
                    'phase_trace': phase_trace
                }

            self.version_manager.save_version(
                request.file_path,
                f"Edit: {request.instruction[:50]}"
            )

            # Update session with action
            edit_action = StateEditAction(
                timestamp=__import__('datetime').datetime.utcnow().isoformat(),
                action_type='edit_document',
                instruction=request.instruction,
                parameters={'actions': len(results)},
                result=f"Executed {len(results)} actions",
                success=True
            )
            self.session_manager.add_edit_action(session.session_id, edit_action)

            # Generate Canvas Display for Edit Result
            file_ext = Path(request.file_path).suffix.lower()
            if file_ext in ['.pdf', '.docx']:
                 display = CanvasService.build_pdf_view(str(request.file_path), title=f"Edited: {Path(request.file_path).name}")
            else:
                # Re-read content for display
                updated_content, _ = extract_document_content(request.file_path)
                display = CanvasService.build_document_view(
                    content=updated_content[:10000],
                    title=f"Edited: {Path(request.file_path).name}",
                    file_path=str(request.file_path)
                )
            
            standard_response = {
                'status': 'success',
                'summary': f"Applied {len(results)} edits to {Path(request.file_path).name}",
                'canvas_display': display.model_dump()
            }

            return {
                'success': True,
                'message': f"Applied {len(results)} edits",
                'file_path': request.file_path,
                'can_undo': len(self.version_manager.get_versions(request.file_path)) > 1,
                'can_redo': False,
                'edit_summary': verification,
                'risk_assessment': risk,
                'status': LocalAgentResponseStatus.COMPLETE.value,
                'phase_trace': phase_trace,
                'standard_response': standard_response
            }

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'status': LocalAgentResponseStatus.ERROR.value,
                'phase_trace': ['understand', 'plan', 'validate', 'execute', 'verify', 'report']
            }

    def _execute_edit_action(self, editor: DocumentEditor, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single edit action."""
        action_type = action.get('type', '').lower()

        try:
            if action_type == 'add_paragraph':
                result = editor.add_paragraph(
                    action.get('text', ''),
                    action.get('style', 'Normal')
                )
            elif action_type == 'add_heading':
                result = editor.add_heading(
                    action.get('text', ''),
                    action.get('level', 1)
                )
            elif action_type == 'format_text':
                result = editor.format_text(
                    action.get('text', ''),
                    **action.get('options', {})
                )
            elif action_type == 'replace_text':
                result = editor.replace_text(
                    action.get('old_text', ''),
                    action.get('new_text', '')
                )
            elif action_type == 'add_table':
                result = editor.add_table(
                    action.get('rows', 2),
                    action.get('cols', 2)
                )
            else:
                result = f"✗ Unknown action type: {action_type}"

            return {
                'type': action_type,
                'result': result,
                'success': '✓' in result
            }

        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                'type': action_type,
                'result': f"✗ Error: {str(e)}",
                'success': False
            }

    # ========== UNDO/REDO OPERATIONS ==========

    async def undo_redo(self, request: UndoRedoRequest) -> Dict[str, Any]:
        """Undo or redo an edit."""
        try:
            self.metrics["api_calls"]["undo_redo"] += 1
            versions = self.version_manager.get_versions(request.file_path)

            if not versions:
                return {
                    'success': False,
                    'message': 'No versions available'
                }

            current_idx = self.version_manager.index.get(
                self.version_manager._get_document_key(request.file_path), {}
            ).get('current_version', -1)

            if request.action.lower() == 'undo':
                if current_idx > 0:
                    target_version = versions[current_idx - 1]['version_id']
                    success = self.version_manager.restore_version(request.file_path, target_version)
                    if success:
                        return {
                            'success': True,
                            'message': 'Undo successful',
                            'file_path': request.file_path,
                            'can_undo': current_idx > 1,
                            'can_redo': True
                        }
                return {'success': False, 'message': 'Nothing to undo'}

            else:  # redo
                if current_idx < len(versions) - 1:
                    target_version = versions[current_idx + 1]['version_id']
                    success = self.version_manager.restore_version(request.file_path, target_version)
                    if success:
                        return {
                            'success': True,
                            'message': 'Redo successful',
                            'file_path': request.file_path,
                            'can_undo': True,
                            'can_redo': current_idx < len(versions) - 2
                        }
                return {'success': False, 'message': 'Nothing to redo'}

        except Exception as e:
            logger.error(f"Undo/redo failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    # ========== VERSION OPERATIONS ==========

    async def get_version_history(self, file_path: str) -> Dict[str, Any]:
        """Get document version history."""
        try:
            self.metrics["api_calls"]["versions"] += 1
            versions = self.version_manager.get_versions(file_path)
            current_idx = self.version_manager.index.get(
                self.version_manager._get_document_key(file_path), {}
            ).get('current_version', -1)

            return {
                'success': True,
                'message': 'Version history retrieved',
                'versions': versions,
                'current_version': current_idx
            }

        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'versions': [],
                'current_version': -1
            }

    # ========== DATA EXTRACTION ==========

    # ========== DATA EXTRACTION ==========

    async def extract_data(self, request: ExtractDataRequest) -> Dict[str, Any]:
        """Extract structured data from document using ContentManagementService."""
        try:
            self.metrics["api_calls"]["extract"] += 1
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}'
                }

            phase_trace = ["register", "extract", "save_artifact", "report"]
            
            # 1. Register Content
            content, _ = extract_document_content(request.file_path)
            meta = await self.service.register_content(
                content=content,
                name=Path(request.file_path).name,
                source=ContentSource.USER_UPLOAD,
                content_type=ContentType.DOCUMENT,
                thread_id=request.thread_id
            )
            
            # 2. Extract Data (Using Strategy)
            proc_result = await self.service.process_large_content(
                content_id=meta.id,
                task_type=ProcessingTaskType.EXTRACT,
                strategy=ProcessingStrategy.DATA_EXTRACTION
            )
            
            # 3. Prepare Response
            # The service returns the extracted artifact ID in metadata
            extracted_data_id = proc_result.metadata.get("extracted_artifact_id")
            extracted_content = proc_result.final_output
            
            grounding = {
                'source_file': request.file_path,
                'artifact_id': extracted_data_id
            }

            return {
                'success': True,
                'message': 'Data extracted successfully',
                'extracted_data': extracted_content,
                'data_format': request.extraction_type,
                'status': AgentResponseStatus.COMPLETE.value,
                'phase_trace': phase_trace,
                'confidence': 1.0 if extracted_data_id else 0.5,
                'grounding': grounding,
            }

        except Exception as e:
            logger.error(f"Data extraction failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'extracted_data': {},
                'data_format': request.extraction_type,
                'status': AgentResponseStatus.ERROR.value,
                'phase_trace': ["extract", "error"],
                'confidence': 0.0
            }

    # ========== CLEANUP ==========

    def cleanup_old_versions(self, file_path: str, keep_count: int = 10) -> int:
        """Clean up old versions to save storage (cloud optimization)."""
        return self.version_manager.cleanup_old_versions(file_path, keep_count)

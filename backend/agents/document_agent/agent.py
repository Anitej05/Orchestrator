"""
Document Agent - Main Orchestrator

Coordinates all document operations and manages the complete lifecycle.
Designed for cloud deployment with efficient resource management.
"""

import logging
import time
import os
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from schemas import AgentResponseStatus
from .schemas import (
    AnalyzeDocumentRequest, EditDocumentRequest, CreateDocumentRequest,
    UndoRedoRequest, VersionHistoryRequest, ExtractDataRequest, EditAction
)
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
    from utils.metrics_display import display_execution_metrics, display_session_metrics
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

        # Risk controls for enterprise-grade editing
        self._risky_action_types = {"delete_content", "replace_content", "convert_format"}
        self._risk_keywords = ["delete", "remove", "overwrite", "purge", "wipe", "truncate"]
        self._max_safe_actions = 25

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

    # ========== SAFETY & RISK HELPERS ==========
    def _classify_edit_intent(self, instruction: str) -> Dict[str, Any]:
        """Lightweight intent/risk classifier to gate destructive edits."""
        text = instruction.lower()
        risk_hits = [kw for kw in self._risk_keywords if kw in text]
        intent = "edit"
        if any(kw in text for kw in ["delete", "remove", "purge", "wipe"]):
            intent = "destructive"
        elif any(kw in text for kw in ["replace", "rewrite", "overwrite"]):
            intent = "overwrite"
        score = min(1.0, 0.2 + 0.15 * len(risk_hits)) if intent != "edit" else 0.25 + 0.05 * len(risk_hits)
        return {
            "intent": intent,
            "risk_score": round(score, 2),
            "risk_signals": risk_hits,
        }

    def _validate_edit_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM-generated plan against allowed actions and payload size."""
        actions = plan.get("actions") or []
        issues = []
        normalized = []
        if not isinstance(actions, list):
            return {"valid": False, "issues": ["Plan actions must be a list"], "actions": []}
        if len(actions) > self._max_safe_actions:
            issues.append(f"Plan proposes {len(actions)} actions (> {self._max_safe_actions})")
        for idx, action in enumerate(actions):
            a_type = str(action.get("type", "")).lower()
            if not a_type:
                issues.append(f"Action {idx+1} missing type")
                continue
            if a_type not in {
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
            }:
                issues.append(f"Unsupported action type: {a_type}")
            normalized.append({"type": a_type, **{k: v for k, v in action.items() if k != "type"}})
        return {"valid": len(issues) == 0, "issues": issues, "actions": normalized}

    def _build_needs_input_response(self, question: str, plan: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        """Shape a pause/approval response compatible with orchestrator AgentResponse."""
        return {
            "success": False,
            "status": AgentResponseStatus.NEEDS_INPUT.value,
            "question": question,
            "pending_plan": plan,
            "risk_assessment": risk,
        }

    def _hash_file(self, file_path: str) -> Optional[str]:
        """Return md5 hash for a file for change detection."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return hashlib.md5(data).hexdigest()
        except Exception:
            return None

    def _verify_edit_result(self, before_hash: Optional[str], after_hash: Optional[str], actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify that edits produced a change and provide minimal telemetry."""
        if not before_hash or not after_hash:
            return {"verified": False, "reason": "Hash unavailable"}
        if before_hash == after_hash:
            return {"verified": False, "reason": "No change detected", "actions": len(actions)}
        return {"verified": True, "reason": "Document content changed", "actions": len(actions)}

    # ========== ANALYSIS OPERATIONS ==========

    def analyze_document(self, request: AnalyzeDocumentRequest) -> Dict[str, Any]:
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
                result = self._analyze_multiple_files(request, file_paths)
            else:
                # Single file processing (optimized path)
                self._files_processed += 1
                file_path = file_paths[0] if file_paths else None
                logger.info(f"📄 [ANALYZE] Single file mode: processing {file_path}")
                
                # Check if we should use RAG (vector store provided) or Direct Analysis
                if request.vector_store_path or request.vector_store_paths:
                    result = self._analyze_single_file(request, file_path)
                else:
                    logger.info("ℹ️ [ANALYZE] No vector store provided, using direct content analysis")
                    if file_path:
                        result = self._process_single_file_safe(file_path, request.query)
                        # Ensure result format compatibility
                        if 'metrics' not in result:
                            result['metrics'] = {}
                        if 'sources' not in result:
                            result['sources'] = [file_path]
                    else:
                        result = {'success': False, 'answer': 'No file path provided'}
            
            # Track performance metrics
            request_latency = (time.time() - request_start) * 1000
            self.metrics["performance"]["total_latency_ms"] += request_latency
            self.metrics["performance"]["requests_completed"] += 1
            
            logger.info(f"⏱️  [ANALYZE] Request latency: {request_latency:.1f}ms")
            
            # Update RAG metrics if present in result
            if 'metrics' in result:
                req_metrics = result['metrics']
                chunks = req_metrics.get('chunks_retrieved', 0)
                if chunks > 0:
                    self.metrics["rag"]["chunks_retrieved_total"] += chunks
                    logger.info(f"📚 [RAG] Retrieved {chunks} chunks from vector store")
                if req_metrics.get('cache_hit', False):
                    self.metrics["cache"]["hits"] += 1
                    logger.info(f"💾 [CACHE] HIT for query")
                else:
                    self.metrics["cache"]["misses"] += 1
                    logger.info(f"💾 [CACHE] MISS - reprocessing query")
            
            # Track errors
            if not result.get('success', False):
                self.metrics["errors"]["total"] += 1
                error_msg = result.get('answer') or result.get('error') or 'Unknown error'
                if 'RAG' in str(error_msg):
                    self.metrics["errors"]["rag_errors"] += 1
                    logger.error(f"❌ [ANALYZE] RAG error: {error_msg[:200]}")
                elif 'LLM' in str(error_msg):
                    self.metrics["errors"]["llm_errors"] += 1
                    logger.error(f"❌ [ANALYZE] LLM error: {error_msg[:200]}")
                else:
                    logger.error(f"❌ [ANALYZE] Analysis failed: {error_msg[:200]}")
            else:
                logger.info(f"✅ [ANALYZE] Analysis completed successfully")
            
            # Add execution summary to result
            result['execution_metrics'] = {
                'latency_ms': round(request_latency, 2),
                'llm_calls': self.metrics["llm_calls"]["analyze"],
                'cache_hit_rate': round(
                    self.metrics["cache"]["hits"] / (self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]) * 100, 1
                ) if (self.metrics["cache"]["hits"] + self.metrics["cache"]["misses"]) > 0 else 0,
                'chunks_retrieved': result.get('metrics', {}).get('chunks_retrieved', 0),
                'rag_retrieval_ms': round(result.get('metrics', {}).get('rag_retrieval_ms', 0), 2),
                'llm_call_ms': round(result.get('metrics', {}).get('llm_call_ms', 0), 2)
            }
            
            # Log detailed execution metrics
            self._log_execution_metrics(result['execution_metrics'], result.get('success', False))
            
            # Display unified metrics for debugging
            display_execution_metrics(
                metrics=result['execution_metrics'],
                agent_name="Document",
                operation_name="analyze",
                success=result.get('success', False)
            )

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

    def _analyze_single_file(self, request: AnalyzeDocumentRequest, file_path: Optional[str]) -> Dict[str, Any]:
        """Analyze a single document using proven LCEL RAG chain."""
        start_time = time.time()
        metrics = {
            'rag_retrieval_ms': 0,
            'llm_call_ms': 0,
            'cache_hit': False,
            'chunks_retrieved': 0,
            'tokens_used': 0
        }
        
        try:
            # Check cache
            cache_key = None
            if file_path:
                cache_key = f"{file_path}:{request.query}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"✅ Cache hit for {file_path}")
                    metrics['cache_hit'] = True
                    cached_result['metrics'] = metrics
                    return cached_result

            # Collect vector store paths
            paths = request.vector_store_paths or ([request.vector_store_path] if request.vector_store_path else [])
            
            if not paths or not paths[0]:
                return {
                    'success': False,
                    'answer': 'Error: No vector store path provided',
                    'errors': ['Missing vector store path'],
                    'metrics': metrics
                }
            
            # Validate all vector stores exist
            for vsp in paths:
                if not os.path.exists(vsp):
                    logger.error(f"Vector store not found: {vsp}")
                    return {
                        'success': False,
                        'answer': f'Error: Vector store not found at {vsp}',
                        'errors': [f'Vector store missing: {vsp}'],
                        'metrics': metrics
                    }
            
            # RAG retrieval using proven LCEL approach
            rag_start = time.time()
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.runnables import RunnablePassthrough
                from langchain_huggingface import HuggingFaceEmbeddings
                
                # Use same embeddings as orchestrator (all-mpnet-base-v2)
                embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
                
                # Load and merge all vector stores
                logger.info(f"🔍 Loading {len(paths)} vector store(s) for RAG")
                combined_vector_store = None
                
                for idx, vsp in enumerate(paths):
                    logger.info(f"📂 Loading vector store {idx+1}/{len(paths)}: {os.path.basename(vsp)}")
                    vs = FAISS.load_local(
                        vsp,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    if combined_vector_store is None:
                        combined_vector_store = vs
                    else:
                        combined_vector_store.merge_from(vs)
                        logger.info(f"✅ Merged vector store {idx+1}")
                
                # Create retriever
                retriever = combined_vector_store.as_retriever(search_kwargs={"k": 5})
                metrics['rag_retrieval_ms'] = (time.time() - rag_start) * 1000
                
                # Build RAG chain using LCEL
                template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
                
                prompt = ChatPromptTemplate.from_template(template)
                
                def format_docs(docs):
                    metrics['chunks_retrieved'] = len(docs)
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm_client.llm
                    | StrOutputParser()
                )
                
                # Execute RAG chain
                logger.info(f"🤖 Processing query: {request.query[:50]}...")
                llm_start = time.time()
                self.metrics["llm_calls"]["analyze"] += 1
                self.metrics["llm_calls"]["total"] += 1
                
                answer = rag_chain.invoke(request.query)
                metrics['llm_call_ms'] = (time.time() - llm_start) * 1000
                logger.info(f"✅ RAG analysis complete ({metrics['chunks_retrieved']} chunks, {metrics['llm_call_ms']:.0f}ms)")
                
            except Exception as e:
                logger.error(f"RAG retrieval failed: {e}", exc_info=True)
                return {
                    'success': False,
                    'answer': f'RAG error: {str(e)}',
                    'errors': [f'RAG failure: {str(e)}'],
                    'metrics': metrics
                }

            result = {
                'success': True,
                'answer': answer,
                'sources': [file_path] if file_path else paths,
                'metrics': metrics,
            }

            # Cache result
            if cache_key:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Document analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'answer': f'Error: {str(e)}',
                'errors': [str(e)],
                'metrics': metrics
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

    def _process_single_file_safe(self, file_path: str, query: str) -> Dict[str, Any]:
        """Thread-safe processing of a single file with timing and error isolation."""
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

    def display_document(self, file_path: str) -> Dict[str, Any]:
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

    def create_document(self, request: CreateDocumentRequest) -> Dict[str, Any]:
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

            return {
                'success': True,
                'message': f'Created {request.file_name}',
                'file_path': str(file_path)
            }

        except Exception as e:
            logger.error(f"Creation failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    # ========== EDITING OPERATIONS ==========

    def edit_document(self, request: EditDocumentRequest) -> Dict[str, Any]:
        """Edit document using natural language instruction with safety gates."""
        try:
            self.metrics["api_calls"]["edit"] += 1
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}'
                }

            phase_trace = ["understand", "plan", "validate", "execute", "verify", "report"]
            risk = self._classify_edit_intent(request.instruction)
            before_hash = self._hash_file(request.file_path)

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
            plan = self.llm_client.interpret_edit_instruction(
                request.instruction,
                content,
                structure
            )

            if not plan.get('success', False):
                return {
                    'success': False,
                    'message': f'Failed to plan edits: {plan.get("error", "Unknown error")}',
                    'status': AgentResponseStatus.ERROR.value,
                    'risk_assessment': risk,
                    'phase_trace': phase_trace,
                }

            validation = self._validate_edit_plan(plan)
            risky_actions = [a for a in validation.get('actions', []) if a.get('type') in self._risky_action_types]
            if risky_actions:
                risk['risk_score'] = max(risk['risk_score'], 0.7)
                risk.setdefault('risk_signals', []).append('risky_action_types')

            if not validation['valid']:
                return {
                    'success': False,
                    'message': 'Edit plan validation failed',
                    'status': AgentResponseStatus.ERROR.value,
                    'errors': validation['issues'],
                    'risk_assessment': risk,
                    'phase_trace': phase_trace,
                }

            if (risk.get('risk_score', 0) >= 0.6 or risk.get('intent') == 'destructive') and not request.auto_approve:
                question = f"Approve edit plan with {len(validation['actions'])} actions (risk_score={risk['risk_score']})?"
                response = self._build_needs_input_response(question, {**plan, 'actions': validation['actions']}, risk)
                response['phase_trace'] = phase_trace
                return response

            # Execute edits
            editor = DocumentEditor(request.file_path)
            results = []

            for action in validation.get('actions', []):
                result = self._execute_edit_action(editor, action)
                results.append(result)

            # Save document and create version
            editor.save()
            after_hash = self._hash_file(request.file_path)
            verification = self._verify_edit_result(before_hash, after_hash, results)
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

            return {
                'success': True,
                'message': f"Applied {len(results)} edits",
                'file_path': request.file_path,
                'can_undo': len(self.version_manager.get_versions(request.file_path)) > 1,
                'can_redo': False,
                'edit_summary': verification,
                'risk_assessment': risk,
                'status': AgentResponseStatus.COMPLETE.value,
                'phase_trace': phase_trace,
            }

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'status': AgentResponseStatus.ERROR.value,
                'phase_trace': ["understand", "plan", "validate", "execute", "verify", "report"],
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

    def undo_redo(self, request: UndoRedoRequest) -> Dict[str, Any]:
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

    def get_version_history(self, file_path: str) -> Dict[str, Any]:
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

    def extract_data(self, request: ExtractDataRequest) -> Dict[str, Any]:
        """Extract structured data from document."""
        try:
            self.metrics["api_calls"]["extract"] += 1
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}'
                }

            phase_trace = ["understand", "extract", "validate", "report"]
            content, _ = extract_document_content(request.file_path)

            self.metrics["llm_calls"]["extract"] += 1
            self.metrics["llm_calls"]["total"] += 1
            result = self.llm_client.extract_structured_data(
                content,
                request.extraction_type
            )

            confidence = 0.8 if result.get('success') else 0.0
            if not result.get('data') and not result.get('content'):
                confidence = 0.2
            grounding = {
                'source_file': request.file_path,
                'notes': 'LLM-only extraction; validate downstream',
            }

            return {
                'success': result.get('success', False),
                'message': 'Data extracted' if result.get('success') else 'Extraction failed',
                'extracted_data': result.get('data', result.get('content', '')),
                'data_format': request.extraction_type,
                'status': AgentResponseStatus.COMPLETE.value if result.get('success') else AgentResponseStatus.ERROR.value,
                'phase_trace': phase_trace,
                'confidence': confidence,
                'grounding': grounding,
            }

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'extracted_data': {},
                'data_format': request.extraction_type,
                'status': AgentResponseStatus.ERROR.value,
                'phase_trace': ["understand", "extract", "validate", "report"],
                'confidence': 0.0,
                'grounding': {'source_file': request.file_path, 'notes': 'exception raised'},
            }

    # ========== CLEANUP ==========

    def cleanup_old_versions(self, file_path: str, keep_count: int = 10) -> int:
        """Clean up old versions to save storage (cloud optimization)."""
        return self.version_manager.cleanup_old_versions(file_path, keep_count)

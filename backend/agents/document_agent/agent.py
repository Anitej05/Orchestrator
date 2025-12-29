"""
Document Agent - Main Orchestrator

Coordinates all document operations and manages the complete lifecycle.
Designed for cloud deployment with efficient resource management.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .schemas import (
    AnalyzeDocumentRequest, EditDocumentRequest, CreateDocumentRequest,
    UndoRedoRequest, VersionHistoryRequest, ExtractDataRequest, EditAction
)
from .editors import DocumentEditor
from .state import DocumentSessionManager, DocumentVersionManager, EditAction as StateEditAction
from .llm import DocumentLLMClient
from .utils import (
    extract_document_content, create_docx, create_pdf, analyze_document_structure,
    convert_docx_to_pdf, create_pdf_canvas_display, ensure_directory, extract_document_content
)

logger = logging.getLogger(__name__)

# Get workspace root (3 levels up from this file: agent.py -> document_agent -> agents -> backend -> root)
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DEFAULT_STORAGE_DIR = WORKSPACE_ROOT / "storage" / "documents"


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

    # ========== ANALYSIS OPERATIONS ==========

    def analyze_document(self, request: AnalyzeDocumentRequest) -> Dict[str, Any]:
        """Analyze document(s) with RAG and answer queries. Supports multi-file batch processing."""
        try:
            # Collect all file paths to process
            file_paths = []
            if request.file_paths:
                file_paths.extend(request.file_paths)
            elif request.file_path:
                file_paths.append(request.file_path)

            # Multi-file batch processing
            if len(file_paths) > 1:
                return self._analyze_multiple_files(request, file_paths)

            # Single file processing (optimized path)
            return self._analyze_single_file(request, file_paths[0] if file_paths else None)

        except Exception as e:
            logger.error(f"Analysis orchestration failed: {e}", exc_info=True)
            return {
                'success': False,
                'answer': f'Critical error: {str(e)}',
                'errors': [str(e)]
            }

    def _analyze_single_file(self, request: AnalyzeDocumentRequest, file_path: Optional[str]) -> Dict[str, Any]:
        """Analyze a single document with caching and error handling."""
        try:
            # Check cache
            cache_key = None
            if file_path:
                cache_key = f"{file_path}:{request.query}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for {file_path}")
                    return cached_result

            # Extract content
            content = None
            if file_path:
                try:
                    content, _ = extract_document_content(file_path)
                    logger.info(f"Extracted {len(content)} chars from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    return {
                        'success': False,
                        'answer': f'Error reading file: {str(e)}',
                        'errors': [f"File read error: {str(e)}"]
                    }

            # Fallback to vector store if no content
            paths = request.vector_store_paths or ([request.vector_store_path] if request.vector_store_path else [])
            if not content:
                if not paths or not paths[0]:
                    return {
                        'success': False,
                        'answer': 'Error: No content or vector store path provided',
                        'errors': ['Missing both file content and vector store path']
                    }
                content = ""  # Empty content for vector-only queries

            # Analyze with LLM
            try:
                answer = self.llm_client.analyze_document_with_query(content, request.query)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                return {
                    'success': False,
                    'answer': f'LLM error: {str(e)}',
                    'errors': [f"LLM failure: {str(e)}"]
                }

            result = {
                'success': True,
                'answer': answer,
                'sources': [file_path] if file_path else (paths if paths else [])
            }

            # Cache result
            if cache_key:
                self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Single file analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'answer': f'Error: {str(e)}',
                'errors': [str(e)]
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
        """Thread-safe cache retrieval."""
        with self._cache_lock:
            return self._analysis_cache.get(cache_key)

    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Thread-safe cache storage with size limit."""
        with self._cache_lock:
            if len(self._analysis_cache) >= self._max_cache_size:
                # Simple FIFO eviction
                first_key = next(iter(self._analysis_cache))
                del self._analysis_cache[first_key]
            self._analysis_cache[cache_key] = result

    # ========== DISPLAY OPERATIONS ==========

    def display_document(self, file_path: str) -> Dict[str, Any]:
        """Display document with canvas."""
        try:
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
        """Edit document using natural language instruction."""
        try:
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}'
                }

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
            plan = self.llm_client.interpret_edit_instruction(
                request.instruction,
                content,
                structure
            )

            if not plan.get('success', False):
                return {
                    'success': False,
                    'message': f'Failed to plan edits: {plan.get("error", "Unknown error")}'
                }

            # Execute edits
            editor = DocumentEditor(request.file_path)
            results = []

            for action in plan.get('actions', []):
                result = self._execute_edit_action(editor, action)
                results.append(result)

            # Save document and create version
            editor.save()
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
                'message': f'Applied {len(results)} edits',
                'file_path': request.file_path,
                'can_undo': len(self.version_manager.get_versions(request.file_path)) > 1,
                'can_redo': False
            }

        except Exception as e:
            logger.error(f"Edit failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
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
            if not Path(request.file_path).exists():
                return {
                    'success': False,
                    'message': f'File not found: {request.file_path}'
                }

            content, _ = extract_document_content(request.file_path)

            result = self.llm_client.extract_structured_data(
                content,
                request.extraction_type
            )

            return {
                'success': result.get('success', False),
                'message': 'Data extracted' if result.get('success') else 'Extraction failed',
                'extracted_data': result.get('data', result.get('content', '')),
                'data_format': request.extraction_type
            }

        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'extracted_data': {},
                'data_format': request.extraction_type
            }

    # ========== CLEANUP ==========

    def cleanup_old_versions(self, file_path: str, keep_count: int = 10) -> int:
        """Clean up old versions to save storage (cloud optimization)."""
        return self.version_manager.cleanup_old_versions(file_path, keep_count)

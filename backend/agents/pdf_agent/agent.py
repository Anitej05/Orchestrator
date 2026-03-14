"""
PDF Agent - Main Agent Orchestrator

Central orchestrator for PDF operations.
Unified /execute endpoint with LLM-powered analysis and Q&A.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from .config import logger
from .agent_schemas import ExecuteResponse, TaskStatus
from .state import session_state, Session
from .llm_helpers import PDFLLMHelpers
from . import utils

from backend.services.content_management_service import (
    ContentManagementService,
    ContentSource,
    ContentType,
    ContentPriority
)
from backend.services.canvas_service import CanvasService
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus

logger = logging.getLogger("pdf_agent.agent")


class PDFAgent(PDFLLMHelpers):
    """
    Central orchestrator for PDF operations.
    
    Features:
    - Unified /execute endpoint
    - LLM-powered PDF analysis and Q&A
    - Session management
    - CMS integration
    
    Inherits from PDFLLMHelpers for LLM methods:
    - analyze_pdf_structure()
    - summarize_pdf()
    - extract_key_information()
    - answer_questions_about_pdf()
    - suggest_pdf_improvements()
    - generate_pdf_metadata()
    """
    
    def __init__(self):
        self.state = session_state
        self.cms = ContentManagementService()
        logger.info("PDFAgent initialized (using PDFLLMHelpers)")
    
    def _extract_prompt(self, params: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from parameters."""
        if not params:
            return None
        
        fields = ['prompt', 'query', 'instruction', 'question', 'content', 'message']
        for field in fields:
            if params.get(field):
                return str(params[field])
        
        return None
    
    async def execute(
        self,
        prompt: str = None,
        action: str = None,
        params: Dict[str, Any] = None,
        thread_id: str = "default",
        task_id: str = None,
        file_content: bytes = None,
        filename: str = None
    ) -> AgentResponse:
        """
        Unified execution endpoint.
        
        Supports:
        1. Complex prompt mode: LLM analyzes/summarizes PDF
        2. Direct action mode: Execute specific PDF operation
        3. File upload mode: Load PDF file
        4. Q&A mode: Answer questions about PDF
        """
        params = params or {}
        
        try:
            # Get or create session
            session = self.state.get_or_create(thread_id)
            
            # Handle file upload
            if file_content and filename:
                upload_result = await self._handle_file_upload(
                    file_content, filename, thread_id, session
                )
                if not prompt:
                    return upload_result
            
            # Execute based on prompt type
            if self._is_qa_prompt(prompt):
                exec_res = await self._execute_qa(
                    prompt, thread_id, session, params
                )
            elif self._is_analysis_prompt(prompt):
                exec_res = await self._execute_analysis(
                    prompt, thread_id, session, params
                )
            elif action:
                exec_res = await self._execute_action(action, params, session)
            else:
                exec_res = await self._execute_simple(prompt, params, session)
            
            # Convert to standard response
            return self._convert_to_standard_response(exec_res, session)
            
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                success=False,
                error_message=str(e)
            )
    
    def _is_qa_prompt(self, prompt: str) -> bool:
        """Check if prompt is a question about PDF content."""
        if not prompt:
            return False
        
        qa_keywords = [
            'what', 'how', 'when', 'where', 'why', 'who',
            'explain', 'describe', 'tell me',
            'question', 'answer', 'about the pdf',
            'in the document', 'according to'
        ]
        
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in qa_keywords)
    
    def _is_analysis_prompt(self, prompt: str) -> bool:
        """Check if prompt requires PDF analysis."""
        if not prompt:
            return False
        
        analysis_keywords = [
            'summarize', 'summary',
            'analyze', 'analysis',
            'extract', 'extraction',
            'key points', 'main points',
            'improve', 'suggest', 'review'
        ]
        
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in analysis_keywords)
    
    async def _handle_file_upload(
        self,
        file_content: bytes,
        filename: str,
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Handle PDF file upload."""
        try:
            # Save file
            file_path = session.save_file(file_content, filename)
            
            # Register with CMS
            content = await self.cms.register_content(
                source=ContentSource(
                    type=ContentType.FILE,
                    data=file_content,
                    filename=filename,
                ),
                processing_strategy="auto_detect",
                thread_id=thread_id,
            )
            
            # Extract text
            text_content = utils.extract_text_from_pdf(str(file_path))
            
            # Extract metadata
            metadata = utils.extract_pdf_metadata(str(file_path))
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=True,
                message=f"Loaded PDF: {filename} ({len(text_content)} characters, {metadata.get('pages', 0)} pages)",
                data={
                    "file_path": str(file_path),
                    "filename": filename,
                    "content_id": content.id if content else None,
                    "text_content": text_content,
                    "metadata": metadata,
                }
            )
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    async def _execute_qa(
        self,
        prompt: str,
        thread_id: str,
        session: Session,
        params: Dict[str, Any]
    ) -> ExecuteResponse:
        """Execute Q&A about PDF content."""
        try:
            # Get PDF text from session or params
            text_content = params.get('text_content') or session.get_text_content()
            
            if not text_content:
                return ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error="No PDF content available. Please upload a PDF first."
                )
            
            # Use LLM to answer question
            logger.info(f"Answering question about PDF...")
            answer = await self.answer_questions_about_pdf(
                text_content=text_content,
                question=prompt
            )
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=answer.get('in_document', False),
                message=answer.get('answer', 'Answer unavailable'),
                data={
                    "question": prompt,
                    "answer": answer.get('answer'),
                    "confidence": answer.get('confidence', 0),
                    "source_quotes": answer.get('source_quotes', []),
                }
            )
            
        except Exception as e:
            logger.error(f"Q&A failed: {e}", exc_info=True)
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    async def _execute_analysis(
        self,
        prompt: str,
        thread_id: str,
        session: Session,
        params: Dict[str, Any]
    ) -> ExecuteResponse:
        """Execute PDF analysis."""
        try:
            # Get PDF text
            text_content = params.get('text_content') or session.get_text_content()
            metadata = params.get('metadata', {})
            
            if not text_content:
                return ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error="No PDF content available."
                )
            
            # Determine analysis type
            prompt_lower = prompt.lower()
            
            if 'summarize' in prompt_lower or 'summary' in prompt_lower:
                # Extract style parameter
                style = params.get('style', 'executive')
                max_length = params.get('max_length', 500)
                
                logger.info(f"Summarizing PDF ({style} style)...")
                summary = await self.summarize_pdf(
                    text_content=text_content,
                    max_length=max_length,
                    style=style
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Summary ({style} style)",
                    data={
                        "summary": summary,
                        "style": style,
                    }
                )
            
            elif 'extract' in prompt_lower:
                # Extract key information
                extraction_type = params.get('extraction_type', 'general')
                
                logger.info(f"Extracting {extraction_type} information...")
                extracted = await self.extract_key_information(
                    text_content=text_content,
                    extraction_type=extraction_type
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Extracted {extraction_type} information",
                    data={
                        "extracted_data": extracted,
                        "extraction_type": extraction_type,
                    }
                )
            
            elif 'improve' in prompt_lower or 'suggest' in prompt_lower:
                # Suggest improvements
                document_type = params.get('document_type', 'general')
                
                logger.info(f"Suggesting improvements...")
                suggestions = await self.suggest_pdf_improvements(
                    text_content=text_content,
                    document_type=document_type
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Improvement suggestions (score: {suggestions.get('overall_score', 0)}/10)",
                    data={
                        "suggestions": suggestions,
                    }
                )
            
            else:
                # General analysis
                logger.info(f"Analyzing PDF structure...")
                analysis = await self.analyze_pdf_structure(
                    text_content=text_content,
                    metadata=metadata
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message="PDF analysis complete",
                    data={
                        "analysis": analysis,
                    }
                )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    async def _execute_action(
        self,
        action: str,
        params: Dict[str, Any],
        session: Session
    ) -> ExecuteResponse:
        """Execute specific PDF operation."""
        try:
            if action == 'merge_pdfs':
                file_paths = params.get('file_paths', [])
                output_path = utils.merge_pdfs(file_paths)
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Merged {len(file_paths)} PDFs",
                    data={"file_path": str(output_path)}
                )
            
            elif action == 'split_pdf':
                file_path = params.get('file_path')
                output_paths = utils.split_pdf(file_path)
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Split into {len(output_paths)} PDFs",
                    data={"file_paths": [str(p) for p in output_paths]}
                )
            
            elif action == 'extract_text':
                file_path = params.get('file_path')
                text = utils.extract_text_from_pdf(file_path)
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Extracted {len(text)} characters",
                    data={"text_content": text}
                )
            
            elif action == 'extract_tables':
                file_path = params.get('file_path')
                tables = utils.extract_tables_from_pdf(file_path)
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Extracted {len(tables)} tables",
                    data={"tables": tables}
                )
            
            else:
                return ExecuteResponse(
                    status=TaskStatus.ERROR,
                    success=False,
                    error=f"Unknown action: {action}"
                )
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    async def _execute_simple(
        self,
        prompt: str,
        params: Dict[str, Any],
        session: Session
    ) -> ExecuteResponse:
        """Execute simple prompt."""
        try:
            response = await self.llm_generate(
                prompt=prompt,
                system_prompt="You are a PDF expert. Help users with PDF-related tasks."
            )
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=True,
                message=response,
                data={"response": response}
            )
            
        except Exception as e:
            logger.error(f"Simple execution failed: {e}")
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    def _convert_to_standard_response(
        self,
        exec_res: ExecuteResponse,
        session: Session
    ) -> AgentResponse:
        """Convert ExecuteResponse to StandardAgentResponse."""
        try:
            # Build canvas display
            canvas_display = None
            data = exec_res.data or {}
            
            if data.get('file_path') and str(data['file_path']).lower().endswith('.pdf'):
                # Use PDF viewer
                canvas_obj = CanvasService.build_pdf_view(
                    file_path=data['file_path'],
                    title=Path(data['file_path']).name,
                )
                if canvas_obj:
                    canvas_display = canvas_obj.model_dump()
            
            return AgentResponse(
                status=AgentResponseStatus.COMPLETE if exec_res.success else AgentResponseStatus.ERROR,
                success=exec_res.success,
                summary=exec_res.message or ("Success" if exec_res.success else exec_res.error),
                standard_response=StandardAgentResponse(
                    status=AgentResponseStatus.COMPLETE if exec_res.success else AgentResponseStatus.ERROR,
                    success=exec_res.success,
                    summary=exec_res.message or ("Success" if exec_res.success else exec_res.error),
                    data=data,
                    canvas_display=canvas_display,
                    files_generated=[{"path": data.get('file_path')}] if data.get('file_path') else [],
                )
            )
            
        except Exception as e:
            logger.error(f"Response conversion failed: {e}")
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                success=False,
                error_message=str(e)
            )


# Singleton instance
pdf_agent = PDFAgent()

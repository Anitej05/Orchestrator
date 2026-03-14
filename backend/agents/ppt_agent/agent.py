"""
PPT Agent - Main Agent Orchestrator

Central orchestrator for PowerPoint operations.
Unified /execute endpoint with LLM-powered task decomposition.
"""

import logging
import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from backend.base_agent import BaseAgent, AgentRequest, capability, ExecutionContext, CapabilityResult
from backend.base_agent.types import AgentResponse as BaseAgentResponse
from .llm_helpers import PPTLLMHelpers
from . import utils

from backend.services.content_management_service import (
    ContentManagementService,
    ContentSource,
    ContentType,
    ContentPriority
)
from backend.services.canvas_service import CanvasService
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus

logger = logging.getLogger("ppt_agent.agent")


class PPTAgent(BaseAgent, PPTLLMHelpers):
    """
    Central orchestrator for PowerPoint operations.
    
    Features:
    - Unified /execute endpoint
    - LLM-powered presentation planning
    - Design-aware layout suggestions
    - Session management
    - CMS integration
    
    Inherits from PPTLLMHelpers for LLM methods:
    - plan_presentation_structure()
    - suggest_slide_layout()
    - generate_slide_content()
    - enhance_text_for_presentation()
    - suggest_color_palette()
    - suggest_visual_elements()
    - check_presentation_consistency()
    - generate_speaker_notes()
    """
    
    def __init__(self):
        super().__init__(
            agent_id="ppt_agent",
            agent_name="PPT Agent",
        )
        self.cms = ContentManagementService()
        self.storage_dir = Path("storage/ppt_agent")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PPTAgent initialized (using PPTLLMHelpers)")
    
    def _extract_prompt(self, params: Dict[str, Any]) -> Optional[str]:
        """Extract prompt from parameters."""
        if not params:
            return None
        
        fields = ['prompt', 'query', 'instruction', 'topic', 'title', 'content', 'message']
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
        1. Complex prompt mode: LLM plans presentation structure
        2. Direct action mode: Execute specific action
        3. File upload mode: Load PPTX file
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
            if self._is_presentation_planning_prompt(prompt):
                exec_res = await self._execute_presentation_creation(
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
    
    def _is_presentation_planning_prompt(self, prompt: str) -> bool:
        """Check if prompt requires LLM-powered presentation planning."""
        if not prompt:
            return False
        
        planning_keywords = [
            'create presentation', 'create ppt', 'create powerpoint',
            'make presentation', 'make ppt', 'make slides',
            'design presentation', 'design slides',
            'presentation about', 'slides about',
            'pitch deck', 'slide deck'
        ]
        
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in planning_keywords)
    
    async def _handle_file_upload(
        self,
        file_content: bytes,
        filename: str,
        thread_id: str,
        session: Session
    ) -> ExecuteResponse:
        """Handle PPTX file upload."""
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
            
            # Extract text for session
            text_content = utils.extract_text_from_pptx(str(file_path))
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=True,
                message=f"Loaded presentation: {filename} ({len(text_content)} characters)",
                data={
                    "file_path": str(file_path),
                    "filename": filename,
                    "content_id": content.id if content else None,
                    "text_length": len(text_content),
                }
            )
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return ExecuteResponse(
                status=TaskStatus.ERROR,
                success=False,
                error=str(e)
            )
    
    async def _execute_presentation_creation(
        self,
        prompt: str,
        thread_id: str,
        session: Session,
        params: Dict[str, Any]
    ) -> ExecuteResponse:
        """Execute presentation creation with LLM planning."""
        try:
            # Extract parameters
            topic = params.get('topic') or self._extract_topic(prompt)
            audience = params.get('audience', 'general')
            slide_count = params.get('slide_count', 5)
            style = params.get('style', 'professional')
            
            # Use LLM to plan presentation structure
            logger.info(f"Planning presentation: {topic[:50]}...")
            plan = await self.plan_presentation_structure(
                topic=topic,
                audience=audience,
                slide_count=slide_count,
                style=style
            )
            
            # Get color palette suggestion
            palette = await self.suggest_color_palette(
                topic=topic,
                mood=style
            )
            
            # Create presentation using plan
            logger.info(f"Creating presentation with {len(plan.get('slides', []))} slides...")
            file_path = await utils.create_presentation_from_plan(
                plan=plan,
                palette=palette,
                output_dir=str(session.storage_dir)
            )
            
            # Read back for verification
            text_content = utils.extract_text_from_pptx(str(file_path))
            
            return ExecuteResponse(
                status=TaskStatus.COMPLETED,
                success=True,
                message=f"Created presentation: {plan.get('presentation_title', 'Untitled')} ({len(plan.get('slides', []))} slides)",
                data={
                    "file_path": str(file_path),
                    "filename": Path(file_path).name,
                    "slide_count": len(plan.get('slides', [])),
                    "text_content": text_content,
                    "plan": plan,
                }
            )
            
        except Exception as e:
            logger.error(f"Presentation creation failed: {e}", exc_info=True)
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
        """Execute specific action."""
        try:
            if action == 'read_presentation':
                file_path = params.get('file_path')
                if not file_path:
                    return ExecuteResponse(
                        status=TaskStatus.ERROR,
                        success=False,
                        error="file_path required"
                    )
                
                text_content = utils.extract_text_from_pptx(file_path)
                structure = utils.analyze_pptx_structure(file_path)
                
                # Use LLM to analyze structure
                analysis = await self.analyze_pdf_structure(
                    text_content=text_content,
                    metadata={'pages': len(structure.get('slides', []))}
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message=f"Read presentation: {Path(file_path).name}",
                    data={
                        "text_content": text_content,
                        "structure": structure,
                        "analysis": analysis,
                    }
                )
            
            elif action == 'edit_presentation':
                # Use LLM to plan edits
                instruction = params.get('instruction', '')
                file_path = params.get('file_path')
                
                plan = await self.generate_slide_content(
                    slide_title=instruction[:100],
                    key_message=instruction
                )
                
                # Apply edits
                edited_path = await utils.edit_presentation(
                    file_path=file_path,
                    edits=plan
                )
                
                return ExecuteResponse(
                    status=TaskStatus.COMPLETED,
                    success=True,
                    message="Edited presentation",
                    data={"file_path": str(edited_path)}
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
            # Use LLM to understand and respond
            response = await self.llm_generate(
                prompt=prompt,
                system_prompt="You are a PowerPoint expert. Help users with presentation-related tasks."
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
    
    def _extract_topic(self, prompt: str) -> str:
        """Extract presentation topic from prompt."""
        # Simple extraction - can be enhanced with LLM
        keywords = ['about', 'on', 'regarding', 'for']
        for kw in keywords:
            if kw in prompt.lower():
                parts = prompt.lower().split(kw, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return prompt[:100]
    
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
            
            if data.get('file_path') and str(data['file_path']).lower().endswith('.pptx'):
                # Use PPTX viewer
                canvas_obj = CanvasService.build_pptx_view(
                    file_path=data['file_path'],
                    title=Path(data['file_path']).name,
                    slide_count=len(data.get('plan', {}).get('slides', []))
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
ppt_agent = PPTAgent()

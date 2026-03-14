# agents/coding_agent/llm_helpers.py
"""
Coding Agent - LLM Helper Functions

Domain-specific LLM methods for coding operations.
All methods use inference_service directly.

Preserves ALL original functionality from CodingAgentLLM.
"""
import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from backend.services.inference_service import inference_service, InferencePriority

logger = logging.getLogger("coding_agent.llm")


# ============================================================================
# SCHEMAS - PRESERVED FROM ORIGINAL coding_agent/llm.py
# ============================================================================

class CanvasDecision(BaseModel):
    """LLM's decision on how to display output in the canvas."""
    canvas_type: str = Field(
        description="One of: code, markdown, html, json, chart, image, spreadsheet"
    )
    canvas_title: str = Field(
        description="Short, descriptive title for the canvas panel"
    )
    canvas_content: Optional[str] = Field(
        None,
        description="For html/markdown: the rendered content string. "
        "For html: must be complete self-contained HTML. "
        "For markdown: formatted markdown text."
    )
    canvas_data: Optional[Dict[str, Any]] = Field(
        None,
        description="For structured types: code (diffs, code blocks), "
        "json (tree data), chart (labels/datasets), spreadsheet (headers/rows)"
    )
    template_id: Optional[str] = Field(
        None,
        description="If using a registered template (e.g. 'code_diff_viewer', 'json_tree', 'chart_bar')"
    )
    requires_confirmation: bool = Field(
        False,
        description="True if file modifications need user approval"
    )
    confirmation_message: Optional[str] = Field(
        None,
        description="Message shown in the approval dialog"
    )
    reasoning: str = Field(
        description="Brief explanation of why this canvas type was chosen"
    )


# ============================================================================
# CANVAS DECISION PROMPT - PRESERVED FROM ORIGINAL coding_agent/llm.py
# ============================================================================

CANVAS_DECISION_SYSTEM = """You are a Canvas Display Selector for the Orbimesh AI orchestrator.

Your job: Given an AI coding agent's output, decide the BEST way to display it visually in the Canvas panel.

## Available Canvas Types

| canvas_type  | Use When                                                    | Data Format                     |
|-------------|-------------------------------------------------------------|---------------------------------|
| code        | File diffs, code blocks, terminal output                    | canvas_data with diffs/code     |
| markdown    | Analysis, explanations, reviews, documentation              | canvas_content (markdown string)|
| html        | Live HTML previews, styled docs, dashboards, interactive UI | canvas_content (full HTML page) |
| json        | Structured JSON data, API responses, config files           | canvas_data with data object    |
| chart       | Numerical data, statistics, metrics                         | canvas_data with labels/datasets|
| spreadsheet | Tabular data, CSV-like output                               | canvas_data with headers/rows   |
| image       | Generated images                                            | canvas_data with src            |

## Registered Templates
{templates}

## Rules

1. If file changes exist → canvas_type="code", template_id="code_diff_viewer", requires_confirmation=True
2. If the output IS a complete HTML page → canvas_type="html", put the HTML in canvas_content (strip any ```html wrapping)
3. If the output is primarily analysis/text → canvas_type="markdown", put formatted text in canvas_content
4. If the output is JSON data → canvas_type="json", template_id="json_tree", put parsed JSON in canvas_data.data
5. If the output has tabular structure → canvas_type="spreadsheet", extract headers and rows into canvas_data
6. For test results → canvas_type="code", include pass/fail in title
7. You CAN transform the output — e.g. reformat markdown, extract HTML from code blocks, restructure JSON
8. canvas_content and canvas_data are mutually exclusive — use ONE based on canvas_type
9. Keep canvas_title short (under 60 chars)

## Output Format

Return ONLY a JSON object matching this schema:
{{
  "canvas_type": "...",
  "canvas_title": "...",
  "canvas_content": "...",  // OR
  "canvas_data": {{...}},
  "template_id": "...",
  "requires_confirmation": true/false,
  "confirmation_message": "...",
  "reasoning": "..."
}}"""


class CodingLLMHelpers:
    """
    Coding-specific LLM helpers.
    
    Mix this into CodingAgent to get coding-specific LLM methods.
    All methods use inference_service directly (preserving original functionality).
    """
    
    # ========================================================================
    # PUBLIC API - PRESERVED FROM ORIGINAL CodingAgentLLM
    # ========================================================================
    
    async def decide_canvas(
        self,
        opencode_output: str,
        task_description: str,
        available_templates: List[Dict[str, Any]] = None
    ) -> CanvasDecision:
        """
        Analyze OpenCode output and decide on optimal canvas display.
        
        PRESERVED FROM ORIGINAL CodingAgentLLM - ALL FUNCTIONALITY INTACT
        
        Args:
            opencode_output: Raw output from OpenCode server
            task_description: Original user task
            available_templates: List of available canvas templates
            
        Returns:
            CanvasDecision with display instructions
        """
        # Build templates section
        if available_templates:
            templates_str = "\n".join([
                f"- {t.get('id', 'unknown')}: {t.get('name', 'Unnamed')}"
                for t in available_templates
            ])
        else:
            templates_str = "No custom templates available."
        
        # Build the full prompt
        prompt = CANVAS_DECISION_SYSTEM.format(templates=templates_str)
        
        # Add user context
        user_message = f"""TASK: {task_description}

OPENCODE OUTPUT:
{opencode_output[:10000]}  # Limit to avoid token overflow

Decide the best canvas display for this output."""
        
        try:
            # Use generate_structured for Pydantic output
            decision = await inference_service.generate_structured(
                messages=[
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_message)
                ],
                schema=CanvasDecision,
                priority=InferencePriority.QUALITY,
                temperature=0.1,
            )
            
            logger.info(f"Canvas decision: {decision.canvas_type} - {decision.canvas_title}")
            return decision
            
        except Exception as e:
            logger.error(f"Canvas decision failed: {e}")
            # Fallback to code display
            return CanvasDecision(
                canvas_type="code",
                canvas_title="Coding Output",
                canvas_content=opencode_output[:5000],
                template_id=None,
                requires_confirmation=False,
                reasoning=f"Canvas decision failed: {str(e)}. Using fallback code display."
            )

"""
Coding Agent LLM Client

Uses the centralized InferenceService for all LLM-powered decisions.
Primary function: decide_canvas() — analyzes OpenCode output and picks
the optimal canvas type, content, and template.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("coding_agent.llm")


# ============================================================================
# SCHEMAS
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
# CANVAS DECISION PROMPT
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
{
  "canvas_type": "...",
  "canvas_title": "...",
  "canvas_content": "..." or null,
  "canvas_data": {...} or null,
  "template_id": "..." or null,
  "requires_confirmation": true/false,
  "confirmation_message": "..." or null,
  "reasoning": "..."
}"""


# ============================================================================
# LLM CLIENT
# ============================================================================


class CodingAgentLLM:
    """LLM client for coding agent decisions."""

    def __init__(self):
        logger.info("CodingAgentLLM initialized (using InferenceService)")

    async def decide_canvas(
        self,
        output: str,
        file_changes: List[Dict[str, Any]],
        files_modified: List[str],
        capability_name: str,
        available_templates: List[str],
    ) -> CanvasDecision:
        """
        Ask the LLM to decide the best canvas display for this output.

        Args:
            output: Raw text output from OpenCode
            file_changes: List of {file, diff, language, status} dicts
            files_modified: List of modified file paths
            capability_name: Which capability produced this (code_task, review_code, etc.)
            available_templates: List of registered template IDs

        Returns:
            CanvasDecision with canvas type, content/data, and metadata
        """
        # Build template context
        templates_str = ", ".join(available_templates) if available_templates else "none"

        system_prompt = CANVAS_DECISION_SYSTEM.replace("{templates}", templates_str)

        # Build user message
        user_parts = [f"Capability: {capability_name}"]

        if files_modified:
            user_parts.append(f"\nFiles modified ({len(files_modified)}):")
            for f in files_modified[:10]:
                user_parts.append(f"  - {f}")
            user_parts.append(f"\nFile diffs ({len(file_changes)}):")
            for change in file_changes[:5]:
                diff_preview = change.get("diff", "")[:500]
                user_parts.append(f"--- {change.get('file', '?')} ({change.get('status', 'modified')}) ---")
                user_parts.append(diff_preview)
                if len(change.get("diff", "")) > 500:
                    user_parts.append("... (truncated)")

        # Truncate output to avoid token limits
        output_preview = output[:3000] if output else "(empty)"
        if len(output) > 3000:
            output_preview += f"\n... ({len(output) - 3000} more chars)"

        user_parts.append(f"\nAgent Output:\n{output_preview}")

        user_message = "\n".join(user_parts)

        try:
            response = await inference_service.generate(
                messages=[
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ],
                priority=InferencePriority.SPEED,
                temperature=0.1,
                json_mode=True,
                strip_think_tags=True,
                strip_markdown=True,
            )

            decision = self._parse_decision(response, output, file_changes, files_modified)
            logger.info(
                f"🎨 LLM canvas decision: {decision.canvas_type} "
                f"(template={decision.template_id}) — {decision.reasoning}"
            )
            return decision

        except Exception as e:
            logger.warning(f"LLM canvas decision failed: {e}. Using fallback.")
            return self._fallback_decision(output, file_changes, files_modified, capability_name)

    def _parse_decision(
        self,
        response: str,
        output: str,
        file_changes: List[Dict],
        files_modified: List[str],
    ) -> CanvasDecision:
        """Parse LLM JSON response into CanvasDecision."""
        try:
            data = self._extract_json(response)
            if not data:
                raise ValueError("No JSON found in response")

            # Validate canvas_type
            valid_types = {"code", "markdown", "html", "json", "chart", "spreadsheet", "image"}
            if data.get("canvas_type") not in valid_types:
                data["canvas_type"] = "markdown"

            # Ensure file diffs are included when files were modified
            if files_modified and data.get("canvas_type") == "code":
                if not data.get("canvas_data") or "diffs" not in data.get("canvas_data", {}):
                    data["canvas_data"] = {
                        "diffs": file_changes[:20],
                        "files_modified": files_modified,
                        "file_count": len(files_modified),
                    }

            return CanvasDecision(**data)

        except Exception as e:
            logger.warning(f"Failed to parse canvas decision: {e}")
            return self._fallback_decision(output, file_changes, files_modified, "unknown")

    def _fallback_decision(
        self,
        output: str,
        file_changes: List[Dict],
        files_modified: List[str],
        capability_name: str,
    ) -> CanvasDecision:
        """Simple fallback when LLM fails — uses file changes as the primary signal."""
        if file_changes:
            return CanvasDecision(
                canvas_type="code",
                canvas_title=f"Code Changes ({len(files_modified)} files)",
                canvas_data={
                    "diffs": file_changes[:20],
                    "files_modified": files_modified,
                    "file_count": len(files_modified),
                },
                template_id="code_diff_viewer",
                requires_confirmation=True,
                confirmation_message=f"Apply changes to {len(files_modified)} file(s)?",
                reasoning="Fallback: file changes detected",
            )

        return CanvasDecision(
            canvas_type="markdown",
            canvas_title=capability_name.replace("_", " ").title(),
            canvas_content=output or "No output",
            requires_confirmation=False,
            reasoning="Fallback: no file changes, using markdown",
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON object from LLM response text."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        # Try extracting from code block
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass
        # Try finding first { ... }
        brace_start = text.find("{")
        if brace_start >= 0:
            depth = 0
            for i, ch in enumerate(text[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start : i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break
        return None


# Singleton
llm_client = CodingAgentLLM()

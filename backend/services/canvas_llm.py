"""
Canvas LLM Service — Shared Dynamic Canvas Decision

Uses InferenceService to analyze agent output and decide the optimal
canvas type, template, and content. Any agent can call decide_canvas().
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("CanvasLLM")


# ============================================================================
# SCHEMA
# ============================================================================

class CanvasDecision(BaseModel):
    """LLM's decision on how to display output in the canvas."""
    canvas_type: str = Field(
        description="One of: code, markdown, html, json, chart, image, spreadsheet, email_preview, pdf, pptx"
    )
    canvas_title: str = Field(description="Short title for the canvas panel (under 60 chars)")
    canvas_content: Optional[str] = Field(
        None, description="For html/markdown: rendered content string"
    )
    canvas_data: Optional[Dict[str, Any]] = Field(
        None, description="For structured types: code, json, chart, spreadsheet, email_preview"
    )
    template_id: Optional[str] = Field(
        None, description="Registered template ID if applicable"
    )
    requires_confirmation: bool = Field(
        False, description="True if action needs user approval"
    )
    confirmation_message: Optional[str] = Field(
        None, description="Approval dialog message"
    )
    reasoning: str = Field(description="Brief explanation of why this canvas type was chosen")


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

CANVAS_DECISION_SYSTEM = """You are the Canvas Display Selector for the Orbimesh AI orchestrator.

Given an agent's output, decide the BEST visual format for the Canvas panel.

## Canvas Types

| canvas_type     | Use When                                                | Data Format                     |
|----------------|----------------------------------------------------------|---------------------------------|
| code           | File diffs, code blocks, terminal output                 | canvas_data with diffs/code     |
| markdown       | Analysis, explanations, reviews, text responses          | canvas_content (markdown string)|
| html           | Live HTML previews, styled docs, dashboards, demos       | canvas_content (full HTML page) |
| json           | Structured JSON data, API responses, config              | canvas_data with data object    |
| chart          | Numerical data, statistics, metrics, aggregations        | canvas_data with labels/datasets|
| spreadsheet    | Tabular data with headers and rows                       | canvas_data with headers/rows   |
| email_preview  | Email drafts with to/subject/body                        | canvas_data with to/subject/body|
| pdf            | PDF documents                                            | canvas_data with file_path      |
| pptx           | PowerPoint presentations (created, edited, previewed)    | canvas_data with file_path      |
| image          | Generated images                                         | canvas_data with src            |

## Registered Templates
{templates}

## Rules

1. File diffs present → canvas_type="code", template_id="code_diff_viewer", requires_confirmation=True
2. Complete HTML page → canvas_type="html", put HTML in canvas_content (strip ```html wrapping)
3. Email with to/subject/body → canvas_type="email_preview"
4. Tabular data with columns/rows → canvas_type="spreadsheet", template_id="spreadsheet_viewer"
5. JSON data → canvas_type="json", template_id="json_tree"
6. Charts/metrics → canvas_type="chart", pick chart_bar/chart_line/chart_pie
7. Text analysis/review → canvas_type="markdown"
8. Agent context matters: use the agent_name and capability to inform your choice
9. canvas_content and canvas_data are mutually exclusive per entry
10. You CAN transform the output (reformat markdown, extract HTML, restructure JSON)
11. For spreadsheet canvas_data: use {"headers": [...], "rows": [[...], ...]} format
12. For chart canvas_data: use {"labels": [...], "datasets": [{"label": "...", "data": [...]}]} format

Return ONLY valid JSON matching:
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
# MAIN API
# ============================================================================

async def decide_canvas(
    output: str,
    agent_name: str,
    capability_name: str = "",
    file_changes: Optional[List[Dict[str, Any]]] = None,
    files_modified: Optional[List[str]] = None,
    available_templates: Optional[List[str]] = None,
    primary_canvas_type: Optional[str] = None,
) -> CanvasDecision:
    """
    Ask the LLM to decide the best canvas display for this output.

    Args:
        output: Raw text output from the agent
        agent_name: Which agent produced this (e.g. "spreadsheet_agent")
        capability_name: What action was performed
        file_changes: List of {file, diff, language, status} dicts (for code agents)
        files_modified: List of modified file paths
        available_templates: Registered canvas template IDs
        primary_canvas_type: Agent's default canvas type (e.g. "spreadsheet")
            The LLM can override this if output is better suited elsewhere

    Returns:
        CanvasDecision with canvas type, content/data, and metadata
    """
    file_changes = file_changes or []
    files_modified = files_modified or []
    templates_str = ", ".join(available_templates) if available_templates else "all registered templates"

    system_prompt = CANVAS_DECISION_SYSTEM.replace("{templates}", templates_str)

    # Build user message
    parts = [
        f"Agent: {agent_name}",
        f"Capability: {capability_name}",
    ]
    if primary_canvas_type:
        parts.append(f"Primary canvas type: {primary_canvas_type} (override only if output clearly needs a different type)")

    if files_modified:
        parts.append(f"\nFiles modified ({len(files_modified)}):")
        for f in files_modified[:10]:
            parts.append(f"  - {f}")

    if file_changes:
        parts.append(f"\nFile diffs ({len(file_changes)}):")
        for change in file_changes[:5]:
            diff_preview = str(change.get("diff", ""))[:400]
            parts.append(f"--- {change.get('file', '?')} ({change.get('status', 'modified')}) ---")
            parts.append(diff_preview)

    # Truncate output
    output_text = output[:3000] if output else "(empty)"
    if len(output or "") > 3000:
        output_text += f"\n... ({len(output) - 3000} more chars)"
    parts.append(f"\nAgent Output:\n{output_text}")

    user_message = "\n".join(parts)

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

        decision = _parse_decision(response, output, file_changes, files_modified, agent_name, capability_name)
        logger.info(
            f"🎨 [{agent_name}] Canvas decision: {decision.canvas_type} "
            f"(template={decision.template_id}) — {decision.reasoning}"
        )
        return decision

    except Exception as e:
        logger.warning(f"LLM canvas decision failed for {agent_name}: {e}")
        return _fallback_decision(output, file_changes, files_modified, agent_name, capability_name, primary_canvas_type)


# ============================================================================
# PARSING
# ============================================================================

def _parse_decision(
    response: str,
    output: str,
    file_changes: List[Dict],
    files_modified: List[str],
    agent_name: str,
    capability_name: str,
) -> CanvasDecision:
    """Parse LLM JSON response into CanvasDecision."""
    try:
        data = _extract_json(response)
        if not data:
            raise ValueError("No JSON found in LLM response")

        valid_types = {"code", "markdown", "html", "json", "chart", "spreadsheet", "email_preview", "pdf", "pptx", "image"}
        if data.get("canvas_type") not in valid_types:
            data["canvas_type"] = "markdown"

        # Ensure file diffs are attached when files were modified
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
        return _fallback_decision(output, file_changes, files_modified, agent_name, capability_name)


def _fallback_decision(
    output: str,
    file_changes: Optional[List[Dict]] = None,
    files_modified: Optional[List[str]] = None,
    agent_name: str = "",
    capability_name: str = "",
    primary_canvas_type: Optional[str] = None,
) -> CanvasDecision:
    """Simple fallback when LLM fails."""
    file_changes = file_changes or []
    files_modified = files_modified or []

    if file_changes:
        return CanvasDecision(
            canvas_type="code",
            canvas_title=f"Code Changes ({len(files_modified)} files)",
            canvas_data={"diffs": file_changes[:20], "files_modified": files_modified, "file_count": len(files_modified)},
            template_id="code_diff_viewer",
            requires_confirmation=True,
            confirmation_message=f"Apply changes to {len(files_modified)} file(s)?",
            reasoning="Fallback: file changes detected",
        )

    # Use primary canvas type if specified
    if primary_canvas_type == "spreadsheet":
        return CanvasDecision(
            canvas_type="spreadsheet",
            canvas_title=capability_name.replace("_", " ").title() or "Data",
            canvas_content=None,
            canvas_data=None,
            requires_confirmation=False,
            reasoning="Fallback: spreadsheet agent default",
        )
    if primary_canvas_type == "email_preview":
        return CanvasDecision(
            canvas_type="markdown",
            canvas_title="Email Agent Response",
            canvas_content=output or "No output",
            requires_confirmation=False,
            reasoning="Fallback: mail agent non-email output",
        )
    if primary_canvas_type == "pdf":
        return CanvasDecision(
            canvas_type="markdown",
            canvas_title=capability_name.replace("_", " ").title() or "PDF Result",
            canvas_content=output or "No output",
            requires_confirmation=False,
            reasoning="Fallback: pdf agent text output",
        )
    if primary_canvas_type == "pptx":
        return CanvasDecision(
            canvas_type="markdown",
            canvas_title=capability_name.replace("_", " ").title() or "Presentation Result",
            canvas_content=output or "No output",
            requires_confirmation=False,
            reasoning="Fallback: ppt agent text output",
        )

    return CanvasDecision(
        canvas_type="markdown",
        canvas_title=capability_name.replace("_", " ").title() or agent_name.replace("_", " ").title(),
        canvas_content=output or "No output",
        requires_confirmation=False,
        reasoning="Fallback: default markdown",
    )


def _extract_json(text: str) -> Optional[Dict]:
    """Extract JSON object from LLM response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
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

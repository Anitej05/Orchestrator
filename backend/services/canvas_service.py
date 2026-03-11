from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import ValidationError
from backend.schemas import CanvasDisplay
from services.canvas_templates import (
    get_template, list_templates, get_template_ids,
    validate_template_data
)
import logging

logger = logging.getLogger("CanvasService")

class CanvasService:
    """
    Centralized service for managing Canvas displays.
    Handles validation, construction, and extraction of visual content
    for the frontend canvas (spreadsheets, documents, email previews, etc).
    """

    # --- FACTORY METHODS (Create standard views) ---

    @staticmethod
    def build_spreadsheet_view(
        filename: str,
        dataframe: Any = None, # pd.DataFrame (lazy import to avoid heavy deps if unused)
        headers: Optional[List[str]] = None,
        rows: Optional[List[List[Any]]] = None,
        title: str = "Spreadsheet View",
        requires_confirmation: bool = False
    ) -> CanvasDisplay:
        """
        Build a standardized spreadsheet canvas.
        Accepts either a pandas DataFrame OR direct headers/rows.
        """
        canvas_data = {
            "filename": filename
        }

        # Handle DataFrame input if pandas is available
        if dataframe is not None:
            try:
                import pandas as pd
                if isinstance(dataframe, pd.DataFrame):
                    # Replace NaN with None (null in JSON)
                    df_clean = dataframe.where(pd.notnull(dataframe), None)
                    canvas_data["headers"] = list(df_clean.columns)
                    canvas_data["rows"] = df_clean.values.tolist()
            except ImportError:
                logger.warning("Pandas not installed, skipping DataFrame conversion")

        # Explicit headers/rows override
        if headers:
            canvas_data["headers"] = headers
        if rows:
            canvas_data["rows"] = rows

        return CanvasDisplay(
            canvas_type="spreadsheet",
            canvas_data=canvas_data,
            canvas_title=title,
            requires_confirmation=requires_confirmation
        )

    @staticmethod
    def build_email_preview(
        to: Union[str, List[str]],
        subject: str,
        body: str,
        cc: Optional[Union[str, List[str]]] = None,
        is_html: bool = False,
        requires_confirmation: bool = True,
        confirmation_message: str = "Send this email?"
    ) -> CanvasDisplay:
        """Build an email preview canvas."""
        
        # Normalize recipients to lists
        def ensure_list(val):
            if not val: return []
            return [val] if isinstance(val, str) else val

        return CanvasDisplay(
            canvas_type="email_preview",
            canvas_data={
                "to": ensure_list(to),
                "cc": ensure_list(cc),
                "subject": subject,
                "body": body,
                "is_html": is_html
            },
            canvas_title="Email Preview",
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message
        )

    @staticmethod
    def build_document_view(
        content: str,
        format: Literal["markdown", "html", "text"] = "markdown",
        title: str = "Document Viewer",
        file_path: Optional[str] = None
    ) -> CanvasDisplay:
        """
        Build a document view.
        Uses `canvas_content` for raw text/html/md, as currently preferred by frontend for docs.
        """
        return CanvasDisplay(
            canvas_type=format if format in ["html", "markdown"] else "markdown",
            canvas_content=content,
            canvas_title=title,
            canvas_data={"file_path": file_path} if file_path else None
        )

    @staticmethod
    def build_pdf_view(
        file_path: str,
        title: str = "PDF Viewer"
    ) -> CanvasDisplay:
        """Build a PDF/DOCX viewer canvas."""
        return CanvasDisplay(
            canvas_type="pdf",
            canvas_title=title,
            canvas_data={"file_path": file_path}
        )

    @staticmethod
    def build_pptx_view(
        file_path: str,
        title: str = "Presentation Viewer",
        slide_count: Optional[int] = None,
    ) -> CanvasDisplay:
        """Build a PPTX presentation viewer canvas."""
        data = {"file_path": file_path}
        if slide_count is not None:
            data["slide_count"] = slide_count
        return CanvasDisplay(
            canvas_type="pptx",
            canvas_title=title,
            canvas_data=data,
        )

    # --- TEMPLATE-BASED BUILDERS ---

    @staticmethod
    def build_from_template(
        template_id: str,
        data: Dict[str, Any],
        title: Optional[str] = None,
        requires_confirmation: Optional[bool] = None,
        confirmation_message: Optional[str] = None,
    ) -> Optional[CanvasDisplay]:
        """
        Build a canvas from a predefined template.

        1. Looks up the template in the registry
        2. Validates data against the template's data_schema
        3. Returns a CanvasDisplay with template_id attached

        Args:
            template_id: Template to use (e.g., 'chart_bar', 'spreadsheet_viewer')
            data: Structured data matching the template's data_schema
            title: Override the default title
            requires_confirmation: Override the template's confirmation setting
            confirmation_message: Override the template's confirmation message
        """
        template = get_template(template_id)
        if not template:
            logger.error(f"❌ Unknown canvas template: {template_id}")
            return None

        # Validate data
        is_valid, error = validate_template_data(template_id, data)
        if not is_valid:
            logger.warning(f"⚠️ Template validation: {error}")
            # Proceed anyway — soft validation

        # Determine settings
        config = template.get("default_config", {})
        canvas_type = template["canvas_type"]

        confirm = requires_confirmation
        if confirm is None:
            confirm = config.get("requires_confirmation", False)

        confirm_msg = confirmation_message
        if confirm_msg is None:
            confirm_msg = config.get("confirmation_message")

        # Attach template_id into canvas_data so frontend knows which template rendered it
        enriched_data = {**data, "template_id": template_id}

        return CanvasDisplay(
            canvas_type=canvas_type,
            canvas_data=enriched_data,
            canvas_title=title or template.get("display_name"),
            requires_confirmation=confirm,
            confirmation_message=confirm_msg,
        )

    @staticmethod
    def build_chart(
        chart_type: Literal["bar", "line", "pie"],
        labels: List[str],
        datasets: Optional[List[Dict[str, Any]]] = None,
        values: Optional[List[float]] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> Optional[CanvasDisplay]:
        """
        Build a chart canvas using predefined chart templates.

        For bar/line charts: provide labels + datasets
        For pie charts: provide labels + values
        """
        template_id = f"chart_{chart_type}"

        if chart_type == "pie":
            if not values:
                logger.error("Pie chart requires 'values'")
                return None
            data = {"labels": labels, "values": values, "title": title or "Pie Chart"}
        else:
            if not datasets:
                logger.error(f"{chart_type} chart requires 'datasets'")
                return None
            data = {
                "labels": labels,
                "datasets": datasets,
                "title": title or f"{chart_type.title()} Chart",
                "chart_subtype": chart_type,
            }
            if x_label:
                data["x_label"] = x_label
            if y_label:
                data["y_label"] = y_label

        return CanvasService.build_from_template(template_id, data, title=title)

    @staticmethod
    def build_code_view(
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
        title: Optional[str] = None,
    ) -> CanvasDisplay:
        """Build a syntax-highlighted code viewer canvas."""
        return CanvasService.build_from_template(
            "code_viewer",
            {"code": code, "language": language, "filename": filename},
            title=title or f"Code: {filename or language}",
        )

    @staticmethod
    def build_code_diff_view(
        diffs: List[Dict[str, str]],
        summary: str = "",
        files_modified: Optional[List[str]] = None,
        terminal_log: Optional[str] = None,
        tests_passed: Optional[bool] = None,
        requires_confirmation: bool = True,
        confirmation_message: str = "Apply these code changes?",
        title: Optional[str] = None,
    ) -> Optional[CanvasDisplay]:
        """
        Build a code diff viewer canvas for the coding agent.

        Args:
            diffs: List of {"file": path, "diff": unified_diff, "language": lang, "status": "modified"|"created"|"deleted"}
            summary: Natural language summary of changes
            files_modified: List of file paths that were modified
            terminal_log: Terminal output from the coding task
            tests_passed: Whether tests passed (None if not run)
            requires_confirmation: Whether user must approve before applying
            confirmation_message: Approval button text
            title: Optional canvas title override
        """
        file_count = len(diffs)
        auto_title = title or f"Code Changes ({file_count} file{'s' if file_count != 1 else ''})"

        data = {
            "diffs": diffs,
            "summary": summary,
            "files_modified": files_modified or [d.get("file", "") for d in diffs],
            "file_count": file_count,
        }
        if terminal_log:
            data["terminal_log"] = terminal_log
        if tests_passed is not None:
            data["tests_passed"] = tests_passed

        return CanvasService.build_from_template(
            "code_diff_viewer",
            data,
            title=auto_title,
            requires_confirmation=requires_confirmation,
            confirmation_message=confirmation_message,
        )

    @staticmethod
    def get_available_templates(
        category: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List available canvas templates, optionally filtered."""
        return list_templates(category=category, agent=agent)

    # --- EXTRACTION METHODS (Parse agent results) ---

    @staticmethod
    def extract_canvas_from_result(
        task_name: str,
        result: Any,
        agent_name: str = "Unknown"
    ) -> Optional[Dict[str, Any]]:
        """
        Extract valid canvas data from an agent's result.
        Handles:
        1. StandardAgentResponse V2 (preferred)
        2. Legacy dictionary formats (nested canvas_display, etc.)
        3. Raw AgentResponse objects
        
        Returns a dict ready for the orchestrator's state (with metadata), or None.
        """
        canvas_display = None

        # 1. Check for StandardResponse V2
        if isinstance(result, dict):
            std_resp = result.get('standard_response')
            if isinstance(std_resp, dict) and std_resp.get('canvas_display'):
                # Direct CanvasDisplay object in V2 response
                canvas_display = std_resp['canvas_display']
                logger.info(f"🎨 Found StandardResponse V2 canvas for '{task_name}'")
            elif isinstance(std_resp, dict) and std_resp.get('canvas_data'):
                # Implicit V2 canvas from canvas_data (backward compat within V2)
                logger.info(f"🎨 Found StandardResponse V2 canvas_data for '{task_name}'")
                canvas_display = {
                    'canvas_data': std_resp.get('canvas_data'),
                    'canvas_type': std_resp.get('canvas_type', 'spreadsheet'),
                    'canvas_title': std_resp.get('canvas_title')
                }

        # 2. Check for Legacy/Direct Dict
        if not canvas_display and isinstance(result, dict):
            # Check top-level
            if 'canvas_display' in result:
                canvas_display = result['canvas_display']
            # Check nested in 'result' key
            elif isinstance(result.get('result'), dict) and 'canvas_display' in result['result']:
                canvas_display = result['result']['canvas_display']
            
            # Special Case: Spreadsheet Agent V1 plan_id propagation
            nested_res = result.get('result', {}) if isinstance(result.get('result'), dict) else {}
            if canvas_display and nested_res.get('plan_id'):
                if isinstance(canvas_display, dict):
                    canvas_display['plan_id'] = nested_res['plan_id']

        # 3. Validate and Enrich
        if canvas_display:
            try:
                # Convert to Pydantic first to validate (strips extra junk)
                # If it's already a dict, validation happens here
                # Convert to Pydantic first to validate (strips extra junk)
                # If it's already a dict, validation happens here
                if isinstance(canvas_display, dict):
                    # STRICT VALIDATION: Check required keys for specific types
                    c_type = canvas_display.get('canvas_type')
                    c_data = canvas_display.get('canvas_data', {})
                    
                    if c_type == 'spreadsheet':
                        if not c_data.get('headers') or not isinstance(c_data['headers'], list):
                             logger.warning(f"⚠️ Canvas '{task_name}' (spreadsheet) missing valid 'headers'")
                        if 'rows' not in c_data or not isinstance(c_data['rows'], list):
                             logger.warning(f"⚠️ Canvas '{task_name}' (spreadsheet) missing valid 'rows'")
                             
                    elif c_type == 'email_preview':
                        required = ['to', 'subject', 'body']
                        missing = [k for k in required if k not in c_data]
                        if missing:
                            logger.error(f"❌ Canvas '{task_name}' (email_preview) missing keys: {missing}")
                            # Could ideally raise ValidationError here to enforce strictness

                    validated = CanvasDisplay(**canvas_display)
                    final_display = validated.model_dump()
                else:
                    # Already an object?
                    final_display = canvas_display.model_dump()
                
                # Add Orchestrator Metadata
                final_display['task_name'] = task_name
                final_display['agent_name'] = agent_name
                
                logger.info(f"✅ Validated canvas '{final_display.get('canvas_title')}' (type={final_display.get('canvas_type')})")
                return final_display

            except ValidationError as e:
                logger.error(f"❌ Invalid canvas data from '{task_name}': {e}")
                return None
            except Exception as e:
                logger.error(f"❌ Error processing canvas from '{task_name}': {e}")
                return None

        return None

    # --- LLM-POWERED CANVAS DECISION ---

    @staticmethod
    async def decide_canvas_llm(
        output: str,
        agent_name: str,
        capability_name: str = "",
        file_changes: Optional[List[Dict[str, Any]]] = None,
        files_modified: Optional[List[str]] = None,
        primary_canvas_type: Optional[str] = None,
    ) -> Optional["CanvasDisplay"]:
        """
        LLM-powered canvas decision. Analyzes agent output and picks the
        best canvas type, template, and content dynamically.

        All agents should call this method for dynamic canvas support.

        Args:
            output: Raw agent output text
            agent_name: e.g. "spreadsheet_agent", "coding_agent"
            capability_name: e.g. "code_task", "load_file"
            file_changes: For coding agents — list of {file, diff, status} dicts
            files_modified: List of modified file paths
            primary_canvas_type: Agent's default type (LLM can override if better)

        Returns:
            Validated CanvasDisplay or None on failure
        """
        try:
            from services.canvas_llm import decide_canvas as _llm_decide
        except ImportError:
            try:
                from backend.services.canvas_llm import decide_canvas as _llm_decide
            except ImportError:
                logger.warning("canvas_llm module not available — skipping LLM canvas decision")
                return None

        try:
            templates = get_template_ids()
        except Exception:
            templates = []

        decision = await _llm_decide(
            output=output,
            agent_name=agent_name,
            capability_name=capability_name,
            file_changes=file_changes,
            files_modified=files_modified,
            available_templates=templates,
            primary_canvas_type=primary_canvas_type,
        )

        # Convert to CanvasDisplay
        try:
            # Try template first
            if decision.template_id:
                template_data = dict(decision.canvas_data or {})
                # For markdown_viewer template: LLM may put content in canvas_content
                # rather than canvas_data["content"]. Bridge the gap here.
                if decision.template_id == "markdown_viewer" and not template_data.get("content") and decision.canvas_content:
                    template_data["content"] = decision.canvas_content
                display = CanvasService.build_from_template(
                    template_id=decision.template_id,
                    data=template_data,
                    title=decision.canvas_title,
                    requires_confirmation=decision.requires_confirmation,
                    confirmation_message=decision.confirmation_message,
                )
                if display:
                    return display

            # Direct CanvasDisplay
            kwargs = {
                "canvas_type": decision.canvas_type,
                "canvas_title": decision.canvas_title,
                "requires_confirmation": decision.requires_confirmation,
                "confirmation_message": decision.confirmation_message,
            }
            if decision.canvas_content:
                kwargs["canvas_content"] = decision.canvas_content
            if decision.canvas_data:
                kwargs["canvas_data"] = decision.canvas_data

            return CanvasDisplay(**kwargs)

        except Exception as e:
            logger.warning(f"Failed to build CanvasDisplay from LLM decision: {e}")
            return None

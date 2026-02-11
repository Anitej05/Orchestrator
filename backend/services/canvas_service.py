from typing import Dict, Any, Optional, List, Union, Literal
from pydantic import ValidationError
from backend.schemas import CanvasDisplay, StandardAgentResponse
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

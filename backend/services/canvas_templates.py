"""
Canvas Template Registry

Predefined, structured templates that agents reference by template_id.
Each template defines the expected data schema and default configuration.
Agents generate structured data → templates tell the frontend how to render it.

Usage:
    from services.canvas_templates import get_template, list_templates, CANVAS_TEMPLATES

    template = get_template("chart_bar")
    # template["data_schema"]["required"] → ["labels", "datasets"]
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("CanvasTemplates")

# ============================================================================
# Template Definitions
# ============================================================================

CANVAS_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # SPREADSHEET TEMPLATES
    # ------------------------------------------------------------------
    "spreadsheet_viewer": {
        "template_id": "spreadsheet_viewer",
        "canvas_type": "spreadsheet",
        "display_name": "Spreadsheet Viewer",
        "description": "Excel-like interactive grid with headers, rows, download buttons, and metadata",
        "category": "data",
        "data_schema": {
            "required": ["headers", "rows"],
            "properties": {
                "headers": {"type": "array", "items": {"type": "string"}, "description": "Column header names"},
                "rows": {"type": "array", "items": {"type": "array"}, "description": "2D array of cell values"},
                "filename": {"type": "string", "description": "Source file name"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "rows_total": {"type": "integer"},
                        "rows_shown": {"type": "integer"},
                        "columns": {"type": "integer"},
                        "truncated": {"type": "boolean"},
                    }
                },
                "file_id": {"type": "string", "description": "File ID for download links"}
            }
        },
        "default_config": {
            "max_rows_display": 100,
            "enable_download": True,
            "show_row_numbers": True,
            "show_column_letters": True,
        },
        "agent_hints": ["spreadsheet_agent", "universal_agent"],
    },

    "spreadsheet_plan": {
        "template_id": "spreadsheet_plan",
        "canvas_type": "spreadsheet_plan",
        "display_name": "Spreadsheet Execution Plan",
        "description": "Pre-execution plan table showing steps the spreadsheet agent will take",
        "category": "data",
        "data_schema": {
            "required": ["headers", "rows"],
            "properties": {
                "headers": {"type": "array", "items": {"type": "string"}, "default": ["Step", "Action", "Description"]},
                "rows": {"type": "array", "items": {"type": "array"}},
                "plan_summary": {"type": "string"},
                "estimated_steps": {"type": "integer"},
            }
        },
        "default_config": {"requires_confirmation": True},
        "agent_hints": ["spreadsheet_agent"],
    },

    # ------------------------------------------------------------------
    # EMAIL TEMPLATES
    # ------------------------------------------------------------------
    "email_preview": {
        "template_id": "email_preview",
        "canvas_type": "email_preview",
        "display_name": "Email Preview",
        "description": "Rich email card with To/CC/Subject/Body, confirmation buttons for send",
        "category": "communication",
        "data_schema": {
            "required": ["to", "subject", "body"],
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient emails"},
                "cc": {"type": "array", "items": {"type": "string"}},
                "bcc": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "is_html": {"type": "boolean", "default": False},
                "attachments": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "files": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        },
        "default_config": {
            "requires_confirmation": True,
            "confirmation_message": "Review and confirm to send this email",
        },
        "agent_hints": ["mail_agent"],
    },

    # ------------------------------------------------------------------
    # DOCUMENT TEMPLATES
    # ------------------------------------------------------------------
    "document_viewer": {
        "template_id": "document_viewer",
        "canvas_type": "document",
        "display_name": "Document Viewer",
        "description": "Rich document viewer with title, status badge, markdown content, and file metadata",
        "category": "document",
        "data_schema": {
            "required": ["content"],
            "properties": {
                "content": {"type": "string", "description": "Markdown or plain text content"},
                "title": {"type": "string"},
                "status": {"type": "string", "enum": ["preview", "created", "edited"]},
                "file_path": {"type": "string"},
                "file_type": {"type": "string", "enum": ["md", "txt", "docx", "html"]},
                "metadata": {"type": "object"},
            }
        },
        "default_config": {},
        "agent_hints": ["document_agent", "universal_agent"],
    },

    "pdf_viewer": {
        "template_id": "pdf_viewer",
        "canvas_type": "pdf",
        "display_name": "PDF Viewer",
        "description": "Embedded PDF viewer with zoom, undo/redo for editable docs",
        "category": "document",
        "data_schema": {
            "required": ["file_path"],
            "properties": {
                "file_path": {"type": "string", "description": "Path to PDF file or API URL"},
                "pdf_data": {"type": "string", "description": "Base64-encoded PDF (alternative to file_path)"},
                "title": {"type": "string"},
                "status": {"type": "string", "enum": ["preview", "created", "edited"]},
                "original_type": {"type": "string", "description": "Original format if converted (e.g., 'docx')"},
            }
        },
        "default_config": {"zoom": 125},
        "agent_hints": ["document_agent"],
    },

    "markdown_viewer": {
        "template_id": "markdown_viewer",
        "canvas_type": "markdown",
        "display_name": "Markdown Viewer",
        "description": "Rendered markdown content with prose styling",
        "category": "document",
        "data_schema": {
            "required": ["content"],
            "properties": {
                "content": {"type": "string", "description": "Markdown text to render"},
                "title": {"type": "string"},
            }
        },
        "default_config": {},
        "agent_hints": ["document_agent", "universal_agent"],
    },

    # ------------------------------------------------------------------
    # CHART / PLOT TEMPLATES
    # ------------------------------------------------------------------
    "chart_bar": {
        "template_id": "chart_bar",
        "canvas_type": "chart",
        "display_name": "Bar Chart",
        "description": "Vertical or horizontal bar chart for comparing categories",
        "category": "visualization",
        "data_schema": {
            "required": ["labels", "datasets"],
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Category labels (X-axis)"},
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "data"],
                        "properties": {
                            "label": {"type": "string", "description": "Series name"},
                            "data": {"type": "array", "items": {"type": "number"}},
                            "color": {"type": "string", "description": "Hex color (optional)"},
                        }
                    }
                },
                "title": {"type": "string"},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
                "orientation": {"type": "string", "enum": ["vertical", "horizontal"], "default": "vertical"},
            }
        },
        "default_config": {
            "colors": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"],
            "show_legend": True,
            "show_grid": True,
        },
        "agent_hints": ["spreadsheet_agent", "universal_agent"],
    },

    "chart_line": {
        "template_id": "chart_line",
        "canvas_type": "chart",
        "display_name": "Line Chart",
        "description": "Line chart for trends over time or sequences",
        "category": "visualization",
        "data_schema": {
            "required": ["labels", "datasets"],
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis labels"},
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["label", "data"],
                        "properties": {
                            "label": {"type": "string"},
                            "data": {"type": "array", "items": {"type": "number"}},
                            "color": {"type": "string"},
                        }
                    }
                },
                "title": {"type": "string"},
                "x_label": {"type": "string"},
                "y_label": {"type": "string"},
            }
        },
        "default_config": {
            "colors": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"],
            "show_legend": True,
            "show_grid": True,
            "curve_type": "monotone",
        },
        "agent_hints": ["spreadsheet_agent", "universal_agent"],
    },

    "chart_pie": {
        "template_id": "chart_pie",
        "canvas_type": "chart",
        "display_name": "Pie Chart",
        "description": "Pie/donut chart for proportional data",
        "category": "visualization",
        "data_schema": {
            "required": ["labels", "values"],
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Slice labels"},
                "values": {"type": "array", "items": {"type": "number"}, "description": "Slice values"},
                "title": {"type": "string"},
                "donut": {"type": "boolean", "default": False, "description": "Render as donut instead of pie"},
            }
        },
        "default_config": {
            "colors": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"],
            "show_legend": True,
            "show_percentages": True,
        },
        "agent_hints": ["spreadsheet_agent", "universal_agent"],
    },

    # ------------------------------------------------------------------
    # CODE / JSON TEMPLATES
    # ------------------------------------------------------------------
    "code_viewer": {
        "template_id": "code_viewer",
        "canvas_type": "code",
        "display_name": "Code Viewer",
        "description": "Syntax-highlighted code block with line numbers and copy button",
        "category": "technical",
        "data_schema": {
            "required": ["code"],
            "properties": {
                "code": {"type": "string", "description": "Source code text"},
                "language": {"type": "string", "description": "Language for syntax highlighting (python, js, etc.)"},
                "filename": {"type": "string"},
                "title": {"type": "string"},
            }
        },
        "default_config": {
            "show_line_numbers": True,
            "enable_copy": True,
            "theme": "dark",
        },
        "agent_hints": ["universal_agent", "coding_agent"],
    },

    "code_diff_viewer": {
        "template_id": "code_diff_viewer",
        "canvas_type": "code",
        "display_name": "Code Diff Viewer",
        "description": "Multi-file diff viewer with syntax highlighting, terminal log, and apply/reject buttons for coding agent output",
        "category": "technical",
        "data_schema": {
            "required": ["diffs"],
            "properties": {
                "diffs": {
                    "type": "array",
                    "description": "Array of file diffs",
                    "items": {
                        "type": "object",
                        "required": ["file", "diff"],
                        "properties": {
                            "file": {"type": "string", "description": "File path"},
                            "diff": {"type": "string", "description": "Unified diff text"},
                            "language": {"type": "string", "description": "Language for syntax highlighting"},
                            "status": {"type": "string", "enum": ["modified", "created", "deleted"]},
                        }
                    }
                },
                "terminal_log": {"type": "string", "description": "Terminal output log from the coding task"},
                "summary": {"type": "string", "description": "Natural language summary of changes"},
                "files_modified": {"type": "array", "items": {"type": "string"}, "description": "List of modified file paths"},
                "file_count": {"type": "integer", "description": "Total number of files changed"},
                "tests_passed": {"type": "boolean", "description": "Whether tests passed after changes (null if not run)"},
            }
        },
        "default_config": {
            "show_line_numbers": True,
            "enable_copy": True,
            "theme": "dark",
            "requires_confirmation": True,
            "confirmation_message": "Apply these code changes?",
        },
        "agent_hints": ["coding_agent"],
    },

    "json_tree": {
        "template_id": "json_tree",
        "canvas_type": "json",
        "display_name": "JSON Tree Viewer",
        "description": "Collapsible, syntax-highlighted JSON tree viewer",
        "category": "technical",
        "data_schema": {
            "required": ["data"],
            "properties": {
                "data": {"type": "object", "description": "JSON data to display"},
                "title": {"type": "string"},
                "collapsed_depth": {"type": "integer", "default": 2, "description": "Depth to collapse by default"},
            }
        },
        "default_config": {"enable_copy": True, "enable_search": True},
        "agent_hints": ["universal_agent", "browser_agent", "coding_agent"],
    },

    # ------------------------------------------------------------------
    # IMAGE TEMPLATE
    # ------------------------------------------------------------------
    "image_viewer": {
        "template_id": "image_viewer",
        "canvas_type": "image",
        "display_name": "Image Viewer",
        "description": "Image display with zoom and download",
        "category": "media",
        "data_schema": {
            "required": ["src"],
            "properties": {
                "src": {"type": "string", "description": "Image URL or base64 data URI"},
                "alt": {"type": "string"},
                "title": {"type": "string"},
            }
        },
        "default_config": {},
        "agent_hints": ["browser_agent", "universal_agent"],
    },
}


# ============================================================================
# Registry Access Functions
# ============================================================================

def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    """Get a template definition by ID. Returns None if not found."""
    return CANVAS_TEMPLATES.get(template_id)


def list_templates(category: Optional[str] = None, agent: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available templates, optionally filtered by category or agent.
    
    Args:
        category: Filter by category (data, communication, document, visualization, technical, media)
        agent: Filter by agent hint (spreadsheet_agent, mail_agent, etc.)
    """
    templates = list(CANVAS_TEMPLATES.values())
    
    if category:
        templates = [t for t in templates if t.get("category") == category]
    
    if agent:
        templates = [t for t in templates if agent in t.get("agent_hints", [])]
    
    return templates


def get_template_ids() -> List[str]:
    """Get all available template IDs."""
    return list(CANVAS_TEMPLATES.keys())


def validate_template_data(template_id: str, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate data against a template's schema.
    
    Returns:
        (True, None) if valid
        (False, error_message) if invalid
    """
    template = get_template(template_id)
    if not template:
        return False, f"Unknown template: {template_id}"
    
    schema = template.get("data_schema", {})
    required = schema.get("required", [])
    
    missing = [field for field in required if field not in data]
    if missing:
        return False, f"Missing required fields for '{template_id}': {missing}"
    
    return True, None


def get_canvas_type_for_template(template_id: str) -> Optional[str]:
    """Get the canvas_type that corresponds to a template_id."""
    template = get_template(template_id)
    return template["canvas_type"] if template else None

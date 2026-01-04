"""
Canvas Display Utilities for Agents

This module provides helper functions for agents to easily create
canvas displays for visual content presentation.
"""

from typing import Dict, Any, Optional, List, Literal
import base64
import json


CanvasType = Literal['html', 'markdown', 'pdf', 'spreadsheet', 'email_preview', 'document', 'image', 'json']


def create_canvas_display(
    canvas_type: CanvasType,
    canvas_data: Optional[Dict[str, Any]] = None,
    canvas_content: Optional[str] = None,
    canvas_title: Optional[str] = None,
    requires_confirmation: bool = False,
    confirmation_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized canvas display object.
    
    Two modes:
    1. Structured data (preferred): Pass canvas_data, frontend templates it
    2. Custom HTML (when needed): Pass canvas_content with raw HTML
    
    Args:
        canvas_type: Type of content ('email_preview', 'spreadsheet', 'document', etc.)
        canvas_data: Structured data for the canvas (frontend will template it) - PREFERRED
        canvas_content: Raw HTML/markdown content (use when custom rendering needed)
        canvas_title: Optional title for the canvas
        requires_confirmation: Whether user must confirm before proceeding
        confirmation_message: Message to show when confirmation is required
    
    Returns:
        Dictionary with canvas display data
    
    Examples:
        # Structured data (preferred)
        create_canvas_display(
            canvas_type="email_preview",
            canvas_data={"to": ["user@example.com"], "subject": "Hello"}
        )
        
        # Custom HTML (when needed)
        create_canvas_display(
            canvas_type="html",
            canvas_content="<div>Custom HTML here</div>"
        )
    """
    display = {
        "canvas_type": canvas_type,
        "canvas_title": canvas_title,
        "requires_confirmation": requires_confirmation,
        "confirmation_message": confirmation_message
    }
    
    # Add data or content (at least one should be provided)
    if canvas_data is not None:
        display["canvas_data"] = canvas_data
    if canvas_content is not None:
        display["canvas_content"] = canvas_content
    
    return display


def create_email_preview(
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    is_html: bool = False,
    attachments_count: int = 0,
    attachment_file_ids: Optional[List[str]] = None,
    attachment_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create an email preview canvas display with structured data.
    Frontend will handle the templating and rendering.
    
    Args:
        to: Recipient email addresses
        subject: Email subject
        body: Email body
        cc: CC recipients
        bcc: BCC recipients
        is_html: Whether body is HTML
        attachments_count: Number of attachments
        attachment_file_ids: List of attachment file IDs
        attachment_paths: List of attachment paths
    
    Returns:
        Canvas display dictionary with structured data
    """
    return create_canvas_display(
        canvas_type="email_preview",
        canvas_data={
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body": body,
            "is_html": is_html,
            "attachments": {
                "count": attachments_count,
                "file_ids": attachment_file_ids or [],
                "paths": attachment_paths or []
            }
        },
        canvas_title=f"Email Preview: {subject}",
        requires_confirmation=True,
        confirmation_message="Review and confirm to send this email"
    )


def create_spreadsheet_display(
    data: List[List[Any]],
    title: Optional[str] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a spreadsheet canvas display with structured data.
    Frontend will render it as a table.
    
    Args:
        data: 2D array of spreadsheet data (first row should be headers)
        title: Optional title
        filename: Optional filename
    
    Returns:
        Canvas display dictionary with structured data
    """
    headers = data[0] if data else []
    rows = data[1:] if len(data) > 1 else []
    
    return create_canvas_display(
        canvas_type="spreadsheet",
        canvas_data={
            "headers": headers,
            "rows": rows,
            "filename": filename,
            "row_count": len(rows),
            "column_count": len(headers)
        },
        canvas_title=title or f"Spreadsheet: {filename}" if filename else "Spreadsheet Data"
    )


def create_document_display(
    content: str,
    title: Optional[str] = None,
    format: Literal['text', 'markdown', 'html'] = 'text'
) -> Dict[str, Any]:
    """
    Create a document canvas display.
    
    Args:
        content: Document content
        title: Optional title
        format: Content format ('text', 'markdown', or 'html')
    
    Returns:
        Canvas display dictionary
    """
    canvas_type = 'document' if format == 'text' else format
    
    return create_canvas_display(
        canvas_type=canvas_type,
        canvas_content=content,
        canvas_title=title or "Document",
        canvas_metadata={"format": format, "length": len(content)}
    )


def create_pdf_display(
    pdf_bytes: bytes,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a PDF canvas display.
    
    Args:
        pdf_bytes: PDF file bytes
        title: Optional title
    
    Returns:
        Canvas display dictionary
    """
    # Convert to base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    
    return create_canvas_display(
        canvas_type="pdf",
        canvas_content=pdf_base64,
        canvas_title=title or "PDF Document",
        canvas_metadata={"size_bytes": len(pdf_bytes)}
    )


def create_image_display(
    image_bytes: bytes,
    title: Optional[str] = None,
    mime_type: str = "image/png"
) -> Dict[str, Any]:
    """
    Create an image canvas display.
    
    Args:
        image_bytes: Image file bytes
        title: Optional title
        mime_type: Image MIME type
    
    Returns:
        Canvas display dictionary
    """
    # Convert to base64 data URL
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:{mime_type};base64,{image_base64}"
    
    return create_canvas_display(
        canvas_type="image",
        canvas_content=data_url,
        canvas_title=title or "Image",
        canvas_metadata={"mime_type": mime_type, "size_bytes": len(image_bytes)}
    )


def create_json_display(
    data: Any,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a JSON canvas display.
    
    Args:
        data: Any JSON-serializable data
        title: Optional title
    
    Returns:
        Canvas display dictionary
    """
    json_str = json.dumps(data, indent=2, default=str)
    
    return create_canvas_display(
        canvas_type="json",
        canvas_content=json_str,
        canvas_title=title or "JSON Data",
        canvas_metadata={"keys": list(data.keys()) if isinstance(data, dict) else None}
    )


# Example usage documentation
"""
Example usage in agents:

from agents.canvas_utils import create_email_preview, create_spreadsheet_display

# In your agent endpoint:
@app.post("/send_email")
async def send_email(request: SendEmailRequest):
    # Generate email preview
    canvas_display = create_email_preview(
        to=request.to,
        subject=request.subject,
        body=request.body,
        cc=request.cc,
        attachments_count=len(request.attachments)
    )
    
    # Send email...
    result = await send_email_logic(request)
    
    # Return with canvas display
    return {
        "success": True,
        "result": result,
        "canvas_display": canvas_display
    }

# For spreadsheets:
@app.post("/analyze_data")
async def analyze_data(request: AnalyzeRequest):
    data = [
        ["Name", "Age", "City"],
        ["Alice", 30, "NYC"],
        ["Bob", 25, "LA"]
    ]
    
    canvas_display = create_spreadsheet_display(
        data=data,
        title="Analysis Results"
    )
    
    return {
        "success": True,
        "result": {"summary": "Analysis complete"},
        "canvas_display": canvas_display
    }
"""


def create_html_display(
    html_content: str,
    title: Optional[str] = None,
    requires_confirmation: bool = False,
    confirmation_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a custom HTML canvas display.
    Use this when you need custom rendering that doesn't fit standard templates.
    
    Args:
        html_content: Raw HTML content
        title: Optional title
        requires_confirmation: Whether user must confirm
        confirmation_message: Confirmation message
    
    Returns:
        Canvas display dictionary with HTML content
    
    Example:
        canvas = create_html_display(
            html_content="<div><h1>Custom Content</h1><p>...</p></div>",
            title="Custom Report"
        )
    """
    return create_canvas_display(
        canvas_type="html",
        canvas_content=html_content,
        canvas_title=title,
        requires_confirmation=requires_confirmation,
        confirmation_message=confirmation_message
    )


def create_markdown_display(
    markdown_content: str,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a markdown canvas display.
    Frontend will render it with proper markdown formatting.
    
    Args:
        markdown_content: Markdown formatted text
        title: Optional title
    
    Returns:
        Canvas display dictionary with markdown content
    
    Example:
        canvas = create_markdown_display(
            markdown_content="# Report\\n\\n## Summary\\n\\nResults...",
            title="Analysis Report"
        )
    """
    return create_canvas_display(
        canvas_type="markdown",
        canvas_content=markdown_content,
        canvas_title=title
    )


def create_spreadsheet_display(
    data: List[List[Any]],
    title: str,
    filename: str,
    display_mode: str = 'full',
    metadata: Optional[Dict[str, Any]] = None,
    file_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a spreadsheet canvas display.
    
    Args:
        data: 2D array where first row is headers, rest are data rows
        title: Display title
        filename: Original filename
        display_mode: 'full', 'query_result', 'comparison', 'statistics'
        metadata: Additional metadata (rows_total, operation, etc.)
        file_id: File ID for download link
    
    Returns:
        Canvas display dictionary
    """
    import time
    
    return create_canvas_display(
        canvas_type='spreadsheet',
        canvas_title=title,
        canvas_data={
            'data': data,
            'filename': filename,
            'display_mode': display_mode,
            'rows': len(data) - 1 if data else 0,
            'columns': len(data[0]) if data and len(data) > 0 else 0,
            'metadata': metadata or {},
            'file_id': file_id,
            'timestamp': int(time.time() * 1000),
            'no_cache': True
        }
    )

"""
Creation Task Handler - Routes document/spreadsheet creation tasks to dedicated endpoints.

This module detects and handles document and spreadsheet creation tasks within the 
orchestrator, routing them to dedicated `/api/documents/create` and `/api/spreadsheets/create`
endpoints instead of the general agent directory search.

Key Features:
- Detects creation intents from task names and descriptions
- Validates required parameters for document/spreadsheet creation
- Handles async execution of creation tasks
- Returns canvas preview for immediate UI display
"""

import re
import logging
import httpx
import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from schemas import PlannedTask

logger = logging.getLogger("CreationHandler")

# Keywords that indicate document/spreadsheet creation tasks
DOCUMENT_CREATE_KEYWORDS = {
    'create', 'generate', 'write', 'compose', 'draft', 'produce', 
    'make', 'build', 'form', 'compile', 'construct'
}

DOCUMENT_FILE_TYPES = {'pdf', 'docx', 'doc', 'txt', 'text', 'word', 'document'}
SPREADSHEET_FILE_TYPES = {'xlsx', 'csv', 'excel', 'spreadsheet', 'sheet'}

# Combined keywords for spreadsheets
SPREADSHEET_CREATE_KEYWORDS = DOCUMENT_CREATE_KEYWORDS | {'export', 'generate data', 'tabulate'}


def is_creation_task(task: PlannedTask) -> Tuple[bool, Optional[str]]:
    """
    Determines if a task is a document/spreadsheet creation task.
    
    Args:
        task: The PlannedTask to analyze
        
    Returns:
        Tuple of (is_creation, task_type)
        - is_creation: True if this is a creation task
        - task_type: 'document' or 'spreadsheet', None if not creation
    """
    task_name = task.task_name.lower() if task.task_name else ""
    task_desc = task.task_description.lower() if task.task_description else ""
    combined_text = f"{task_name} {task_desc}".lower()
    
    # Check for spreadsheet creation
    if _is_spreadsheet_creation(task_name, task_desc, combined_text):
        logger.info(f"[CREATION_HANDLER] Detected spreadsheet creation task: {task_name}")
        return True, 'spreadsheet'
    
    # Check for document creation
    if _is_document_creation(task_name, task_desc, combined_text):
        logger.info(f"[CREATION_HANDLER] Detected document creation task: {task_name}")
        return True, 'document'
    
    return False, None


def _is_spreadsheet_creation(task_name: str, task_desc: str, combined: str) -> bool:
    """Check if task is spreadsheet creation."""
    # First check for analysis/summary keywords - these are NOT creation tasks
    analysis_keywords = ['summarize', 'summary', 'analyze', 'analysis', 'describe', 'explain', 
                        'review', 'insights', 'examine', 'interpret', 'report about', 'report on']
    is_analysis = any(keyword in combined for keyword in analysis_keywords)
    
    # Check for modification keywords - these are NOT creation tasks when operating on uploaded files
    modification_keywords = ['add', 'append', 'insert', 'modify', 'update', 'change', 'edit', 'calculate',
                             'compute', 'average', 'sum', 'total', 'transform', 'apply', 'filter']
    is_modification = any(keyword in combined for keyword in modification_keywords)
    
    # If operating on existing/uploaded file, NOT creation
    has_existing_file = any(phrase in combined for phrase in ['uploaded', 'existing', 'current', 'this file', 
                                                                'the file', 'the sheet', 'the spreadsheet'])
    
    # If it's asking to analyze/summarize an existing file, NOT creation
    if is_analysis and any(phrase in combined for phrase in ['of the', 'about the', 'from the', 'existing', 'uploaded']):
        return False
    
    # If it's asking to modify an existing file, NOT creation
    if is_modification and has_existing_file:
        logger.info(f"[CREATION_HANDLER] Detected modification task (not creation): modification={is_modification}, has_existing={has_existing_file}")
        return False
    
    # Keywords must be present + file type or spreadsheet reference
    has_create_keyword = any(keyword in combined for keyword in SPREADSHEET_CREATE_KEYWORDS)
    has_file_type = any(ftype in combined for ftype in SPREADSHEET_FILE_TYPES)
    is_data_table_creation = 'create table' in combined or 'create data table' in combined or 'create spreadsheet' in combined
    
    # Exclude if it's clearly an analysis task even with creation keywords
    if is_analysis:
        return False
    
    return has_create_keyword and (has_file_type or is_data_table_creation)


def _is_document_creation(task_name: str, task_desc: str, combined: str) -> bool:
    """Check if task is document creation."""
    # EXCLUSION: Email/Mail Agent tasks should NEVER be creation tasks
    email_keywords = ['email', 'mail', 'mailbox', 'inbox', 'gmail', 'search_emails', 
                      'summarize_emails', 'send_email', 'draft_reply', 'fetch_emails']
    if any(keyword in combined for keyword in email_keywords):
        return False
    
    # EXCLUSION: Search/analysis tasks should NOT be creation tasks
    analysis_keywords = ['search', 'find', 'summarize', 'summary', 'analyze', 'analysis',
                         'list', 'describe', 'explain', 'review', 'idiom', 'extract']
    if any(keyword in combined for keyword in analysis_keywords):
        return False
    
    # Keywords must be present + file type or document reference
    has_create_keyword = any(keyword in combined for keyword in DOCUMENT_CREATE_KEYWORDS)
    has_file_type = any(ftype in combined for ftype in DOCUMENT_FILE_TYPES)
    is_doc_creation = 'create document' in combined or 'create file' in combined or 'create report' in combined
    
    # Make sure it's not a spreadsheet (to avoid overlap)
    is_spreadsheet = any(ftype in combined for ftype in SPREADSHEET_FILE_TYPES)
    
    return has_create_keyword and (has_file_type or is_doc_creation) and not is_spreadsheet


def extract_creation_parameters(task: PlannedTask) -> Dict[str, Any]:
    """
    Extracts parameters needed for document/spreadsheet creation from task.
    
    Returns:
        Dictionary with:
        - content: The text content to write (from task_description if not in params)
        - file_name: The desired file name
        - file_type: The file format (pdf, docx, csv, xlsx, txt)
        - data: For spreadsheets, the structured data
    """
    params = task.parameters or {}
    task_desc = task.task_description or ""
    
    creation_params = {
        'content': params.get('content') or task_desc,
        'file_name': params.get('file_name') or params.get('filename') or _extract_filename(task_desc),
        'file_type': params.get('file_type') or params.get('file_format') or _extract_file_type(task_desc),
        'data': params.get('data') or params.get('rows') or params.get('spreadsheet_data'),
    }
    
    logger.info(f"[CREATION_HANDLER] Extracted params: file_name={creation_params['file_name']}, file_type={creation_params['file_type']}")
    
    return creation_params


def _extract_filename(text: str) -> Optional[str]:
    """
    Attempts to extract filename from text.
    Looks for patterns like 'file_name:', 'as', 'named', or quoted strings.
    """
    # Pattern: "file named X" or "file 'X'" or "as X"
    patterns = [
        r'(?:file\s+)?named?\s+["\']?([^\s"\']+)["\']?',
        r'as\s+["\']?([^\s"\']+)["\']?',
        r'(?:save|write|create)\s+(?:file\s+)?["\']?([^\s"\']+)["\']?',
        r'filename[:\s]+["\']?([^\s"\']+)["\']?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            filename = match.group(1)
            # Clean up the filename
            if not any(filename.lower().endswith(ext) for ext in ['.pdf', '.docx', '.xlsx', '.csv', '.txt']):
                filename = f"{filename}.txt"  # Default extension
            return filename
    
    return None


def _extract_file_type(text: str) -> Optional[str]:
    """
    Attempts to extract file type from text.
    Looks for explicit mentions of file formats or file extensions.
    """
    # Direct format mentions: "pdf", "docx", "xlsx", "csv", etc.
    format_patterns = r'\b(pdf|docx|xlsx?|csv|txt|word|document|spreadsheet|excel)\b'
    match = re.search(format_patterns, text, re.IGNORECASE)
    if match:
        fmt = match.group(1).lower()
        # Normalize variations
        if fmt in ['doc', 'word']: return 'docx'
        if fmt == 'excel': return 'xlsx'
        if fmt == 'spreadsheet': return 'xlsx'
        if fmt == 'document': return 'docx'
        return fmt
    
    return None


def validate_creation_task(task: PlannedTask, task_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validates that a creation task has the necessary parameters.
    
    Args:
        task: The PlannedTask
        task_type: 'document' or 'spreadsheet'
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    params = extract_creation_parameters(task)
    
    if task_type == 'document':
        # Documents need content and file_type
        if not params.get('content') or params.get('content').strip() == "":
            return False, "Document content is empty. Please provide content to create."
        if not params.get('file_type'):
            return False, "File type not specified. Please specify format (pdf, docx, txt)."
        if params.get('file_type') not in ['pdf', 'docx', 'doc', 'txt', 'text']:
            return False, f"Unsupported document format: {params.get('file_type')}. Use pdf, docx, or txt."
    
    elif task_type == 'spreadsheet':
        # Spreadsheets need data and file_type
        if not params.get('data') and not params.get('content'):
            return False, "Spreadsheet data is empty. Please provide data to create."
        if not params.get('file_type'):
            return False, "File type not specified. Please specify format (xlsx, csv)."
        if params.get('file_type') not in ['xlsx', 'csv', 'excel', 'spreadsheet']:
            return False, f"Unsupported spreadsheet format: {params.get('file_type')}. Use xlsx or csv."
    
    return True, None


def build_creation_payload(task: PlannedTask, task_type: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds the HTTP request payload for creation endpoint.
    
    Args:
        task: The PlannedTask
        task_type: 'document' or 'spreadsheet'
        thread_id: Optional thread ID for conversation tracking
        
    Returns:
        Dictionary to send to /api/documents/create or /api/spreadsheets/create
    """
    params = extract_creation_parameters(task)
    
    payload = {
        'thread_id': thread_id,
    }
    
    if task_type == 'document':
        payload['file_type'] = params.get('file_type', 'txt').lower()
        payload['content'] = params.get('content', task.task_description)
        payload['file_name'] = params.get('file_name') or 'document.txt'
    
    elif task_type == 'spreadsheet':
        # For spreadsheets, check if data is string (CSV) or dict/list (structured)
        data = params.get('data') or params.get('content', '{}')
        
        # If it's a string that looks like JSON, parse it
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except:
                # If not JSON, treat as CSV-like string and parse it
                data = _parse_csv_string(data)
        
        payload['data'] = data
        # FIX: Use 'filename' and 'file_format' for spreadsheet API (not file_name and file_type)
        payload['filename'] = params.get('file_name') or params.get('filename') or 'spreadsheet.xlsx'
        payload['file_format'] = params.get('file_type', 'xlsx').lower()
    
    logger.info(f"[CREATION_HANDLER] Built payload for {task_type}: {list(payload.keys())}")
    return payload


def _parse_csv_string(csv_string: str) -> Dict[str, Any]:
    """
    Parses a CSV string into a dict structure for spreadsheet creation.
    
    Format expected: "name,age\nJohn,30\nJane,25"
    
    Returns:
        Dict with 'columns' and 'rows' keys
    """
    lines = [line.strip() for line in csv_string.strip().split('\n') if line.strip()]
    if not lines:
        return {'columns': [], 'rows': []}
    
    columns = [col.strip() for col in lines[0].split(',')]
    rows = []
    for line in lines[1:]:
        row = [val.strip() for val in line.split(',')]
        rows.append(row)
    
    return {
        'columns': columns,
        'rows': rows,
    }


async def execute_creation_task_async(task: PlannedTask, task_type: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes a creation task by calling the dedicated creation endpoints.
    
    Args:
        task: The PlannedTask to execute
        task_type: 'document' or 'spreadsheet'
        thread_id: Optional thread ID for conversation tracking
        
    Returns:
        Dict with status, result, canvas_display, etc.
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get API URL from environment or use localhost
    api_base_url = os.getenv('ORCHESTRATOR_API_URL', 'http://localhost:8000').rstrip('/')
    
    if task_type == 'document':
        endpoint = f'{api_base_url}/api/documents/create'
    elif task_type == 'spreadsheet':
        endpoint = f'{api_base_url}/api/spreadsheets/create'
    else:
        return {
            'status': 'failed',
            'result': f'Unknown creation task type: {task_type}',
            'task_name': task.task_name,
        }
    
    # Build the payload
    payload = build_creation_payload(task, task_type, thread_id)
    
    logger.info(f"[CREATION_HANDLER] Executing {task_type} creation task at {endpoint}")
    logger.info(f"[CREATION_HANDLER] Payload keys: {list(payload.keys())}")
    
    try:
        # Use httpx for async HTTP calls
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'X-Internal-Request': 'true',  # Bypass auth for internal calls
                }
            )
            
            logger.info(f"[CREATION_HANDLER] Response status: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"[CREATION_HANDLER] Creation succeeded. File path: {response_data.get('file_path')}")
                
                # Extract canvas display from response
                canvas_display = {
                    'canvas_type': 'file_preview',
                    'canvas_title': f'Created: {response_data.get("file_name", "file")}',
                    'canvas_html': f'<div class="file-preview"><p>File created successfully: <strong>{response_data.get("file_name")}</strong></p><p>Type: {response_data.get("file_type")}</p></div>',
                    'preview_url': response_data.get('preview_url'),
                    'canvas_display': response_data.get('canvas_display'),  # Include agent's own canvas display
                }
                
                return {
                    'status': 'success',
                    'task_name': task.task_name,
                    'result': response_data,
                    'canvas_display': canvas_display,
                    'execution_time': 0,  # Will be set by execute_batch
                }
            else:
                error_msg = response.text
                logger.error(f"[CREATION_HANDLER] Creation failed: {response.status_code} - {error_msg}")
                
                return {
                    'status': 'failed',
                    'task_name': task.task_name,
                    'result': f'Error: {response.status_code} - {error_msg}',
                    'raw_response': error_msg,
                    'execution_time': 0,
                }
    
    except asyncio.TimeoutError:
        error_msg = f"Creation task timeout after 60 seconds"
        logger.error(f"[CREATION_HANDLER] {error_msg}")
        return {
            'status': 'failed',
            'task_name': task.task_name,
            'result': f'Error: {error_msg}',
            'execution_time': 60,
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[CREATION_HANDLER] Exception during creation: {e}", exc_info=True)
        return {
            'status': 'failed',
            'task_name': task.task_name,
            'result': f'Error: {error_msg}',
            'execution_time': 0,
        }


__all__ = [
    'is_creation_task',
    'extract_creation_parameters',
    'validate_creation_task',
    'build_creation_payload',
    'execute_creation_task_async',
]

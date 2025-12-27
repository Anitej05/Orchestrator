"""
Agent Utilities Module

Common utilities used by agents for file management, canvas displays, etc.
"""

from .agent_file_manager import (
    AgentFileManager,
    FileType,
    FileStatus,
    AgentFileMetadata,
    FileTypeDetector
)

from .canvas_utils import (
    create_canvas_display,
    create_email_preview,
    create_spreadsheet_display,
    create_document_display,
    create_pdf_display,
    create_image_display,
    create_json_display,
    create_html_display,
    create_markdown_display
)

from .standard_file_interface import (
    StandardFileHandler,
    AgentFileMetadata as StandardAgentFileMetadata,
    FileUploadResponse,
    FileMetadataResponse,
    create_file_endpoints,
    setup_standard_file_handling
)

__all__ = [
    # File Manager
    'AgentFileManager',
    'FileType',
    'FileStatus',
    'AgentFileMetadata',
    'FileTypeDetector',
    
    # Canvas Utils
    'create_canvas_display',
    'create_email_preview',
    'create_spreadsheet_display',
    'create_document_display',
    'create_pdf_display',
    'create_image_display',
    'create_json_display',
    'create_html_display',
    'create_markdown_display',
    
    # Standard File Interface
    'StandardFileHandler',
    'StandardAgentFileMetadata',
    'FileUploadResponse',
    'FileMetadataResponse',
    'create_file_endpoints',
    'setup_standard_file_handling',
]

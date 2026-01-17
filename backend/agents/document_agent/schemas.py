"""
Document Agent - Data Schemas

Pydantic models for API contracts, document operations, and data validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


# ============================================================================
# ORCHESTRATOR-COMPATIBLE STATUS FIELDS (OPTIONAL)
# ============================================================================

class AgentResponseStatus(str, Enum):
    """Status compatible with backend.schemas.AgentResponseStatus."""
    COMPLETE = "complete"
    ERROR = "error"
    NEEDS_INPUT = "needs_input"
    PARTIAL = "partial"


# ============================================================================
# ENUMS
# ============================================================================

class DocumentType(str, Enum):
    """Supported document types."""
    DOCX = "docx"
    PDF = "pdf"
    TXT = "txt"


class EditActionType(str, Enum):
    """Types of document edit actions."""
    FORMAT_TEXT = "format_text"
    ADD_CONTENT = "add_content"
    REPLACE_CONTENT = "replace_content"
    DELETE_CONTENT = "delete_content"
    ADD_HEADING = "add_heading"
    ADD_TABLE = "add_table"
    ADD_IMAGE = "add_image"
    MODIFY_STYLE = "modify_style"
    EXTRACT_DATA = "extract_data"
    CONVERT_FORMAT = "convert_format"


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class AnalyzeDocumentRequest(BaseModel):
    """Request to analyze documents using RAG."""
    vector_store_path: Optional[str] = Field(None, description="Path to single FAISS vector store")
    vector_store_paths: Optional[List[str]] = Field(None, description="Paths to multiple FAISS vector stores")
    query: str = Field(..., description="User's question about the document(s)")
    file_path: Optional[str] = Field(None, description="Optional path to the source document for direct content extraction")
    file_paths: Optional[List[str]] = Field(None, description="Optional list of document paths for batch processing")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID for context")
    max_workers: Optional[int] = Field(4, description="Maximum concurrent threads for multi-file processing")
    include_per_file_results: Optional[bool] = Field(False, description="Include individual results for each file")


class FileAnalysisResult(BaseModel):
    """Result for a single file analysis."""
    file_path: str
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None


class AnalyzeDocumentResponse(BaseModel):
    """Response with analysis results."""
    success: bool
    answer: str
    canvas_display: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None
    status: Optional[str] = Field(
        None,
        description="Execution status for orchestrator compatibility (complete | needs_input | error)",
    )
    phase_trace: Optional[List[str]] = Field(
        None,
        description="Ordered phases executed for analysis",
    )
    confidence: Optional[float] = Field(None, description="Confidence score for grounded answer")
    grounding: Optional[Dict[str, Any]] = Field(
        None,
        description="Grounding metadata such as chunk ids, source files, and validation issues",
    )
    file_results: Optional[List[FileAnalysisResult]] = Field(None, description="Per-file results for batch processing")
    total_files: Optional[int] = Field(None, description="Total number of files processed")
    successful_files: Optional[int] = Field(None, description="Number of successfully processed files")
    failed_files: Optional[int] = Field(None, description="Number of failed files")
    errors: Optional[List[str]] = Field(None, description="List of errors encountered")

    # Enterprise/orchestrator metadata (non-breaking)
    status: Optional[AgentResponseStatus] = None
    question: Optional[str] = None
    question_type: Optional[str] = None
    pending_plan: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    phase_trace: Optional[List[str]] = None
    grounding: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    review_required: Optional[bool] = None


class DisplayDocumentRequest(BaseModel):
    """Request to display a document."""
    file_path: str = Field(..., description="Path to the document file (PDF, DOCX, TXT)")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class DisplayDocumentResponse(BaseModel):
    """Response with document display data."""
    success: bool
    message: str
    canvas_display: Dict[str, Any]
    file_type: str


class CreateDocumentRequest(BaseModel):
    """Request to create a new document."""
    content: str = Field(..., description="Content for the document")
    file_name: str = Field(..., description="Name for the document file (e.g., 'report.docx')")
    file_type: DocumentType = Field(default=DocumentType.DOCX, description="Document type")
    output_dir: str = Field(default="storage/documents", description="Output directory")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class CreateDocumentResponse(BaseModel):
    """Response after creating a document."""
    success: bool
    message: str
    file_path: str
    canvas_display: Optional[Dict[str, Any]] = None


class EditDocumentRequest(BaseModel):
    """Request to edit a document using natural language."""
    file_path: str = Field(..., description="Path to the document to edit")
    instruction: str = Field(..., description="Natural language instruction describing the edit")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")
    use_vision: bool = Field(default=False, description="Use vision-based planning if available")
    auto_approve: bool = Field(
        default=False,
        description="Auto-approve high-risk edit plans without pausing for confirmation",
    )

    # Enterprise: allow orchestrator to resume/approve a paused edit.
    auto_approve: bool = Field(default=False, description="If true, bypass approval gating and execute the plan")
    approval_response: Optional[str] = Field(default=None, description="Optional user approval text/answer (for audit)")


class EditDocumentResponse(BaseModel):
    """Response after editing a document."""
    success: bool
    message: str
    file_path: str
    canvas_display: Optional[Dict[str, Any]] = None
    can_undo: bool = False
    can_redo: bool = False
    edit_summary: Optional[str] = None
    status: Optional[str] = Field(
        None,
        description="Execution status for orchestrator compatibility (complete | needs_input | error)",
    )
    phase_trace: Optional[List[str]] = Field(
        None,
        description="Ordered phases executed for the edit flow",
    )
    question: Optional[str] = Field(None, description="Clarifying or approval question when paused")
    pending_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Serialized plan awaiting approval when status is needs_input",
    )
    risk_assessment: Optional[Dict[str, Any]] = Field(
        None,
        description="Risk/intention classification for the edit instruction",
    )


class UndoRedoRequest(BaseModel):
    """Request to undo/redo an edit."""
    file_path: str = Field(..., description="Path to the document")
    action: str = Field(..., description="'undo' or 'redo'")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class UndoRedoResponse(BaseModel):
    """Response after undo/redo."""
    success: bool
    message: str
    file_path: str
    canvas_display: Optional[Dict[str, Any]] = None
    can_undo: bool = False
    can_redo: bool = False


class VersionHistoryRequest(BaseModel):
    """Request to get version history."""
    file_path: str = Field(..., description="Path to the document")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class VersionHistoryResponse(BaseModel):
    """Response with version history."""
    success: bool
    message: str
    versions: List[Dict[str, Any]]
    current_version: int


class ExtractDataRequest(BaseModel):
    """Request to extract data from a document."""
    file_path: str = Field(..., description="Path to the document")
    extraction_type: str = Field(..., description="'text', 'tables', 'structured'")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class ExtractDataResponse(BaseModel):
    """Response with extracted data."""
    success: bool
    message: str
    extracted_data: Dict[str, Any]
    data_format: str
    status: Optional[str] = Field(
        None,
        description="Execution status for orchestrator compatibility (complete | needs_input | error)",
    )
    phase_trace: Optional[List[str]] = Field(
        None,
        description="Ordered phases executed for extraction",
    )
    confidence: Optional[float] = Field(None, description="Confidence score for extraction")
    grounding: Optional[Dict[str, Any]] = Field(None, description="Grounding metadata for extracted fields")


class DocumentStructure(BaseModel):
    """Represents document structure for analysis."""
    styles_used: Dict[str, int]
    headings: List[Dict[str, str]]
    table_count: int
    total_paragraphs: int
    total_sections: int
    file_name: str


class EditAction(BaseModel):
    """Represents a single edit action."""
    action_type: EditActionType
    description: str
    parameters: Dict[str, Any]
    timestamp: str
    success: bool
    message: Optional[str] = None

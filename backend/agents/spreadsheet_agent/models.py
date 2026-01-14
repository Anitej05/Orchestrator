"""
Pydantic models for requests, responses, and data structures
"""
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, model_serializer
import numpy as np


# ============== RESPONSE MODELS ==============

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    canvas_display: Optional[Dict[str, Any]] = None  # For orchestrator approval integration
    
    @model_serializer
    def serialize_model(self):
        """Custom serializer to handle numpy types"""
        def convert_numpy(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj
        
        return {
            'success': self.success,
            'result': convert_numpy(self.result),
            'error': self.error,
            'canvas_display': self.canvas_display
        }


class SummaryResponse(BaseModel):
    """Response model for spreadsheet summary"""
    filename: str
    headers: list
    rows: list
    dtypes: dict


class QueryResponse(BaseModel):
    """Response model for query results"""
    query_result: list


class StatsResponse(BaseModel):
    """Response model for column statistics"""
    column_stats: dict


# ============== REQUEST MODELS ==============

class CreateSpreadsheetRequest(BaseModel):
    """Request model for LLM-powered spreadsheet creation"""
    content: Optional[str] = Field(None, description="Raw CSV/JSON text or natural language content")
    reference_file_id: Optional[str] = Field(None, description="File ID of existing file to transform")
    instruction: Optional[str] = Field(None, description="Natural language instruction")
    output_format: str = Field(default="csv", description="Output format: 'csv' or 'xlsx'")
    output_filename: Optional[str] = Field(None, description="Optional filename for output")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class NaturalLanguageQueryRequest(BaseModel):
    """Request model for natural language queries"""
    file_id: str = Field(..., description="ID of the uploaded file")
    question: str = Field(..., description="Natural language question")
    max_iterations: int = Field(default=5, description="Maximum reasoning iterations")


class CompareFilesRequest(BaseModel):
    """Request model for comparing multiple spreadsheet files"""
    file_ids: List[str] = Field(..., min_length=2, description="List of file IDs to compare (2 or more)")
    comparison_mode: str = Field(default="schema_and_key", description="Comparison mode: 'schema_only', 'schema_and_key', 'full_diff'")
    key_columns: Optional[List[str]] = Field(None, description="Columns to use as keys for row matching")
    output_format: str = Field(default="json", description="Output format: 'json', 'csv', 'xlsx'")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


class MergeFilesRequest(BaseModel):
    """Request model for merging multiple spreadsheet files"""
    file_ids: List[str] = Field(..., min_length=2, description="List of file IDs to merge (2 or more)")
    merge_type: str = Field(default="join", description="Merge type: 'join', 'union', 'concat'")
    join_type: Optional[str] = Field("inner", description="For join: 'inner', 'left', 'right', 'outer'")
    key_columns: Optional[List[str]] = Field(None, description="Columns to join/merge on")
    output_filename: Optional[str] = Field(None, description="Optional filename for merged output")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID")


# ============== OPERATION TRACKING ==============

class QueryPlan(BaseModel):
    """A single step in the query plan"""
    step: int
    description: str
    pandas_code: str
    reasoning: str


class ObservationData(BaseModel):
    """Observation data showing before/after state of spreadsheet operations"""
    before_shape: Tuple[int, int]  # (rows, cols)
    after_shape: Tuple[int, int]
    before_columns: List[str]
    after_columns: List[str]
    before_sample: List[Dict[str, Any]]  # First 3 rows
    after_sample: List[Dict[str, Any]]
    changes_summary: str
    columns_added: List[str] = []
    columns_removed: List[str] = []


class ComparisonResult(BaseModel):
    """Result of comparing two or more spreadsheet files"""
    file_ids: List[str]
    schema_diff: Dict[str, Any]  # Column differences, dtype changes
    row_diff: Optional[Dict[str, Any]] = None  # Added/removed/changed rows
    summary: str
    diff_artifact_id: Optional[str] = None  # ID of generated diff report file
    columns_renamed: Dict[str, str] = {}  # old -> new
    rows_added: int = 0
    rows_removed: int = 0
    data_modified: bool = False


class UserChoice(BaseModel):
    """Single user choice option for anomaly resolution"""
    id: str  # e.g., "convert_numeric", "ignore_rows", "treat_as_text", "cancel"
    label: str  # Human-readable label
    description: str  # What this action does
    is_safe: bool = True  # Whether this action is non-destructive


class AnomalyDetails(BaseModel):
    """Detailed information about detected anomaly"""
    anomaly_type: str  # e.g., "dtype_drift", "missing_values", "outliers"
    affected_columns: List[str]
    message: str  # Human-readable explanation
    current_dtypes: Dict[str, str]  # Column -> current dtype
    expected_dtypes: Optional[Dict[str, str]] = None  # Column -> expected dtype
    sample_values: Optional[Dict[str, List[Any]]] = None  # Column -> problematic values
    severity: str = "warning"  # "info", "warning", "error"


class QueryResult(BaseModel):
    """Result of a natural language query"""
    question: str
    answer: str
    steps_taken: List[Dict[str, Any]]
    final_data: Optional[List[Dict]] = None
    success: bool
    error: Optional[str] = None
    execution_metrics: Optional[Dict[str, Any]] = None  # Add metrics field
    final_dataframe: Optional[Any] = None  # Store the modified DataFrame
    observation: Optional[ObservationData] = None  # Track changes made
    
    # Anomaly detection and user input handling
    status: str = "completed"  # "completed", "anomaly_detected", "failed"
    needs_user_input: bool = False
    anomaly: Optional[AnomalyDetails] = None
    user_choices: Optional[List[UserChoice]] = None
    pending_action: Optional[str] = None  # Action waiting for user decision
    
    class Config:
        arbitrary_types_allowed = True
    
    # Not a pydantic field, but carries the dataframe state
    final_dataframe: Any = None

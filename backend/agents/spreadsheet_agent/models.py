"""
Pydantic models for requests, responses, and data structures
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, model_serializer
import numpy as np


# ============== RESPONSE MODELS ==============

class ApiResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    
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
            'error': self.error
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


# ============== OPERATION TRACKING ==============

class QueryPlan(BaseModel):
    """A single step in the query plan"""
    step: int
    description: str
    pandas_code: str
    reasoning: str


class QueryResult(BaseModel):
    """Result of a natural language query"""
    question: str
    answer: str
    steps_taken: List[Dict[str, Any]]
    final_data: Optional[List[Dict]] = None
    success: bool
    error: Optional[str] = None
    execution_metrics: Optional[Dict[str, Any]] = None  # Add metrics field
    
    class Config:
        arbitrary_types_allowed = True
    
    # Not a pydantic field, but carries the dataframe state
    final_dataframe: Any = None

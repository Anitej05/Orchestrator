# In Project_Agent_Directory/schemas.py

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Literal, Optional, Dict, Any
from cryptography.hazmat.primitives import serialization

# --- Core Data Structures ---

class EndpointParameterDetail(BaseModel):
    """Defines the structure for a single parameter of an agent's API endpoint."""
    name: str
    description: Optional[str] = None
    param_type: str
    required: bool = True
    default_value: Optional[str] = None

    class Config:
        from_attributes = True

class EndpointDetail(BaseModel):
    """Defines the structure for an agent's API endpoint."""
    endpoint: str  # Changed from HttpUrl to support MCP tool names
    http_method: str
    description: Optional[str] = None
    parameters: List[EndpointParameterDetail] = []

    class Config:
        from_attributes = True

class AgentCard(BaseModel):
    """The main schema for an agent's registration and data."""
    id: str
    owner_id: str
    name: str
    description: str
    capabilities: List[str]
    price_per_call_usd: float
    status: Literal['active', 'inactive', 'deprecated'] = 'active'
    endpoints: List[EndpointDetail]
    rating: float = 0.0
    public_key_pem: Optional[str] = None
    agent_type: Literal['http_rest', 'mcp_http'] = 'http_rest'
    connection_config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
    
    @field_validator('status', mode='before')
    def serialize_status(cls, v):
        """Convert StatusEnum to string"""
        if hasattr(v, 'value'):
            return v.value
        return v
    
    @field_validator('agent_type', mode='before')
    def serialize_agent_type(cls, v):
        """Convert AgentType enum to string"""
        if hasattr(v, 'value'):
            return v.value
        return v

    @field_validator('public_key_pem')
    def validate_public_key(cls, pem: Optional[str]):
        if pem is None:
            return pem
        
        # 1. Un-escape newline characters and strip whitespace
        clean_pem = pem.replace('\\n', '\n').strip()

        # 2. The rest of the validation remains the same
        if not clean_pem.startswith('-----BEGIN PUBLIC KEY-----') or not clean_pem.endswith('-----END PUBLIC KEY-----'):
            raise ValueError('Invalid PEM format.')
        try:
            serialization.load_pem_public_key(clean_pem.encode())
        except Exception as e:
            raise ValueError(f"Invalid PEM public key format: {e}")
        
        # 3. Return the cleaned, correct key
        return clean_pem

# --- Orchestrator Models ---

class Task(BaseModel):
    """Represents a single, discrete task parsed from the user's prompt."""
    task_name: str = Field(..., description="A short, actionable name for the task.")
    task_description: str = Field(..., description="A detailed description of the task.")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Pre-extracted parameters for the task (e.g., {'ticker': 'AAPL'})")

class ParsedRequest(BaseModel):
    """The output of the initial prompt parsing node."""
    tasks: List[Task]
    user_expectations: Optional[Dict[str, float]] = None
    direct_response: Optional[str] = Field(None, description="If set, skip the graph and return this immediately (for greetings, thanks, etc.)")
    suggested_title: Optional[str] = Field(None, description="Suggested conversation title (generated on first turn)")

class TaskAgentPair(BaseModel):
    """Pairs a task with its ranked primary and fallback agents."""
    task_name: str
    task_description: str
    primary: AgentCard
    fallbacks: List[AgentCard]

class ExecutionStep(BaseModel):
    """A concise, executable step for a single agent call in the final plan."""
    id: str
    http_method: str
    endpoint: str  # Changed from HttpUrl to support MCP tool names
    payload: Dict[str, Any]

class PlannedTask(BaseModel):
    """A task as defined within the final execution plan."""
    task_name: str
    task_description: str
    primary: ExecutionStep
    fallbacks: List[AgentCard] = []

class ExecutionPlan(BaseModel):
    """The final, structured execution plan with parallel batches."""
    plan: List[List[PlannedTask]]

# --- API Request/Response Models ---

class ProcessResponse(BaseModel):
    """The schema for the main `/api/chat` endpoint response."""
    message: str
    thread_id: str
    task_agent_pairs: List[TaskAgentPair]
    final_response: Optional[str] = None
    pending_user_input: bool = False
    question_for_user: Optional[str] = None
    # Canvas fields
    has_canvas: bool = False
    canvas_content: Optional[str] = None
    canvas_type: Optional[str] = None
    browser_view: Optional[str] = None
    plan_view: Optional[str] = None
    current_view: Optional[str] = None
    
class PlanResponse(BaseModel):
    """The schema for the GET /api/plan/{thread_id} endpoint response."""
    thread_id: str
    content: str

# --- Utility Models for Graph Nodes ---

class SelectedEndpoint(BaseModel):
    """Used to validate the LLM's choice of endpoint for a task."""
    endpoint: str  # Changed from HttpUrl to support MCP tool names
    http_method: str

class PriorityMapper(BaseModel):
    """Used to map a task to a priority level."""
    task_name: str
    priority: str

class PriorityMappingResponse(BaseModel):
    """The expected response when asking the LLM to prioritize tasks."""
    mappings: List[PriorityMapper]

class FileObject(BaseModel):
    """Represents a file uploaded by the user (legacy - for backward compatibility)."""
    file_name: str
    file_path: str
    file_type: Literal['image', 'document']
    base64_content: Optional[str] = None
    vector_store_path: Optional[str] = None

class EnrichedFileObject(BaseModel):
    """File with semantic content understanding for orchestrator context."""
    # Basic metadata
    file_name: str
    file_path: str
    file_type: Literal['image', 'document']
    
    # Semantic content (KEY FEATURE - orchestrator understands file contents)
    content_summary: str = Field(..., description="AI-generated description/summary of file content")
    
    # Vector data for retrieval
    vector_store_path: Optional[str] = Field(None, description="Path to FAISS vector store for documents")
    clip_embedding: Optional[List[float]] = Field(None, description="CLIP embedding vector for images")
    
    # Additional metadata
    chunk_count: Optional[int] = Field(None, description="Number of text chunks for documents")
    dimensions: Optional[Dict[str, int]] = Field(None, description="Image dimensions (width, height)")
    upload_timestamp: str = Field(..., description="ISO timestamp when file was uploaded")

class ConversationTurn(BaseModel):
    """Single turn in conversation history for persistent memory."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    attached_files: List[str] = Field(default_factory=list, description="File names referenced in this turn")

class ProcessRequest(BaseModel):
    prompt: str
    thread_id: Optional[str] = None
    user_response: Optional[str] = None
    files: Optional[List[FileObject]] = None # Accepts basic FileObject, will be enriched internally

class AnalysisResult(BaseModel):
    """Schema for the analysis node's output."""
    needs_complex_processing: bool = Field(..., description="Whether the request requires complex orchestration or can be handled with a simple response")
    reasoning: str = Field(..., description="The reasoning behind the analysis decision")
    response: Optional[str] = Field(None, description="Direct response if needs_complex_processing is False")

class PlanValidationResult(BaseModel):
    """Schema for the advanced validation node's output."""
    status: Literal["ready", "replan_needed", "user_input_required"] = Field(..., description="The status of the plan validation.")
    reasoning: Optional[str] = Field(None, description="Required explanation if status is 'replan_needed' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The direct question for the user if input is absolutely required.")

class AgentResponseEvaluationEnhanced(BaseModel):
    """Enhanced schema for evaluating agent responses with reactive routing support."""
    status: Literal["complete", "partial_success", "failed", "user_input_required"] = Field(..., description="The status of the agent response evaluation.")
    reasoning: str = Field(..., description="Explanation of the evaluation decision.")
    feedback_for_replanning: Optional[str] = Field(None, description="Specific feedback for the planner if status is 'failed'.")
    question: Optional[str] = Field(None, description="Question to ask the user if status is 'user_input_required'.")

class FinalResponse(BaseModel):
    """Schema for unified final response generation (text + canvas)."""
    response_text: str = Field(..., description="The text response for the user")
    canvas_required: bool = Field(..., description="Whether canvas visualization is needed")
    canvas_type: Optional[Literal["html", "markdown"]] = Field(None, description="Type of canvas content")
    canvas_content: Optional[str] = Field(None, description="The actual canvas content (HTML or Markdown)")

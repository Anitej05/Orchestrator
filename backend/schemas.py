# In Orbimesh Backend/schemas.py

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Literal, Optional, Dict, Any
from enum import Enum
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
    request_format: Optional[str] = None  # 'json' or 'form', defaults to agent's connection_config

    class Config:
        from_attributes = True

class AgentCard(BaseModel):
    """The main schema for an agent's registration and data."""
    id: str
    owner_id: str
    name: str
    description: str
    capabilities: List[str] | Dict[str, Any] | None = []  # Optional; accepts old list format, structured, or None
    price_per_call_usd: float
    status: Literal['active', 'inactive', 'deprecated'] = 'active'
    endpoints: List[EndpointDetail]
    rating: float = 0.0
    public_key_pem: Optional[str] = None
    agent_type: Literal['http_rest', 'mcp_http', 'tool'] = 'http_rest'  # Added 'tool' type
    connection_config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
    
    @field_validator('capabilities', mode='before')
    def normalize_capabilities(cls, v):
        """Convert new structured format to flat list for backward compatibility"""
        if v is None:
            return []
        if isinstance(v, dict):
            # New structured format - extract all_keywords
            return v.get('all_keywords', [])
        # Old flat list format - return as-is
        return v
    
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
        if pem is None or pem == "YOUR_PUBLIC_KEY_HERE":
            return None  # Allow placeholder or None
        
        # 1. Un-escape newline characters and strip whitespace
        clean_pem = pem.replace('\\n', '\n').strip()

        # 2. The rest of the validation remains the same
        if not clean_pem.startswith('-----BEGIN PUBLIC KEY-----') or not clean_pem.endswith('-----END PUBLIC KEY-----'):
            raise ValueError('Invalid PEM format.')
        try:
            serialization.load_pem_public_key(clean_pem.encode())
        except Exception as e:
            # If validation fails, return None instead of raising error (for development)
            return None
        
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
    # Pre-extracted/injected parameters for the task (used by tools and some handlers).
    # NOTE: This field is referenced throughout the orchestrator codebase.
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    primary: ExecutionStep
    fallbacks: List[AgentCard] = []

    # Optional routing metadata (safe to ignore if unused).
    route_type: Optional[Literal["tool", "agent"]] = None
    selected_tool_capability: Optional[str] = None

class ExecutionPlan(BaseModel):
    """The final, structured execution plan with parallel batches."""
    plan: List[List[PlannedTask]]


# --- Multi-Turn Agent Dialogue Models ---

class DialogueAction(BaseModel):
    """A single action in the dialogue loop."""
    endpoint: str = Field(..., description="Endpoint to call (e.g., /search, /manage_emails)")
    http_method: str = Field(default="POST", description="HTTP method")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the call")
    action_description: str = Field(..., description="Human-readable description of what this action does")


class DialogueNextStep(BaseModel):
    """LLM's decision for the next step in dialogue."""
    is_complete: bool = Field(..., description="True if the goal has been achieved")
    reasoning: str = Field(..., description="Why this decision was made")
    next_action: Optional[DialogueAction] = Field(None, description="The next action to take (if not complete)")
    final_summary: Optional[str] = Field(None, description="Summary for user (if complete)")


class DialogueTask(BaseModel):
    """
    A task that requires multi-turn conversation with an agent.
    Used when the Orchestrator needs to see intermediate results before deciding next steps.
    """
    goal: str = Field(..., description="The user's ultimate goal")
    agent_id: str = Field(..., description="Which agent to converse with")
    agent_base_url: str = Field(..., description="Base URL of the agent")
    available_endpoints: List[Dict[str, Any]] = Field(default_factory=list, description="Endpoints available on this agent")
    initial_action: DialogueAction = Field(..., description="First action to take")
    max_turns: int = Field(default=5, description="Maximum dialogue turns")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for LLM analysis")



# --- Bidirectional Dialogue Models ---

class AgentResponseStatus(str, Enum):
    """Status of an agent's execution response."""
    COMPLETE = "complete"
    ERROR = "error"
    NEEDS_INPUT = "needs_input"
    PARTIAL = "partial"

# --- Canvas Display Schemas ---

class CanvasDisplay(BaseModel):
    """
    Standardized schema for agents to send visual content to the canvas.
    
    Two modes:
    1. Structured data (preferred): Send canvas_data, frontend templates it
    2. Custom HTML (when needed): Send canvas_content with raw HTML
    """
    canvas_type: Literal['email_preview', 'spreadsheet', 'document', 'pdf', 'image', 'json', 'html', 'markdown'] = Field(
        ...,
        description="Type of content being displayed in canvas"
    )
    canvas_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured data for the canvas (frontend will template it) - PREFERRED"
    )
    canvas_content: Optional[str] = Field(
        None,
        description="Raw HTML/markdown content (use when custom rendering needed)"
    )
    canvas_title: Optional[str] = Field(
        None,
        description="Optional title for the canvas display"
    )
    requires_confirmation: bool = Field(
        False,
        description="If True, user must confirm before agent proceeds (for critical actions)"
    )
    confirmation_message: Optional[str] = Field(
        None,
        description="Message to show when confirmation is required"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "canvas_type": "email_preview",
                    "canvas_data": {
                        "to": ["user@example.com"],
                        "cc": [],
                        "bcc": [],
                        "subject": "Meeting Reminder",
                        "body": "Don't forget our meeting tomorrow",
                        "is_html": False,
                        "attachments": []
                    },
                    "requires_confirmation": True,
                    "confirmation_message": "Review and confirm to send this email"
                },
                {
                    "canvas_type": "spreadsheet",
                    "canvas_data": {
                        "headers": ["Name", "Age", "City"],
                        "rows": [
                            ["Alice", 30, "NYC"],
                            ["Bob", 25, "LA"]
                        ],
                        "filename": "data.csv"
                    }
                }
            ]
        }

    class Config:
        json_schema_extra = {
            "example": {
                "canvas_type": "email_preview",
                "canvas_content": "<html><body><h1>Email Preview</h1>...</body></html>",
                "canvas_title": "Email to be sent",
                "requires_confirmation": True,
                "confirmation_message": "Please review the email before sending"
            }
        }


class StandardAgentResponse(BaseModel):
    """
    New Standardized Schema for Agent Responses (v2).
    Explicitly separates context for the Orchestrator (LLM) from data for the Client.
    """
    status: Literal["success", "error", "partial"] = Field(..., description="High-level status")
    summary: str = Field(..., description="Concise natural language summary for the Orchestrator LLM (e.g. 'Found 50 rows')")
    data: Optional[Dict[str, Any]] = Field(None, description="Heavy data payload for Client/Canvas (NOT for Orchestrator LLM)")
    canvas_display: Optional[CanvasDisplay] = Field(None, description="Visual configuration for the frontend")
    error_message: Optional[str] = Field(None, description="Details if status is error")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Tracing, timing, or debug info")


class AgentResponse(BaseModel):
    """
    Standardized response from an agent to the orchestrator.
    Supports pausing for input, streaming partial results, and final completion.
    """
    status: AgentResponseStatus = Field(..., description="Execution status")
    result: Optional[Any] = Field(None, description="Final result (if complete)")
    # Optional - populate if migrating to v2
    standard_response: Optional[StandardAgentResponse] = Field(None, description="v2 response structure")
    
    error: Optional[str] = Field(None, description="Error message (if error)")
    
    # For needs_input status
    question: Optional[str] = Field(None, description="Clarifying question for the orchestrator/user")
    question_type: Optional[Literal["choice", "text", "confirmation"]] = Field(None, description="Type of input needed")
    options: Optional[List[str]] = Field(None, description="Valid options for choice questions")
    context: Optional[Dict[str, Any]] = Field(None, description="Context to help answer the question")
    
    # For partial status
    partial_result: Optional[Any] = Field(None, description="Intermediate result")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress indicator (0.0-1.0)")

class OrchestratorMessage(BaseModel):
    """Message from Orchestrator to Agent during a dialogue."""
    type: Literal["execute", "continue", "cancel", "context_update"] = Field("execute", description="Message type")
    
    # For execute
    action: Optional[str] = Field(None, description="Action/Endpoint to execute (optional if prompt is provided)")
    prompt: Optional[str] = Field(None, description="Natural language prompt for complex multi-step tasks (agent handles decomposition)")
    payload: Optional[Dict[str, Any]] = Field(None, description="Arguments for the action")
    
    # For continue
    answer: Optional[str] = Field(None, description="Answer to agent's question")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Updated context")

class DialogueContext(BaseModel):
    """Tracks the state of a specific agent conversation."""
    task_id: str
    agent_id: str
    status: Literal["active", "paused", "completed"]
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Message history")
    current_question: Optional[AgentResponse] = Field(None, description="Pending question if paused")


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
    canvas_data: Optional[Dict[str, Any]] = None # Added V2 structured data support
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
    """
    Represents a file in the orchestrator system.
    
    This schema is compatible with both the legacy file system and the new
    unified content management system.
    """
    file_name: str
    file_path: str
    file_type: Literal['image', 'document', 'spreadsheet', 'code', 'data', 'archive', 'other'] = 'document'
    
    # Optional fields for unified content system
    file_id: Optional[str] = Field(None, description="Unified content ID")
    content_id: Optional[str] = Field(None, description="Alias for file_id (unified content system)")
    mime_type: Optional[str] = Field(None, description="MIME type of the file")
    size: Optional[int] = Field(None, description="File size in bytes")
    source: Optional[str] = Field(None, description="Source of the file (user_upload, agent_output, etc.)")
    thread_id: Optional[str] = Field(None, description="Associated conversation thread")
    
    # Legacy fields for backward compatibility
    base64_content: Optional[str] = None
    vector_store_path: Optional[str] = None
    
    @classmethod
    def from_unified_metadata(cls, metadata) -> 'FileObject':
        """Create FileObject from UnifiedContentMetadata"""
        return cls(
            file_name=metadata.name,
            file_path=metadata.storage_path,
            file_type=metadata.content_type.value if hasattr(metadata.content_type, 'value') else str(metadata.content_type),
            file_id=metadata.id,
            content_id=metadata.id,
            mime_type=metadata.mime_type,
            size=metadata.size_bytes,
            source=metadata.source.value if hasattr(metadata.source, 'value') else str(metadata.source),
            thread_id=metadata.thread_id
        )


class EnrichedFileObject(BaseModel):
    """File with semantic content understanding for orchestrator context."""
    # Basic metadata
    file_name: str
    file_path: str
    file_type: Literal['image', 'document', 'spreadsheet', 'code', 'data', 'archive', 'other'] = 'document'
    
    # Unified content system fields
    file_id: Optional[str] = Field(None, description="Unified content ID")
    content_id: Optional[str] = Field(None, description="Alias for file_id")
    mime_type: Optional[str] = Field(None, description="MIME type")
    size: Optional[int] = Field(None, description="File size in bytes")
    source: Optional[str] = Field(None, description="Source of the file")
    
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
    canvas_confirmation_action: Optional[Literal["confirm", "cancel", "modify"]] = Field(None, description="Action if user is responding to a confirmation request")
    canvas_confirmation_task: Optional[str] = Field(None, description="Task name being confirmed/cancelled")

class PlanValidationResult(BaseModel):
    """Schema for the advanced validation node's output."""
    status: Literal["ready", "replan_needed", "user_input_required"] = Field(..., description="The status of the plan validation.")
    reasoning: Optional[str] = Field(None, description="Required explanation if status is 'replan_needed' or 'user_input_required'.")
    question: Optional[str] = Field(None, description="The direct question for the user if input is absolutely required.")

class AgentResponseEvaluationEnhanced(BaseModel):
    """Enhanced schema for evaluating agent responses with reactive routing support."""
    status: Literal["complete", "partial_success", "failed", "user_input_required", "anomaly_detected"] = Field(..., description="The status of the agent response evaluation.")
    reasoning: str = Field(..., description="Explanation of the evaluation decision.")
    feedback_for_replanning: Optional[str] = Field(None, description="Specific feedback for the planner if status is 'failed'.")
    question: Optional[str] = Field(None, description="Question to ask the user if status is 'user_input_required' or 'anomaly_detected'.")

class FinalResponse(BaseModel):
    """Schema for unified final response generation (text + canvas)."""
    response_text: str = Field(..., description="The text response for the user")
    canvas_required: bool = Field(..., description="Whether canvas visualization is needed")
    canvas_type: Optional[Literal["html", "markdown"]] = Field(None, description="Type of canvas content")
    canvas_content: Optional[str] = Field(None, description="The actual canvas content (HTML or Markdown)")

# --- Plan Modification Models (NEW) ---

class PlanModification(BaseModel):
    """Tracks a single modification made to the plan."""
    modification_type: Literal['add_task', 'update_task', 'remove_task', 'reorder_tasks'] = Field(..., description="Type of modification")
    affected_tasks: List[str] = Field(..., description="Task names affected by this modification")
    reasoning: str = Field(..., description="Why this modification was made")
    timestamp: str = Field(..., description="ISO timestamp of modification")
    user_instruction: str = Field(..., description="The user's actual request that triggered this")

class UserUpdateAnalysis(BaseModel):
    """Result of analyzing user's update request in context of existing plan."""
    request_type: Literal['add_task', 'update_task', 'update_specific_task', 'clarification_only', 'execution_question'] = Field(...)
    confidence: float = Field(..., description="Confidence in this analysis (0.0-1.0)")
    reasoning: str = Field(...)
    affected_tasks: List[str] = Field(default_factory=list, description="Existing tasks affected")
    new_task_description: Optional[str] = Field(None, description="If add_task, the new task description")
    specific_task_to_update: Optional[str] = Field(None, description="If update_specific_task, which task")
    update_reason: Optional[str] = Field(None, description="Why the task should be updated")
    requires_new_agent: bool = Field(default=False)
    suggested_agent: Optional[str] = Field(None, description="Suggested agent for new task")
    next_action: Literal['add_node', 'update_node', 'modify_node', 'ask_clarification', 'proceed_with_execution'] = Field(...)

class PlanState(BaseModel):
    """Complete state of the plan including history of modifications."""
    version: int = Field(default=1, description="Plan version number")
    original_plan: List[List[PlannedTask]] = Field(default_factory=list, description="Original plan before any modifications")
    current_plan: List[List[PlannedTask]] = Field(default_factory=list, description="Current plan state after modifications")
    modifications: List[PlanModification] = Field(default_factory=list, description="All modifications made to the plan")
    is_editable: bool = Field(default=True, description="Whether plan can still be edited")
    last_modification: Optional[str] = Field(None, description="ISO timestamp of last modification")




class AgentResponseWithCanvas(BaseModel):
    """
    Standard response format for agents that want to display content in canvas.
    Agents should include this in their response when they want to show visual content.
    """
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    canvas_display: Optional[CanvasDisplay] = None
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": {"email_id": "123"},
                "canvas_display": {
                    "canvas_type": "email_preview",
                    "canvas_content": "<html>...</html>",
                    "requires_confirmation": True
                },
                "message": "Email prepared and ready to send"
            }
        }

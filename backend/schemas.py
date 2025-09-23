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
    endpoint: HttpUrl
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

    class Config:
        from_attributes = True

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

class ParsedRequest(BaseModel):
    """The output of the initial prompt parsing node."""
    tasks: List[Task]
    user_expectations: Optional[Dict[str, float]] = None

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
    endpoint: HttpUrl
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

class ProcessRequest(BaseModel):
    """The schema for the main `/api/chat` endpoint request."""
    prompt: str
    thread_id: Optional[str] = None  # To continue an existing conversation
    user_response: Optional[str] = None # To provide an answer to a question

class ProcessResponse(BaseModel):
    """The schema for the main `/api/chat` endpoint response."""
    message: str
    thread_id: str
    task_agent_pairs: List[TaskAgentPair]
    final_response: Optional[str] = None
    pending_user_input: bool = False
    question_for_user: Optional[str] = None
    
class PlanResponse(BaseModel):
    """The schema for the GET /api/plan/{thread_id} endpoint response."""
    thread_id: str
    content: str

# --- Utility Models for Graph Nodes ---

class SelectedEndpoint(BaseModel):
    """Used to validate the LLM's choice of endpoint for a task."""
    endpoint: HttpUrl
    http_method: str

class PriorityMapper(BaseModel):
    """Used to map a task to a priority level."""
    task_name: str
    priority: str

class PriorityMappingResponse(BaseModel):
    """The expected response when asking the LLM to prioritize tasks."""
    mappings: List[PriorityMapper]
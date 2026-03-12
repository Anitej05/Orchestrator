# In Orbimesh Backend/models.py

from sqlalchemy import Column, String, Float, Text, Enum as SAEnum, ForeignKey, Integer, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
# from sqlalchemy.dialects.postgresql import ARRAY, JSON
from database import Base, engine
try:
    from pgvector.sqlalchemy import Vector
    # Check if we are using SQLite - if so, mock Vector even if pgvector is installed
    if "sqlite" in engine.url.drivername:
        raise ImportError("Force mock for SQLite")
except ImportError:
    # Fallback for systems without pgvector installed or using SQLite
    from sqlalchemy.types import UserDefinedType
    class Vector(UserDefinedType):
        def __init__(self, dimensions):
            self.dimensions = dimensions
        def get_col_spec(self, **kw):
            return "TEXT" # SQLite doesn't have a vector type, use TEXT as fallback
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional, Literal
import enum
import uuid
from datetime import datetime, timezone

class StatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    deprecated = "deprecated"

class AgentType(str, enum.Enum):
    HTTP_REST = "http_rest"   # Legacy OpenAPI/REST agents
    MCP_HTTP = "mcp_http"     # Modern MCP Servers
    TOOL = "tool"             # Direct Python function calls via LangChain @tool

class AuthType(str, enum.Enum):
    NONE = "none"
    API_KEY = "api_key"       # Bearer token or custom header
    OAUTH2 = "oauth2"         # Access/Refresh tokens

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    capabilities = Column(JSON, nullable=True)  # Now nullable - endpoints are the primary source of truth
    price_per_call_usd = Column(Float, nullable=False, default=0.0)
    status = Column(SAEnum(StatusEnum), nullable=False, default=StatusEnum.active, index=True)
    public_key_pem = Column(Text, nullable=True)  # Optional for MCP agents
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))  # Track when the agent was created
    
    # MCP Support
    agent_type = Column(String, default=AgentType.HTTP_REST.value)
    # Connection Config
    # REST: { "base_url": "https://api.weather.com" }
    # MCP:  { "url": "https://mcp.supabase.com/mcp" }
    connection_config = Column(JSON, nullable=True)
    
    # Credential Management
    requires_credentials = Column(Boolean, default=False)  # Does this agent need credentials?
    credential_fields = Column(JSON, nullable=True)  # Define what credentials are needed
    # Format: [{"name": "api_key", "label": "API Key", "type": "password", "required": true, "description": "..."}]

    capability_vectors = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")

class AgentCapability(Base):
    __tablename__ = "agent_capabilities"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    capability_text = Column(String, nullable=False)
    embedding = Column(Vector(768)) # Storing one vector per row

    agent = relationship("Agent", back_populates="capability_vectors")

class UserThread(Base):
    __tablename__ = "user_threads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    thread_id = Column(String, nullable=False, unique=True, index=True)
    title = Column(String, nullable=True)  # Title for the conversation
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    blueprint = Column(JSON, nullable=False)  # Full workflow structure
    plan_graph = Column(JSON, nullable=True)  # Execution graph/visualization
    version = Column(Integer, default=1)
    status = Column(String, default='active')  # active, archived
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String, nullable=False, unique=True, index=True)
    workflow_id = Column(String, ForeignKey("workflows.workflow_id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    status = Column(String, default='running')  # queued, running, completed, failed
    inputs = Column(JSON)
    outputs = Column(JSON)
    error = Column(Text)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    completed_at = Column(DateTime)

class WorkflowSchedule(Base):
    __tablename__ = "workflow_schedules"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(String, nullable=False, unique=True, index=True)
    workflow_id = Column(String, ForeignKey("workflows.workflow_id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    cron_expression = Column(String, nullable=False)
    input_template = Column(JSON)
    is_active = Column(Boolean, default=True)
    conversation_thread_id = Column(String, nullable=True)  # Thread ID for scheduled execution results
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class WorkflowWebhook(Base):
    __tablename__ = "workflow_webhooks"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(String, nullable=False, unique=True, index=True)
    workflow_id = Column(String, ForeignKey("workflows.workflow_id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    webhook_token = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class Credential(Base):
    """
    Generic, scope-based credential storage.

    Unlike AgentCredential (which is FK'd to agents.id), this table supports
    any scope: agents, tools, system-level keys, third-party integrations, etc.

    Scope examples:
      scope="agent",  scope_id="gmail_agent"      → Gmail agent's Composio key
      scope="tool",   scope_id="web_search"        → SerpAPI key for web search tool
      scope="system", scope_id="global"            → System-wide shared credentials
    """
    __tablename__ = "credentials"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)  # "system" for shared creds
    scope = Column(String(50), nullable=False, index=True)     # "agent", "tool", "system"
    scope_id = Column(String(255), nullable=False, index=True) # agent_id, tool_name, "global"

    # All credentials stored as encrypted JSON: {"api_key": "enc...", "secret": "enc..."}
    encrypted_credentials = Column(JSON, nullable=True, default=dict)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# INTEGRATIONS / COMPOSIO OAUTH TABLES
# ============================================================================

class UserConnection(Base):
    """
    Stores OAuth connection metadata for external services (Composio).
    Links users to external app connections (Gmail, Zoho Books, etc.)
    
    Canonical schema aligned with MASTER_IMPLEMENTATION_PLAN_v2:
    - internal_user_id  → local system user ID
    - composio_entity_id → Composio's per-user entity ID
    - app_name          → normalised app name (e.g. GMAIL)
    - connection_id     → AES-256-GCM ciphertext; NEVER store plaintext
    - status            → ACTIVE | PENDING | EXPIRED | REVOKED
    """
    __tablename__ = "user_connections"
    
    id = Column(String, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)           # legacy alias kept
    internal_user_id = Column(String(255), nullable=True, index=True)   # canonical field
    composio_entity_id = Column(String(255), nullable=True, index=True) # Composio entity
    app_slug = Column(String(100), nullable=False, index=True)          # e.g. 'gmail'
    app_name = Column(String(100), nullable=True)                       # normalised e.g. 'GMAIL'
    connection_id = Column(Text, nullable=True)  # Encrypted ciphertext — decrypt only at execution boundary
    status = Column(String(50), default='PENDING')  # ACTIVE | PENDING | EXPIRED | REVOKED
    
    # Connection metadata
    app_metadata = Column(JSON, nullable=True)  # App-specific data
    auth_timestamp = Column(DateTime, nullable=True)
    last_verified = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class IntegrationsSession(Base):
    """
    Per-user, per-app session state for the Integrations Agent.
    Persists context (e.g. folder_id, sheet_id) across conversations so the
    agent can offer continuity: "Want to pick up where you left off?"
    """
    __tablename__ = "integrations_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    app_slug = Column(String(100), nullable=False)          # e.g. 'gmail', 'googlesheets'
    session_id = Column(String(255), nullable=False)        # Composio session ID
    last_context = Column(JSON, nullable=True)              # Preserved context dict
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    last_used = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ComposioEntity(Base):
    """
    Bidirectional mapping between local user IDs and Composio entity IDs.
    Recommended by architecture: maintain the mapping in our own DB so we
    can resolve it without a Composio round-trip.
    """
    __tablename__ = "composio_entities"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    internal_user_id = Column(String(255), unique=True, nullable=False, index=True)
    composio_entity_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


# ============================================================================
# ENHANCED ANALYTICS & TAGGING TABLES
# ============================================================================

class ConversationPlan(Base):
    """
    Tracks plan iterations for a conversation.
    Allows storing multiple plan attempts per conversation.
    """
    __tablename__ = "conversation_plans"
    
    id = Column(Integer, primary_key=True)
    plan_id = Column(String(255), nullable=False, unique=True, index=True)
    thread_id = Column(String(255), ForeignKey("user_threads.thread_id"), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    plan_version = Column(Integer, default=1)
    
    # Plan content
    task_agent_pairs = Column(JSON, nullable=False)
    task_plan = Column(JSON, nullable=False)
    plan_graph = Column(JSON, nullable=True)  # Execution visualization graph
    
    # Execution info
    status = Column(String(50), default='draft')  # 'draft', 'executing', 'completed', 'failed'
    result = Column(JSON, nullable=True)  # Execution results
    execution_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ConversationTag(Base):
    """
    User-defined or system tags for organizing conversations.
    """
    __tablename__ = "conversation_tags"
    
    id = Column(Integer, primary_key=True)
    tag_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    tag_name = Column(String(100), nullable=False)
    tag_color = Column(String(7), default='#808080')  # Hex color
    tag_description = Column(Text, nullable=True)
    is_system = Column(Boolean, default=False)  # System vs user-created
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class ConversationTagAssignment(Base):
    """
    Many-to-many junction table for conversations and tags.
    """
    __tablename__ = "conversation_tag_assignments"
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(255), ForeignKey("user_threads.thread_id"), nullable=False, index=True)
    tag_id = Column(String(255), ForeignKey("conversation_tags.tag_id"), nullable=False, index=True)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class AgentUsageAnalytics(Base):
    """
    Tracks agent usage patterns per user.
    """
    __tablename__ = "agent_usage_analytics"
    
    id = Column(Integer, primary_key=True)
    analytics_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    agent_id = Column(String(255), ForeignKey("agents.id"), nullable=False)
    
    # Execution metrics
    execution_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_execution_time_ms = Column(Float, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class UserActivitySummary(Base):
    """
    Daily/periodic activity summary for each user.
    Used for analytics dashboards and reporting.
    """
    __tablename__ = "user_activity_summary"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    activity_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    
    # Daily counts
    total_conversations_started = Column(Integer, default=0)
    total_workflows_executed = Column(Integer, default=0)
    total_plans_created = Column(Integer, default=0)
    successful_executions = Column(Integer, default=0)
    failed_executions = Column(Integer, default=0)
    total_execution_time_ms = Column(Integer, default=0)
    agents_used = Column(Integer, default=0)
    api_calls_made = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class WorkflowExecutionAnalytics(Base):
    """
    Detailed analytics for each workflow execution.
    """
    __tablename__ = "workflow_execution_analytics"
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(String(255), ForeignKey("workflow_executions.execution_id"), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    workflow_id = Column(String(255), ForeignKey("workflows.workflow_id"), nullable=False, index=True)
    
    # Execution metrics
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    failed_steps = Column(Integer, default=0)
    total_duration_ms = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)
    error_type = Column(String(100), nullable=True)
    success_rate = Column(Float, default=0)  # Percentage (0-100)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

class LLMTelemetryRecord(Base):
    """
    Detailed tracking of every LLM call for cost, token usage, and analytics.
    """
    __tablename__ = "llm_telemetry_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    thread_id = Column(String(255), nullable=True, index=True)
    agent_name = Column(String(100), nullable=True, index=True)
    operation_type = Column(String(50), nullable=False) # e.g., 'agent', 'orchestrator', 'tool'
    model_name = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

class AgentExecutionRecord(Base):
    """
    Detailed tracking of agent executions per session.
    """
    __tablename__ = "agent_execution_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    thread_id = Column(String(255), nullable=True, index=True)
    agent_name = Column(String(100), nullable=False, index=True)
    duration_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

class ToolExecutionRecord(Base):
    """
    Detailed tracking of specific tool calls per session.
    """
    __tablename__ = "tool_execution_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), index=True)
    user_id = Column(String(255), nullable=True, index=True)
    thread_id = Column(String(255), nullable=True, index=True)
    tool_name = Column(String(100), nullable=False, index=True)
    duration_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)


class ArtifactEmbedding(Base):
    """
    Semantic vector index for the orchestrator's self-learning artifact memory.
    Enables pgvector cosine similarity search across task results, knowledge,
    playbooks, and error patterns — even when keywords don't overlap.
    """
    __tablename__ = "artifact_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    artifact_id = Column(String(255), nullable=False, unique=True, index=True)
    artifact_type = Column(String(50), nullable=False, index=True)  # task_result, knowledge, playbook
    summary = Column(Text, nullable=False)
    embedding = Column(Vector(768))  # all-mpnet-base-v2 outputs 768-dim vectors
    tags = Column(JSON, nullable=True)
    source_objective = Column(Text, nullable=True)
    source_agent = Column(String(255), nullable=True)
    file_path = Column(Text, nullable=True)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    last_used_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


# ============================================================================
# ORCHESTRATOR → AGENT CONTRACTS (Phase 1: Refactor)
# ============================================================================

class DecisionContract(BaseModel):
    """
    Contract from orchestrator to agent defining task intent and constraints.
    
    The orchestrator is the SOLE authority on task classification.
    The agent MUST validate and enforce this contract.
    
    Purpose: Eliminate redundant reasoning layers by making intent explicit.
    """
    task_type: Literal["transform", "summary", "compare", "merge", "qa", "create", "preview"]
    allow_write: bool = False
    allow_schema_change: bool = False
    confidence_required: float = 0.8
    source: Literal["orchestrator", "user"] = "orchestrator"
    
    class Config:
        from_attributes = True


class StandardResponseMetrics(BaseModel):
    """Standardized metrics for all agent operations"""
    rows_processed: int = 0
    columns_affected: int = 0
    execution_time_ms: float = 0.0
    llm_calls: int = 0
    
    class Config:
        from_attributes = True


class StandardResponse(BaseModel):
    """
    Unified response schema for ALL spreadsheet agent endpoints.
    
    CRITICAL RULES:
    - success=false MUST include message
    - data MUST always exist (even if empty dict)
    - artifact MUST be explicit (never implicit)
    - route and task_type MUST match endpoint called
    
    Purpose: Eliminate 13 response schema variants, enable predictable frontend rendering.
    """
    success: bool
    route: str = Field(..., description="Endpoint called (e.g., '/compare')")
    task_type: str = Field(..., description="From Decision Contract or inferred")
    data: Dict[str, Any] = Field(default_factory=dict, description="Always present, even if empty")
    preview: Optional[Dict[str, Any]] = Field(None, description="Explicit preview data for UI")
    artifact: Optional[Dict[str, str]] = Field(None, description="{id, filename, url} or None")
    metrics: StandardResponseMetrics = Field(default_factory=StandardResponseMetrics)
    confidence: float = Field(1.0, description="1.0 for deterministic, <1.0 for LLM-based")
    needs_clarification: bool = False
    message: str = Field("", description="Required when success=false or needs_clarification=true")
    
    @model_validator(mode='after')
    def validate_message_required(self):
        """Ensure message is provided when needed"""
        if not self.success and not self.message:
            raise ValueError("message is required when success=false")
        if self.needs_clarification and not self.message:
            raise ValueError("message is required when needs_clarification=true")
        return self
    
    class Config:
        from_attributes = True

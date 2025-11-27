# In Project_Agent_Directory/models.py

from sqlalchemy import Column, String, Float, Text, Enum as SAEnum, ForeignKey, Integer, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from pgvector.sqlalchemy import Vector
from database import Base
import enum
import uuid
from datetime import datetime

class StatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    deprecated = "deprecated"

class AgentType(str, enum.Enum):
    HTTP_REST = "http_rest"   # Legacy OpenAPI/REST agents
    MCP_HTTP = "mcp_http"     # Modern MCP Servers

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
    capabilities = Column(JSON, nullable=False) # Store raw text capabilities in a JSON array
    price_per_call_usd = Column(Float, nullable=False, default=0.0)
    status = Column(SAEnum(StatusEnum), nullable=False, default=StatusEnum.active, index=True)
    rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0, nullable=False)
    public_key_pem = Column(Text, nullable=True)  # Optional for MCP agents
    created_at = Column(DateTime, default=datetime.utcnow)  # Track when the agent was created
    
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
    endpoints = relationship("AgentEndpoint", back_populates="agent", cascade="all, delete-orphan", lazy="joined")
    credentials = relationship("AgentCredential", back_populates="agent", cascade="all, delete-orphan")

class AgentCapability(Base):
    __tablename__ = "agent_capabilities"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    capability_text = Column(String, nullable=False)
    embedding = Column(Vector(768)) # Storing one vector per row

    agent = relationship("Agent", back_populates="capability_vectors")

class AgentEndpoint(Base):
    __tablename__ = "agent_endpoints"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    endpoint = Column(String, nullable=False)
    http_method = Column(String, nullable=False, default="POST")
    description = Column(Text)

    agent = relationship("Agent", back_populates="endpoints")
    parameters = relationship("EndpointParameter", back_populates="endpoint", cascade="all, delete-orphan")

class EndpointParameter(Base):
    __tablename__ = "endpoint_parameters"

    id = Column(Integer, primary_key=True, index=True)
    endpoint_id = Column(Integer, ForeignKey("agent_endpoints.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    param_type = Column(String, nullable=False)
    required = Column(Boolean, default=True)
    default_value = Column(String)

    endpoint = relationship("AgentEndpoint", back_populates="parameters")

class UserThread(Base):
    __tablename__ = "user_threads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    thread_id = Column(String, nullable=False, unique=True, index=True)
    title = Column(String, nullable=True)  # Title for the conversation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    blueprint = Column(JSON, nullable=False)  # Full workflow structure
    version = Column(Integer, default=1)
    status = Column(String, default='active')  # active, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
    started_at = Column(DateTime, default=datetime.utcnow)
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
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class WorkflowWebhook(Base):
    __tablename__ = "workflow_webhooks"

    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(String, nullable=False, unique=True, index=True)
    workflow_id = Column(String, ForeignKey("workflows.workflow_id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    webhook_token = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AgentCredential(Base):
    """
    Stores user-specific authentication for an agent.
    Links a User + Agent + Encrypted Keys.
    Supports multiple credential fields per agent.
    """
    __tablename__ = "agent_credentials"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)  # Clerk User ID
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    
    # New: Store all credentials as encrypted JSON
    # Format: {"api_key": "encrypted_value", "connection_id": "encrypted_value", ...}
    encrypted_credentials = Column(JSON, nullable=True, default=dict)
    
    # Legacy fields (kept for backward compatibility)
    auth_type = Column(String, default=AuthType.NONE.value)
    encrypted_access_token = Column(Text, nullable=True)
    encrypted_refresh_token = Column(Text, nullable=True)
    auth_header_name = Column(String, default="Authorization")
    token_expires_at = Column(DateTime, nullable=True)
    
    # Metadata
    is_active = Column(Boolean, default=True)  # Can be disabled without deleting
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    agent = relationship("Agent", back_populates="credentials")
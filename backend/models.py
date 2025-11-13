# In Project_Agent_Directory/models.py

from sqlalchemy import Column, String, Float, Text, Enum as SAEnum, ForeignKey, Integer, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from pgvector.sqlalchemy import Vector
from database import Base
import enum
from datetime import datetime

class StatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    deprecated = "deprecated"

class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, index=True)
    owner_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    capabilities = Column(JSON, nullable=False) # Store raw text capabilities in a JSON array
    price_per_call_usd = Column(Float, nullable=False)
    status = Column(SAEnum(StatusEnum), nullable=False, default=StatusEnum.active, index=True)
    rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0, nullable=False)
    public_key_pem = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)  # Track when the agent was created

    capability_vectors = relationship("AgentCapability", back_populates="agent", cascade="all, delete-orphan")
    endpoints = relationship("AgentEndpoint", back_populates="agent", cascade="all, delete-orphan", lazy="joined")

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
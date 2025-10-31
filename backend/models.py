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
    created_at = Column(DateTime, default=datetime.utcnow)
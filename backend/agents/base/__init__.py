"""
Base Agent Package
Core infrastructure for intelligent agents.
"""

# Core types
from .types import (
    AgentStatus,
    AgentRequest,
    AgentResponse,
    AgentConfig,
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    CapabilityResult,
    RecoveryPlan,
    ParameterSchema,
    CapabilityType,
)

# Services
from .services import AgentServices, get_services

# Capabilities
from .capability import (
    Capability,
    SimpleCapability,
    CompoundCapability,
    CapabilityRegistry,
    capability,
)

# Base Agent
from .agent import BaseAgent

# Server
from .server import AgentServer, create_agent_server

__all__ = [
    # Types
    "AgentStatus",
    "AgentRequest",
    "AgentResponse",
    "AgentConfig",
    "ExecutionContext",
    "ExecutionPlan",
    "ExecutionStep",
    "CapabilityResult",
    "RecoveryPlan",
    "ParameterSchema",
    "CapabilityType",
    
    # Services
    "AgentServices",
    "get_services",
    
    # Capabilities
    "Capability",
    "SimpleCapability",
    "CompoundCapability",
    "CapabilityRegistry",
    "capability",
    
    # Base Agent
    "BaseAgent",
    
    # Server
    "AgentServer",
    "create_agent_server",
]

__version__ = "1.0.0"

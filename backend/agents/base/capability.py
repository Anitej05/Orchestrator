"""
Capability System
Defines capabilities and the decorator pattern for registration.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field

from .types import CapabilityType, ParameterSchema, ExecutionContext, CapabilityResult
from .services import AgentServices

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """
    A capability that an agent can perform.
    """

    name: str
    description: str
    capability_type: CapabilityType = CapabilityType.SIMPLE
    parameters: List[ParameterSchema] = field(default_factory=list)
    handler: Optional[Callable] = None

    # For compound capabilities
    internal_tools: Optional[Dict[str, "Capability"]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.capability_type.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
        }

    async def execute(
        self, agent: Any, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """Execute this capability."""
        if self.handler is None:
            raise ValueError(f"Capability {self.name} has no handler")

        try:
            # If the handler is a bound method (stored via discover_from_agent,
            # which uses getattr on the live agent instance), self is already
            # baked in — calling handler(agent, params, context) would pass
            # agent as an extra first positional arg, causing a TypeError.
            # Unbound / standalone handlers still need the agent passed explicitly.
            if hasattr(self.handler, "__self__"):
                result = await self.handler(params, context)
            else:
                result = await self.handler(agent, params, context)

            # Normalize result
            if isinstance(result, CapabilityResult):
                return result
            elif isinstance(result, dict) and "success" in result:
                # Map dict keys to CapabilityResult fields.
                # Extra keys (message, canvas_display, file_id, etc.) are
                # bundled into 'metadata' so CapabilityResult(**) doesn't blow up.
                known_keys = {"success", "data", "error", "metadata"}
                cap_kwargs: Dict[str, Any] = {}
                extra: Dict[str, Any] = {}
                for k, v in result.items():
                    if k in known_keys:
                        cap_kwargs[k] = v
                    elif k == "result" and "data" not in result:
                        # "result" is a common alias for "data" in capability returns
                        cap_kwargs["data"] = v
                    else:
                        extra[k] = v
                # Merge extra into metadata
                if extra:
                    existing_meta = cap_kwargs.get("metadata") or {}
                    if isinstance(existing_meta, dict):
                        existing_meta.update(extra)
                    else:
                        existing_meta = extra
                    cap_kwargs["metadata"] = existing_meta
                return CapabilityResult(**cap_kwargs)
            else:
                return CapabilityResult.ok(data=result)

        except Exception as e:
            logger.error(f"Capability {self.name} failed: {e}")
            return CapabilityResult.fail(error=str(e))


class SimpleCapability(Capability):
    """A simple, single-operation capability."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: List[ParameterSchema] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            capability_type=CapabilityType.SIMPLE,
            parameters=parameters or [],
            handler=handler,
        )


class CompoundCapability(Capability):
    """
    A compound capability with internal tools.
    LLM decides which internal tools to use.
    """

    def __init__(
        self, name: str, description: str, internal_tools: Dict[str, Capability] = None
    ):
        super().__init__(
            name=name,
            description=description,
            capability_type=CapabilityType.COMPOUND,
            internal_tools=internal_tools or {},
        )

    def add_tool(self, name: str, capability: Capability):
        """Add an internal tool."""
        self.internal_tools[name] = capability

    async def execute(
        self, agent: Any, params: Dict[str, Any], context: ExecutionContext
    ) -> CapabilityResult:
        """
        Execute compound capability.
        LLM decides which internal tools to use.
        """
        from langchain_core.messages import HumanMessage

        # LLM plans internal tool sequence
        tools_description = "\n".join(
            [
                f"- {name}: {cap.description}"
                for name, cap in self.internal_tools.items()
            ]
        )

        prompt = f"""You are orchestrating a compound capability: {self.name}

Task: {params.get("task", "Perform the capability")}

Available internal tools:
{tools_description}

Plan the sequence of internal tool calls needed to complete this task.
For each step, specify:
1. Which tool to use
2. Parameters for that tool
3. Expected outcome

Be efficient - use only the tools necessary."""

        # Get LLM to plan
        response = await agent.services.inference.generate_structured(
            messages=[HumanMessage(content=prompt)],
            schema=Dict,  # Should define a proper schema
        )

        # Execute planned steps
        results = []
        for step in response.get("steps", []):
            tool_name = step.get("tool")
            tool_params = step.get("params", {})

            if tool_name in self.internal_tools:
                tool = self.internal_tools[tool_name]
                result = await tool.execute(agent, tool_params, context)
                results.append(result)

                # Store result in context for next tools
                context.set(f"tool_{tool_name}_result", result)
            else:
                return CapabilityResult.fail(
                    error=f"Unknown internal tool: {tool_name}"
                )

        # Aggregate results
        return CapabilityResult.ok(
            data={"results": results}, metadata={"tools_used": len(results)}
        )


def capability(
    name: str,
    description: str,
    parameters: List[ParameterSchema] = None,
    mode: str = "simple",
):
    """
    Decorator to register a method as a capability.

    Usage:
        @capability(name="read_csv", description="Load CSV file")
        async def read_csv(self, params, context):
            # Implementation
            pass
    """

    def decorator(func):
        func._is_capability = True
        func._capability_name = name
        func._capability_description = description
        func._capability_parameters = parameters or []
        func._capability_mode = mode
        return func

    return decorator


class CapabilityRegistry:
    """Registry for managing agent capabilities."""

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability):
        """Register a capability."""
        self._capabilities[capability.name] = capability
        logger.debug(f"Registered capability: {capability.name}")

    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def list_all(self) -> List[Capability]:
        """List all registered capabilities."""
        return list(self._capabilities.values())

    def to_llm_context(self) -> str:
        """Generate context string for LLM."""
        lines = ["Available Capabilities:"]
        for cap in self._capabilities.values():
            lines.append(f"\n{cap.name}:")
            lines.append(f"  Description: {cap.description}")
            lines.append(f"  Type: {cap.capability_type.value}")
            if cap.parameters:
                lines.append("  Parameters:")
                for param in cap.parameters:
                    req = "(required)" if param.required else "(optional)"
                    lines.append(f"    - {param.name}: {param.type} {req}")
        return "\n".join(lines)

    def discover_from_agent(self, agent_instance: Any):
        """
        Auto-discover capabilities from an agent instance.
        Finds methods decorated with @capability.
        """
        for attr_name in dir(agent_instance):
            attr = getattr(agent_instance, attr_name)
            if callable(attr) and hasattr(attr, "_is_capability"):
                # Create capability from decorated method
                cap = Capability(
                    name=attr._capability_name,
                    description=attr._capability_description,
                    handler=attr,
                    parameters=attr._capability_parameters,
                )
                self.register(cap)
                logger.debug(f"Discovered capability from agent: {cap.name}")

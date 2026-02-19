"""
Base Agent
Abstract base class for all intelligent agents.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from .types import (
    AgentStatus,
    AgentRequest,
    AgentResponse,
    ExecutionContext,
    ExecutionPlan,
    ExecutionStep,
    RecoveryPlan,
    CapabilityResult,
)
from .services import AgentServices
from .capability import Capability, CapabilityRegistry, capability

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    max_retries: int = 3
    use_llm_recovery: bool = True
    llm_temperature: float = 0.0  # Deterministic for planning
    enable_telemetry: bool = True
    request_timeout: float = 120.0


class BaseAgent(ABC):
    """
    Abstract base class for all intelligent agents.

    Provides:
    - Service injection
    - Capability management
    - LLM-driven planning
    - Error recovery
    - Lifecycle management
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        services: Optional[AgentServices] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.services = services or AgentServices.create_default()
        self.config = config or AgentConfig()

        # Capabilities
        self.capability_registry = CapabilityRegistry()

        # Register capabilities from subclass
        self.register_capabilities()

        # Auto-discover capabilities from decorated methods
        self._discover_capabilities()

        # State
        self.status = AgentStatus.INITIALIZING
        self._initialized = False
        self._spawn_time: Optional[float] = None
        self._last_used: Optional[float] = None

        # Execution tracking
        self.execution_history: List[Dict] = []

        logger.info(f"BaseAgent initialized: {agent_name} ({agent_id})")

    def register_capabilities(self):
        """Override in subclass to register agent capabilities."""
        pass

    def _discover_capabilities(self):
        """Auto-discover capabilities from decorated methods."""
        self.capability_registry.discover_from_agent(self)
        logger.debug(
            f"Discovered {len(self.capability_registry.list_all())} capabilities"
        )

    async def initialize(self):
        """
        Initialize agent resources.
        Lazy loading - only initialize when first needed.
        """
        if self._initialized:
            return

        logger.info(f"Initializing agent: {self.agent_name}")

        # Initialize essential services
        self.services.initialize_essential()

        # Agent-specific initialization (override in subclass)
        await self._initialize_resources()

        self._initialized = True
        self._spawn_time = time.time()
        self.status = AgentStatus.READY

        logger.info(f"Agent {self.agent_name} ready")

    @abstractmethod
    async def _initialize_resources(self):
        """Override in subclass for agent-specific initialization."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status."""
        return {
            "status": self.status.value,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "initialized": self._initialized,
            "capabilities_count": len(self.capability_registry.list_all()),
            "uptime_seconds": time.time() - self._spawn_time if self._spawn_time else 0,
        }

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Main execution entry point.
        LLM-driven planning and execution.
        """
        start_time = time.time()

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        self.status = AgentStatus.BUSY
        self._last_used = time.time()

        try:
            # Execute with recovery
            result = await self._execute_with_recovery(request)

            # Log telemetry
            if self.config.enable_telemetry:
                duration_ms = (time.time() - start_time) * 1000
                self.services.telemetry.log_agent_call(
                    agent_name=self.agent_name,
                    success=result.status != "error",
                    duration_ms=duration_ms,
                )

            self.status = AgentStatus.READY
            return result

        except Exception as e:
            logger.error(f"Unexpected error in agent execution: {e}")
            self.status = AgentStatus.ERROR

            if self.config.enable_telemetry:
                self.services.telemetry.log_error(
                    category="agent",
                    error_message=str(e),
                    context={"agent_id": self.agent_id, "request": request.prompt},
                )

            return AgentResponse.error(
                message=f"Agent execution failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_with_recovery(self, request: AgentRequest) -> AgentResponse:
        """
        Execute with intelligent error recovery.
        """
        attempt = 0
        last_error = None

        while attempt < self.config.max_retries:
            try:
                # Try normal execution
                return await self._execute_single_attempt(request, attempt)

            except Exception as e:
                attempt += 1
                last_error = e
                logger.warning(f"Execution attempt {attempt} failed: {e}")

                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    break

                if not self.config.use_llm_recovery:
                    # Simple retry without LLM
                    continue

                # LLM analyzes error and plans recovery
                recovery_plan = await self._llm_plan_recovery(
                    error=e, request=request, attempt=attempt
                )

                if recovery_plan.should_retry:
                    # Apply fixes and retry
                    request = await self._apply_recovery_fixes(request, recovery_plan)
                    logger.info(f"Retrying with fixes: {recovery_plan.reasoning}")
                    continue

                elif recovery_plan.should_alternate:
                    # Try alternative approach
                    logger.info(
                        f"Trying alternative approach: {recovery_plan.reasoning}"
                    )
                    return await self._execute_alternative_approach(
                        request, recovery_plan
                    )

                elif recovery_plan.should_escalate:
                    # Ask user for help
                    logger.info(f"Escalating to user: {recovery_plan.user_question}")
                    return AgentResponse.needs_input(
                        question=recovery_plan.user_question,
                        reasoning=recovery_plan.reasoning,
                    )
                else:
                    # Can't recover
                    break

        # All retries exhausted
        return AgentResponse.error(
            message=f"Failed after {attempt} attempts. Last error: {str(last_error)}",
            metadata={"attempts": attempt},
        )

    async def _execute_single_attempt(
        self, request: AgentRequest, attempt: int
    ) -> AgentResponse:
        """
        Single execution attempt.
        """
        # Step 1: LLM understands the task
        understanding = await self._llm_understand_task(request)

        # Step 2: LLM creates execution plan
        plan = await self._llm_create_plan(understanding)

        # Step 3: Execute plan
        context = ExecutionContext(
            thread_id=request.thread_id or "",
            user_id=request.user_id or "",
            task_id=request.task_id,
        )

        results = []
        for step in plan.steps:
            result = await self._execute_step(step, context)
            results.append(result)

            # Store result for subsequent steps
            context.set(f"step_{step.step_number}_result", result)

            # Check if step failed
            if not result.success:
                raise Exception(f"Step {step.step_number} failed: {result.error}")

        # Step 4: LLM synthesizes final response
        final_response = await self._llm_synthesize_response(
            results=results, understanding=understanding, request=request
        )

        return final_response

    async def _llm_understand_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Use LLM to understand the task intent and requirements.
        """
        capabilities_context = self.capability_registry.to_llm_context()

        system_prompt = f"""You are an intelligent agent understanding system.
Analyze the user's request and extract key information.

You have these capabilities available:
{capabilities_context}"""

        user_prompt = f"""User Request: {request.prompt}

Analyze this request and provide:
1. Primary intent (what does the user want to achieve?)
2. Key entities (files, data, specific items mentioned)
3. Implicit requirements (what else might they need?)
4. Complexity level (simple/medium/complex)

Respond with structured JSON."""

        # Use structured output
        from pydantic import BaseModel, Field

        class TaskUnderstanding(BaseModel):
            intent: str = Field(description="Primary intent of the task")
            entities: Dict[str, Any] = Field(description="Key entities identified")
            implicit_needs: List[str] = Field(description="Implicit requirements")
            complexity: str = Field(description="simple, medium, or complex")
            confidence: float = Field(description="Confidence in understanding (0-1)")

        response = await self.services.inference.generate_structured(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            schema=TaskUnderstanding,
            temperature=self.config.llm_temperature,
        )

        # Convert Pydantic model to dict
        return response.model_dump()

    async def _llm_create_plan(self, understanding: Dict[str, Any]) -> ExecutionPlan:
        """
        Use LLM to create execution plan with capabilities.
        """
        capabilities_context = self.capability_registry.to_llm_context()

        system_prompt = f"""You are an intelligent agent planning system.
Create a step-by-step plan to accomplish the task using available capabilities.

Guidelines:
- Break complex tasks into logical steps
- Each step should use one capability
- Consider dependencies between steps
- Include fallback options for critical steps"""

        user_prompt = f"""Task Understanding:
- Intent: {understanding.get("intent")}
- Entities: {understanding.get("entities")}
- Complexity: {understanding.get("complexity")}

Available Capabilities:
{capabilities_context}

Create an execution plan with:
1. Step-by-step breakdown
2. Capability to use for each step
3. Parameters for each capability
4. Expected outcome
5. Fallback strategy

Respond with structured JSON."""

        from pydantic import BaseModel, Field

        class PlanStep(BaseModel):
            step_number: int
            capability_name: str
            description: str
            parameters: Dict[str, Any]
            expected_outcome: str
            fallback_capability: Optional[str] = None

        class PlanCreation(BaseModel):
            reasoning: str
            steps: List[PlanStep]
            estimated_complexity: str
            fallback_strategy: Optional[str] = None

        response = await self.services.inference.generate_structured(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            schema=PlanCreation,
            temperature=self.config.llm_temperature,
        )

        return ExecutionPlan(
            reasoning=response.reasoning,
            steps=[
                ExecutionStep(
                    step_number=s.step_number,
                    capability_name=s.capability_name,
                    description=s.description,
                    parameters=s.parameters,
                    expected_outcome=s.expected_outcome,
                    fallback_capability=s.fallback_capability,
                )
                for s in response.steps
            ],
            estimated_complexity=response.estimated_complexity,
            fallback_strategy=response.fallback_strategy,
        )

    async def _execute_step(
        self, step: ExecutionStep, context: ExecutionContext
    ) -> CapabilityResult:
        """
        Execute a single step using the appropriate capability.
        """
        capability = self.capability_registry.get(step.capability_name)

        if not capability:
            return CapabilityResult.fail(
                error=f"Capability not found: {step.capability_name}"
            )

        logger.debug(f"Executing step {step.step_number}: {step.capability_name}")

        # Execute capability
        result = await capability.execute(
            agent=self, params=step.parameters, context=context
        )

        return result

    async def _llm_synthesize_response(
        self,
        results: List[CapabilityResult],
        understanding: Dict[str, Any],
        request: AgentRequest,
    ) -> AgentResponse:
        """
        Use LLM to synthesize final response from step results.
        """
        system_prompt = """You are an intelligent response synthesis system.
Combine the results from multiple execution steps into a coherent response.

Guidelines:
- Summarize key findings
- Highlight important data
- Provide actionable insights
- Use natural, helpful language"""

        # Build results summary
        results_summary = "\n".join(
            [
                f"Step {i + 1}: {'Success' if r.success else 'Failed'} - {r.data if r.success else r.error}"
                for i, r in enumerate(results)
            ]
        )

        user_prompt = f"""Original Request: {request.prompt}
Task Intent: {understanding.get("intent")}

Execution Results:
{results_summary}

Synthesize a helpful response that:
1. Answers the user's request
2. Summarizes what was accomplished
3. Presents any important data or insights
4. Suggests next steps if appropriate

Respond with structured JSON."""

        from pydantic import BaseModel, Field

        class SynthesizedResponse(BaseModel):
            summary: str = Field(description="Brief summary of what was done")
            detailed_result: str = Field(description="Detailed response")
            key_data: Optional[Dict[str, Any]] = Field(
                description="Important data points"
            )
            next_steps: Optional[List[str]] = Field(
                description="Suggested next actions"
            )

        response = await self.services.inference.generate_structured(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            schema=SynthesizedResponse,
            temperature=0.3,  # Slightly creative for natural language
        )

        return AgentResponse.success(
            result=response.detailed_result,
            summary=response.summary,
            data=response.key_data,
            metadata={"next_steps": response.next_steps},
        )

    async def _llm_plan_recovery(
        self, error: Exception, request: AgentRequest, attempt: int
    ) -> RecoveryPlan:
        """
        Use LLM to analyze error and plan recovery strategy.
        """
        system_prompt = """You are an error recovery planning system.
Analyze the error and decide the best recovery strategy.

Strategies:
1. RETRY - Fix parameters and try again (transient errors)
2. ALTERNATE - Use different approach (method doesn't work)
3. ESCALATE - Ask user for help (ambiguous situation)
4. FAIL - Cannot recover (critical error)"""

        capabilities_context = self.capability_registry.to_llm_context()

        user_prompt = f"""Error: {error.__class__.__name__}: {str(error)}
Attempt: {attempt}/3
Original Request: {request.prompt}

Available Capabilities:
{capabilities_context}

Analyze this error and decide:
1. What went wrong?
2. Which recovery strategy is best?
3. How confident are you (0-1)?
4. Specific fixes or alternatives?

Respond with structured JSON."""

        from pydantic import BaseModel, Field

        class RecoveryDecision(BaseModel):
            analysis: str = Field(description="What went wrong")
            strategy: str = Field(description="retry, alternate, escalate, or fail")
            confidence: float = Field(description="Confidence in strategy (0-1)")
            reasoning: str = Field(description="Why this strategy")
            fixes: Optional[Dict[str, Any]] = Field(
                description="Parameter adjustments for retry"
            )
            alternative: Optional[str] = Field(
                description="Alternative capability for alternate strategy"
            )
            user_question: Optional[str] = Field(
                description="Question for user if escalating"
            )

        response = await self.services.inference.generate_structured(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            schema=RecoveryDecision,
            temperature=self.config.llm_temperature,
        )

        return RecoveryPlan(
            strategy=response.strategy,
            reasoning=f"{response.analysis}. {response.reasoning}",
            confidence=response.confidence,
            adjusted_parameters=response.fixes,
            alternative_capability=response.alternative,
            user_question=response.user_question,
        )

    async def _apply_recovery_fixes(
        self, request: AgentRequest, recovery_plan: RecoveryPlan
    ) -> AgentRequest:
        """
        Apply recovery fixes to request.
        """
        # Create modified request with fixes
        modified = AgentRequest(
            prompt=request.prompt,
            action=request.action,
            payload={**request.payload, **(recovery_plan.adjusted_parameters or {})},
            task_id=request.task_id,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )
        return modified

    async def _execute_alternative_approach(
        self, request: AgentRequest, recovery_plan: RecoveryPlan
    ) -> AgentResponse:
        """
        Execute alternative approach using different capabilities.
        """
        logger.info(
            f"Executing alternative approach: {recovery_plan.alternative_capability}"
        )

        # Modify request to use alternative
        modified_request = AgentRequest(
            prompt=request.prompt
            + f"\n[Use alternative: {recovery_plan.alternative_capability}]",
            action=recovery_plan.alternative_capability,
            payload=request.payload,
            task_id=request.task_id,
            thread_id=request.thread_id,
            user_id=request.user_id,
        )

        # Execute with the alternative
        return await self._execute_single_attempt(modified_request, attempt=0)

    async def terminate(self):
        """Cleanup agent resources."""
        logger.info(f"Terminating agent: {self.agent_name}")

        await self._cleanup_resources()

        self._initialized = False
        self.status = AgentStatus.TERMINATED

        logger.info(f"Agent {self.agent_name} terminated")

    async def _cleanup_resources(self):
        """Override in subclass for agent-specific cleanup."""
        pass

    def get_capabilities_info(self) -> List[Dict[str, Any]]:
        """Get information about all capabilities."""
        return [cap.to_dict() for cap in self.capability_registry.list_all()]

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent metrics and telemetry.
        Override in subclass for agent-specific metrics.
        """
        import time

        uptime_seconds = 0
        if self._spawn_time:
            uptime_seconds = time.time() - self._spawn_time

        metrics = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "initialized": self._initialized,
            "uptime_seconds": uptime_seconds,
            "capabilities_count": len(self.capability_registry.list_all()),
            "execution_history_count": len(self.execution_history),
        }

        # Add telemetry metrics if available
        if self.config.enable_telemetry and self.services.telemetry:
            try:
                telemetry_metrics = await self.services.telemetry.get_agent_metrics(
                    agent_name=self.agent_name
                )
                metrics["telemetry"] = telemetry_metrics
            except Exception:
                metrics["telemetry"] = {"error": "Failed to retrieve telemetry"}

        # Allow subclasses to add custom metrics
        custom_metrics = await self._get_custom_metrics()
        if custom_metrics:
            metrics["custom"] = custom_metrics

        return metrics

    async def _get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        """Override in subclass to provide custom metrics."""
        return None

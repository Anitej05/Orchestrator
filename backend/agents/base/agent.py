"""
Base Agent
Abstract base class for all intelligent agents.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

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
from .capability import CapabilityRegistry

logger = logging.getLogger(__name__)


class _StepFailure(Exception):
    """Internal exception that carries completed step results from a failed attempt.
    
    Used by _execute_single_attempt to communicate which steps succeeded
    before the failure, so _execute_with_recovery can skip them on retry.
    """

    def __init__(
        self,
        failed_step: int,
        original_error: Exception,
        completed_results: list,
    ):
        self.failed_step = failed_step
        self.original_error = original_error
        self.completed_results = completed_results
        super().__init__(str(original_error))


class ExecutionMode(str, Enum):
    PLAN_EXECUTE = "plan_execute"
    REACT = "react"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    max_retries: int = 3
    use_llm_recovery: bool = True
    llm_temperature: float = 0.0  # Deterministic for planning
    enable_telemetry: bool = True
    request_timeout: float = 120.0
    execution_mode: ExecutionMode = ExecutionMode.REACT
    max_react_steps: int = 15


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

        # Per-request progress queue (set by /execute/stream endpoint)
        self._progress_queue: Optional[asyncio.Queue] = None

        logger.info(f"BaseAgent initialized: {agent_name} ({agent_id})")

    # ------------------------------------------------------------------
    # Credential helpers — delegate to CredentialManager singleton
    # ------------------------------------------------------------------

    def get_credential(self, key: str, user_id: str = "system") -> Optional[str]:
        """
        Get a single credential for this agent from the DB (with .env fallback).

        Usage from any agent subclass::

            api_key = self.get_credential("COMPOSIO_API_KEY")
        """
        return self.services.credentials.get(
            scope="agent",
            scope_id=self.agent_id,
            key=key,
            user_id=user_id,
        )

    def get_all_credentials(self, user_id: str = "system") -> Dict[str, str]:
        """
        Get all credentials for this agent.
        Returns a dict of ``{credential_name: plaintext_value}``.
        """
        return self.services.credentials.get_all(
            scope="agent",
            scope_id=self.agent_id,
            user_id=user_id,
        )

    async def emit_progress(self, message: str) -> None:
        """Push a progress message to the streaming queue (no-op if not streaming)."""
        if self._progress_queue is not None:
            try:
                self._progress_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # Drop silently if queue is full

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
            # ── Direct dispatch path ──────────────────────────────────────
            # When the caller explicitly sets request.action, bypass LLM
            # planning and call the capability directly with request.payload.
            # This is used by tests, the orchestrator, and any client that
            # already knows the exact capability name + parameters.
            if request.action:
                capability = self.capability_registry.get(request.action)
                if capability:
                    context = ExecutionContext(
                        thread_id=request.thread_id or "",
                        user_id=request.user_id or "",
                        task_id=request.task_id,
                    )
                    cap_result = await capability.execute(
                        agent=self,
                        params={**{"prompt": request.prompt}, **(request.payload or {})},
                        context=context,
                    )
                    # Reset status to READY *before* inspecting cap_result.
                    # This must happen regardless of success/failure so the agent
                    # isn't stuck in BUSY if the capability raises or returns an error.
                    # The caller (agent_manager) reads .status after this returns, so
                    # it must reflect the idle state by the time we branch on success.
                    self.status = AgentStatus.READY
                    if cap_result.success:
                        # Surface metadata (message, canvas_display, etc.) in AgentResponse
                        meta = cap_result.metadata or {}
                        return AgentResponse.success(
                            result=cap_result.data,
                            summary=meta.get("message", f"{request.action} completed"),
                            data=meta,
                            execution_time_ms=(time.time() - start_time) * 1000,
                        )
                    else:
                        return AgentResponse.error(
                            message=cap_result.error or "Capability failed",
                            execution_time_ms=(time.time() - start_time) * 1000,
                        )
                else:
                    logger.warning(
                        f"Action '{request.action}' not found in registry, "
                        "falling back to LLM planning"
                    )

            # ── LLM-driven planning path ──────────────────────────────────
            # Execute with recovery
            result = await self._execute_with_recovery(request)

            # Log telemetry
            if self.config.enable_telemetry:
                duration_ms = (time.time() - start_time) * 1000
                self.services.telemetry.log_agent_call(
                    agent_name=self.agent_name,
                    success=result.status != "error",
                    duration_ms=duration_ms,
                    user_id=request.user_id,
                    thread_id=request.thread_id
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
        Tracks completed steps to skip them on retry.
        """
        attempt = 0
        last_error = None
        completed_results: List[CapabilityResult] = []

        while attempt < self.config.max_retries:
            try:
                # Dispatch based on execution mode
                if self.config.execution_mode == ExecutionMode.REACT:
                    return await self._execute_react_attempt(
                        request, attempt, completed_results=completed_results
                    )
                else:
                    return await self._execute_single_attempt(
                        request, attempt, completed_results=completed_results
                    )

            except _StepFailure as sf:
                # Preserve results from steps that completed before the failure
                completed_results = sf.completed_results
                attempt += 1
                last_error = sf.original_error
                logger.warning(
                    f"Execution attempt {attempt} failed at step {sf.failed_step}: "
                    f"{sf.original_error} ({len(completed_results)} steps already completed)"
                )

                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    break

                if not self.config.use_llm_recovery:
                    # Simple retry without LLM
                    continue

                # LLM analyzes error and plans recovery
                try:
                    recovery_plan = await self._llm_plan_recovery(
                        error=sf.original_error, request=request, attempt=attempt
                    )
                except Exception as recovery_err:
                    logger.warning(
                        f"Recovery planning itself failed: {recovery_err}. "
                        "Falling back to simple retry."
                    )
                    continue

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
                        metadata={"reasoning": recovery_plan.reasoning},
                    )
                else:
                    # Can't recover
                    break

            except Exception as e:
                # Non-step failures (e.g. LLM planning errors) — reset completed
                completed_results = []
                attempt += 1
                last_error = e
                logger.warning(f"Execution attempt {attempt} failed: {e}")

                if attempt >= self.config.max_retries:
                    logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                    break

        # All retries exhausted
        return AgentResponse.error(
            message=f"Failed after {attempt} attempts. Last error: {str(last_error)}",
            metadata={"attempts": attempt},
        )

    async def _execute_single_attempt(
        self, request: AgentRequest, attempt: int,
        completed_results: List[CapabilityResult] = None,
    ) -> AgentResponse:
        """
        Single execution attempt.
        Skips already-completed steps on retry to avoid redundant work.
        """
        completed_results = completed_results or []
        skip_count = len(completed_results)

        # Step 1: LLM understands the task
        understanding = await self._llm_understand_task(request)

        # Step 2: LLM creates execution plan
        plan = await self._llm_create_plan(understanding, request)

        # Step 3: Execute plan
        context = ExecutionContext(
            thread_id=request.thread_id or "",
            user_id=request.user_id or "",
            task_id=request.task_id,
        )

        # Seed context with results from previously completed steps
        results: List[CapabilityResult] = []
        for i, prev_result in enumerate(completed_results):
            results.append(prev_result)
            context.set(f"step_{i + 1}_result", prev_result)

        for step in plan.steps:
            # Skip steps that already completed successfully in a prior attempt
            if step.step_number <= skip_count:
                logger.info(
                    f"[SKIP] Step {step.step_number}/{len(plan.steps)} already completed, skipping"
                )
                continue

            # Enrich step params with outputs from prior steps
            step = self._enrich_step_params(step, results)

            result = await self._execute_step(step, context)
            results.append(result)

            # Store result for subsequent steps
            context.set(f"step_{step.step_number}_result", result)

            # Check if step failed — raise _StepFailure so retry can skip done steps
            if not result.success:
                raise _StepFailure(
                    failed_step=step.step_number,
                    original_error=Exception(
                        f"Step {step.step_number} failed: {result.error}"
                    ),
                    completed_results=results[:-1],  # exclude the failed step
                )

        # Step 4: LLM synthesizes final response
        final_response = await self._llm_synthesize_response(
            results=results, understanding=understanding, request=request
        )

        return final_response

    async def _get_step_context(self, request: AgentRequest, context: ExecutionContext, previous_results: List[CapabilityResult]) -> Any:
        """
        Virtual hook: Subclasses should override this to return current environment context.
        For example, BrowserAgent returns DOM state and screenshots; SpreadsheetAgent returns active DataFrames.
        """
        return None

    async def _update_state_post_step(self, step: ExecutionStep, result: CapabilityResult, context: ExecutionContext) -> None:
        """
        Virtual hook: Subclasses override to run post-step logic.
        For example, Visual state verification, or maintaining conversational memory.
        """
        pass

    async def _execute_react_attempt(
        self, request: AgentRequest, attempt: int,
        completed_results: List[CapabilityResult] = None,
    ) -> AgentResponse:
        """
        Iterative Execution (ReAct) Attempt.
        The agent observes state, decides next action, executes, and repeats.
        """
        completed_results = completed_results or []
        skip_count = len(completed_results)

        # Step 1: LLM understands the task
        await self.emit_progress("Analyzing task...")
        understanding = await self._llm_understand_task(request)

        context = ExecutionContext(
            thread_id=request.thread_id or "",
            user_id=request.user_id or "",
            task_id=request.task_id,
        )

        # Seed context with results from previously completed steps
        results: List[CapabilityResult] = []
        for i, prev_result in enumerate(completed_results):
            results.append(prev_result)
            context.set(f"step_{i + 1}_result", prev_result)

        for step_num in range(1, self.config.max_react_steps + 1):
            if step_num <= skip_count:
                logger.info(f"[SKIP] React Step {step_num} already completed, skipping")
                continue

            # 1. Observe state
            step_context = await self._get_step_context(request, context, results)

            # 2. Decide next action
            await self.emit_progress(f"Deciding next action (step {step_num})...")
            plan_step = await self._llm_decide_next_react_step(request, understanding, step_context, results, step_num)

            # 3. Check for completion or escalation
            if plan_step.capability_name.lower() in ["finish", "complete", "done"]:
                break
            if plan_step.capability_name.lower() == "escalate":
                return AgentResponse.needs_input(
                    question=plan_step.description,
                    metadata={"reasoning": "Agent self-escalated"}
                )
            
            # Construct ExecutionStep
            step = ExecutionStep(
                step_number=step_num,
                capability_name=plan_step.capability_name,
                description=plan_step.description,
                parameters=plan_step.parameters,
                expected_outcome=plan_step.expected_outcome
            )

            # Execution
            cap_label = plan_step.capability_name.replace("_", " ")
            await self.emit_progress(f"Running: {cap_label}...")
            result = await self._execute_step(step, context)
            results.append(result)
            context.set(f"step_{step_num}_result", result)

            if result.success:
                await self.emit_progress(f"Completed: {cap_label}")
            else:
                await self.emit_progress(f"Step failed: {cap_label} — retrying...")

            # 4. Update post step
            await self._update_state_post_step(step, result, context)

            # In ReAct, if a step fails, we expose it to the standard error recovery
            if not result.success:
                raise _StepFailure(
                    failed_step=step.step_number,
                    original_error=Exception(
                        f"ReAct step {step.step_number} failed: {result.error}"
                    ),
                    completed_results=results[:-1],
                )

        # Step 5: Synthesize
        await self.emit_progress("Synthesizing answer...")
        final_response = await self._llm_synthesize_response(
            results=results, understanding=understanding, request=request
        )

        return final_response

    async def _llm_decide_next_react_step(self, request: AgentRequest, understanding: Dict[str, Any], step_context: Any, previous_results: List[CapabilityResult], step_num: int):
        """LLM decides the next immediate action based on current state."""
        from pydantic import BaseModel, Field, model_validator

        class ReactDecision(BaseModel):
            reasoning: str = Field(description="Reasoning based on current context and past results on what to do next.")
            capability_name: str = Field(description="The name of the capability to use. Use 'finish' if the goal is completely achieved.")
            description: str = Field(description="Description of what this action will achieve, or question if escalating.")
            parameters: Dict[str, Any] = Field(description="Parameters for the capability, or empty if none.")
            expected_outcome: str = Field(description="What you expect to observe after this action.")

            @model_validator(mode="after")
            def coerce_fields(self):
                if not isinstance(self.parameters, dict):
                    self.parameters = {}
                return self

        sys_prompt = f"""You are determining the NEXT immediate action for the agent: '{self.agent_name}'.
You are using a ReAct (Reason + Act) loop. 
Review the original task, the current context, and previous step results.

AVAILABLE CAPABILITIES:
"""
        for count, cap in enumerate(self.capability_registry.list_all(), 1):
            sys_prompt += f"{count}. {cap.name}: {cap.description}\\n"
            if isinstance(cap.parameters, list):
                sys_prompt += f"   Parameters: {[p.model_dump() if hasattr(p, 'model_dump') else p.__dict__ for p in cap.parameters]}\n"
            else:
                sys_prompt += f"   Parameters: {cap.parameters.schema() if hasattr(cap.parameters, 'schema') else str(cap.parameters)}\n"

        sys_prompt += """
SPECIAL CAPABILITIES:
- finish: Use this when the user's task is fully complete. Leave parameters empty.
- escalate: Use this to ask the user a clarifying question before proceeding. Set the question as the description.

Your job: output the singular NEXT step to take. Output strictly as JSON.
"""
        
        # Prepare context payload
        user_content = f"Task: {request.prompt}\n\nUnderstanding: {understanding}\n\n"
        if previous_results:
            user_content += "Previous Step Results:\n"
            for i, res in enumerate(previous_results, 1):
                user_content += f"Step {i}: Success={res.success}, Output={str(res.data)[:500]}\n"
        
        user_content += f"\nCurrent State Context:\n{str(step_context)[:2000]}\n\n"
        user_content += f"Determine Step {step_num}."

        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_content)
        ]

        logger.info(f"LLM: Deciding next ReAct step ({step_num})")
        decision = await self.services.inference.generate_structured(
            messages=messages,
            schema=ReactDecision,
            temperature=self.config.llm_temperature
        )
        return decision

    def _enrich_step_params(
        self, step: ExecutionStep, previous_results: List[CapabilityResult]
    ) -> ExecutionStep:
        """
        Inject outputs from previous steps into the current step's parameters.
        Looks for file_id/file_path in prior step results and threads them forward.
        """
        if not previous_results:
            return step

        # Find the latest file_id produced by a previous step
        latest_file_id = None
        latest_file_path = None
        for prev in reversed(previous_results):
            if not prev.success:
                continue
            data = prev.data if isinstance(prev.data, dict) else {}
            meta = prev.metadata if isinstance(prev.metadata, dict) else {}
            # Check data first, then metadata
            fid = data.get("file_id") or meta.get("file_id")
            fpath = data.get("file_path") or meta.get("file_path")
            if fid:
                latest_file_id = fid
                latest_file_path = fpath
                break

        if not latest_file_id:
            return step

        # Only inject if the step doesn't already have a valid file_id
        params = dict(step.parameters)
        current_fid = params.get("file_id")

        # Inject if: no file_id set, or the LLM set a placeholder that likely won't resolve
        if not current_fid or current_fid != latest_file_id:
            logger.info(
                f"[ENRICH] Step {step.step_number}: injecting file_id='{latest_file_id}' "
                f"(was '{current_fid}')"
            )
            params["file_id"] = latest_file_id
            if latest_file_path:
                params.setdefault("file_path", latest_file_path)

        # Return a new step with enriched params (ExecutionStep is a dataclass)
        from dataclasses import replace
        return replace(step, parameters=params)

    async def _llm_understand_task(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Use LLM to understand the task intent and requirements.
        """
        capabilities_context = self.capability_registry.to_llm_context()

        system_prompt = f"""You are an intelligent agent understanding system.
You are running LOCALLY on the user's machine with FULL access to the local filesystem.
You CAN read and write files at any path the user provides. Never say you cannot access local files.
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

    async def _llm_create_plan(
        self, understanding: Dict[str, Any], request: AgentRequest = None
    ) -> ExecutionPlan:
        """
        Use LLM to create execution plan with capabilities.
        """
        capabilities_context = self.capability_registry.to_llm_context()

        # Build session context: tell the LLM what data is already loaded
        session_context = ""
        thread_id = (request.thread_id if request else None) or "default"
        has_loaded_data = False
        if hasattr(self, "state"):
            session = None
            if hasattr(self.state, "get"):
                session = self.state.get(thread_id)
            elif hasattr(self.state, "get_or_create"):
                session = self.state.get_or_create(thread_id)
            if session and hasattr(session, "dataframes") and session.dataframes:
                has_loaded_data = True
                loaded_info = []
                for fid, df in session.dataframes.items():
                    loaded_info.append(
                        f"  - file_id='{fid}', {df.shape[0]} rows × {df.shape[1]} cols, columns: {list(df.columns)}"
                    )
                session_context = (
                    "\n\nALREADY LOADED DATA (thread_id='" + thread_id + "'):\n"
                    + "\n".join(loaded_info)
                    + "\n\nIMPORTANT: Do NOT call load_file again for these files. "
                    "Use the file_id and thread_id directly in your plan parameters."
                )
        
        has_load_file = "load_file" in [cap.name for cap in self.capability_registry.list_all()]
        
        # CRITICAL FIX: Explicitly tell LLM when no data is loaded
        if not has_loaded_data and has_load_file:
            session_context = (
                "\n\n**NO DATA LOADED YET**\n"
                "You MUST include a load_file step FIRST before any data processing steps. "
                "Extract the file path from the user's request and use it in load_file."
            )

        system_prompt = """You are an intelligent agent planning system.
You are running LOCALLY on the user's machine with FULL access to the local filesystem.
You CAN read and write files at any path. Never ask the user to upload or share files — just use load_file with the path.
Create a step-by-step plan to accomplish the task using available capabilities.

Guidelines:
- Break complex tasks into logical steps
- Each step should use one capability
- Consider dependencies between steps
- Include fallback options for critical steps
- If data is already loaded, reference it by file_id and thread_id — do NOT re-load it
- If a file path is mentioned, use it directly with load_file — you have full filesystem access"""

        user_prompt = f"""Task Understanding:
- Intent: {understanding.get("intent")}
- Entities: {understanding.get("entities")}
- Complexity: {understanding.get("complexity")}
{session_context}

Available Capabilities:
{capabilities_context}

Create an execution plan with:
1. Step-by-step breakdown
2. Capability to use for each step
3. Parameters for each capability (include thread_id='{thread_id}' and the correct file_id)
4. Expected outcome
5. Fallback strategy

Respond with structured JSON."""

        from pydantic import BaseModel, model_validator

        class PlanStep(BaseModel):
            step_number: int
            capability_name: str
            description: str
            parameters: Dict[str, Any]
            expected_outcome: str
            fallback_capability: Optional[Any] = None

        class PlanCreation(BaseModel):
            reasoning: str
            steps: List[PlanStep]
            estimated_complexity: str
            fallback_strategy: Optional[Any] = None

            @model_validator(mode="after")
            def coerce_fields(self):
                """Coerce dict/non-string values for fallback_strategy."""
                if self.fallback_strategy and not isinstance(self.fallback_strategy, str):
                    if isinstance(self.fallback_strategy, dict):
                        self.fallback_strategy = self.fallback_strategy.get("description", str(self.fallback_strategy))
                    else:
                        self.fallback_strategy = str(self.fallback_strategy)
                return self

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
                    fallback_capability=(
                        s.fallback_capability.get("capability_name")
                        if isinstance(s.fallback_capability, dict)
                        else str(s.fallback_capability)
                        if s.fallback_capability
                        else None
                    ),
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

        # Always enforce correct thread_id from request context.
        # LLM plans often omit it OR pass wrong values ("", null, "default"),
        # which would create a new empty session instead of using the loaded data.
        params = dict(step.parameters)
        params["thread_id"] = context.thread_id if context.thread_id else "default"

        # Execute capability
        result = await capability.execute(
            agent=self, params=params, context=context
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

        # Collect canvas_display from the last step that produced one
        canvas = None
        for r in reversed(results):
            if r.metadata and r.metadata.get("canvas_display"):
                canvas = r.metadata["canvas_display"]
                break

        return AgentResponse.success(
            result=response.detailed_result,
            summary=response.summary,
            data=response.key_data,
            canvas_display=canvas,
            metadata={"next_steps": response.next_steps},
        )

    async def _llm_plan_recovery(
        self, error: Exception, request: AgentRequest, attempt: int
    ) -> RecoveryPlan:
        """
        Use LLM to analyze error and plan recovery strategy.
        """
        system_prompt = """You are an error recovery planning system.
You are running LOCALLY on the user's machine with FULL access to the local filesystem.
You CAN read and write files at any path. Never escalate just because a file path was mentioned — you CAN access it.
Analyze the error and decide the best recovery strategy.

Strategies:
1. RETRY - Fix parameters and try again (transient errors, wrong params)
2. ALTERNATE - Use different approach (method doesn't work)
3. ESCALATE - Ask user for help ONLY if truly ambiguous — NOT for file access issues
4. FAIL - Cannot recover (critical error)

Prefer RETRY over ESCALATE when the error is about missing data or wrong parameters."""

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

        from pydantic import BaseModel, Field, model_validator

        class RecoveryDecision(BaseModel):
            analysis: str = Field(description="What went wrong")
            strategy: str = Field(description="retry, alternate, escalate, or fail")
            confidence: float = Field(description="Confidence in strategy (0-1)")
            reasoning: str = Field(description="Why this strategy")
            fixes: Optional[Any] = Field(
                default=None,
                description="Parameter adjustments for retry"
            )
            alternative: Optional[Any] = Field(
                default=None,
                description="Alternative capability name for alternate strategy"
            )
            user_question: Optional[Any] = Field(
                default=None,
                description="Question for user if escalating"
            )

            @model_validator(mode="after")
            def coerce_fields(self):
                """Coerce dict/non-string values to strings for safety."""
                if self.alternative and not isinstance(self.alternative, str):
                    self.alternative = str(self.alternative)
                if self.user_question and not isinstance(self.user_question, str):
                    self.user_question = str(self.user_question)
                # Coerce fixes: LLM sometimes returns a list of dicts instead of a dict
                if isinstance(self.fixes, list):
                    merged = {}
                    for item in self.fixes:
                        if isinstance(item, dict):
                            merged.update(item)
                    self.fixes = merged if merged else None
                return self

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

"""
Universal Agent - General Purpose Task Executor

A flexible agent capable of handling any arbitrary task through
LLM reasoning, code execution, and tool usage.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from backend.agents.base import BaseAgent, AgentRequest, AgentResponse, capability

logger = logging.getLogger(__name__)


class UniversalAgent(BaseAgent):
    """
    Universal Agent - handles any arbitrary task not covered by specialized agents.

    Capabilities:
    - General task execution with planning
    - Code generation and execution
    - Analysis and research
    - Creative writing
    - Problem solving
    """

    def __init__(
        self,
        agent_id="universal_agent",
        agent_name="Universal Agent",
        services=None,
        config=None,
    ):
        super().__init__(
            agent_id=agent_id, agent_name=agent_name, services=services, config=config
        )
        self.description = "General-purpose agent for arbitrary tasks"

    async def _initialize_resources(self):
        """Initialize agent-specific resources."""
        logger.info("Initializing Universal Agent resources")
        # No special resources needed for universal agent
        pass

    @capability(
        name="execute_task",
        description="Execute any arbitrary task through planning and execution",
    )
    async def execute_task(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Main entry point for task execution."""
        request = AgentRequest(**params)
        return await self._execute_general_task(request)

    @capability(
        name="analyze",
        description="Analyze data, text, or situations and provide insights",
    )
    async def analyze(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Analyze content and provide insights."""
        request = AgentRequest(**params)
        return await self._analyze_content(request)

    @capability(
        name="generate_code",
        description="Generate and optionally execute code to solve problems",
    )
    async def generate_code(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Generate code for the given task."""
        request = AgentRequest(**params)
        return await self._generate_code(request)

    @capability(
        name="research",
        description="Research topics and compile comprehensive information",
    )
    async def research(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Research a topic and compile information."""
        request = AgentRequest(**params)
        return await self._research_topic(request)

    @capability(
        name="creative_write",
        description="Create creative content like stories, poems, scripts",
    )
    async def creative_write(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Generate creative writing."""
        request = AgentRequest(**params)
        return await self._creative_writing(request)

    @capability(
        name="solve_problem",
        description="Break down and solve complex problems systematically",
    )
    async def solve_problem(self, params: Dict[str, Any], context) -> Dict[str, Any]:
        """Solve a complex problem step by step."""
        request = AgentRequest(**params)
        return await self._solve_problem(request)

    async def _execute_general_task(self, request: AgentRequest) -> Dict[str, Any]:
        """Execute a general task with planning and execution."""
        prompt = request.prompt
        logger.info(f"Universal Agent executing task: {prompt[:100]}...")

        # Step 1: Plan the task
        plan = await self._plan_task(prompt, request.payload)
        logger.info(f"Task plan: {plan}")

        # Step 2: Execute each step
        results = []
        for step in plan.get("steps", []):
            step_result = await self._execute_step(step, request)
            results.append(step_result)

            # If step fails, attempt recovery
            if not step_result.get("success", False):
                recovery = await self._attempt_recovery(step, step_result, request)
                if recovery:
                    results.append(recovery)

        # Step 3: Synthesize final response
        final_response = await self._synthesize_response(prompt, results)

        return {
            "result": final_response,
            "plan": plan,
            "steps_executed": len(results),
            "success": True,
        }

    async def _plan_task(self, prompt: str, payload: Dict) -> Dict:
        """Create a plan for executing the task."""
        planning_prompt = f"""You are a task planning assistant. Break down the following task into clear, executable steps.

Task: {prompt}

Context: {json.dumps(payload, default=str)}

Create a step-by-step plan. Each step should be:
1. Clear and specific
2. Executable on its own
3. Ordered logically

Respond in JSON format:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "What to do in this step",
            "action_type": "reasoning|code|research|writing",
            "estimated_difficulty": "easy|medium|hard"
        }}
    ],
    "estimated_total_time": "brief|moderate|extended",
    "requires_code": true/false,
    "requires_research": true/false
}}"""

        try:
            # Use the agent's LLM service for planning
            response = await self.llm_service.generate(
                prompt=planning_prompt,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Fallback to simple single-step plan
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "description": prompt,
                        "action_type": "reasoning",
                        "estimated_difficulty": "medium",
                    }
                ],
                "estimated_total_time": "brief",
                "requires_code": False,
                "requires_research": False,
            }

    async def _execute_step(self, step: Dict, request: AgentRequest) -> Dict:
        """Execute a single step from the plan."""
        step_num = step.get("step_number", 1)
        description = step.get("description", "")
        action_type = step.get("action_type", "reasoning")

        logger.info(f"Executing step {step_num}: {description}")

        try:
            if action_type == "code":
                return await self._execute_code_step(description, request)
            elif action_type == "research":
                return await self._execute_research_step(description, request)
            elif action_type == "writing":
                return await self._execute_writing_step(description, request)
            else:  # reasoning
                return await self._execute_reasoning_step(description, request)
        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            return {
                "step_number": step_num,
                "success": False,
                "error": str(e),
                "result": None,
            }

    async def _execute_reasoning_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a reasoning/analysis step."""
        response = await self.llm_service.generate(
            prompt=description,
            temperature=0.7,
            context=request.payload.get("context", {}),
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "reasoning",
        }

    async def _execute_code_step(self, description: str, request: AgentRequest) -> Dict:
        """Execute a code generation step."""
        # Generate code
        code_prompt = f"""Write Python code to accomplish the following:

{description}

Requirements:
- Write clean, well-commented code
- Include error handling
- Save any output to a 'result' variable
- Don't include example usage or test code

Provide the code in a code block."""

        response = await self.llm_service.generate(prompt=code_prompt, temperature=0.3)

        # Extract code from response
        code = self._extract_code(response)

        # Execute the code in sandbox
        try:
            from backend.services.code_sandbox_service import code_sandbox

            execution_result = await code_sandbox.execute(
                code=code, user_id=request.user_id, thread_id=request.thread_id
            )

            return {
                "step_number": 1,
                "success": execution_result.get("status") == "completed",
                "code": code,
                "result": execution_result.get("result"),
                "output": execution_result.get("output"),
                "type": "code",
            }
        except Exception as e:
            return {
                "step_number": 1,
                "success": False,
                "code": code,
                "error": str(e),
                "type": "code",
            }

    async def _execute_research_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a research step using web search."""
        # Use web search tool if available
        try:
            from backend.services.tool_registry_service import tool_registry

            tool_registry.initialize()
            search_tool = tool_registry.get_tool("web_search_and_summarize")

            if search_tool:
                search_result = await search_tool.ainvoke({"query": description})
                return {
                    "step_number": 1,
                    "success": True,
                    "result": search_result,
                    "type": "research",
                }
        except Exception as e:
            logger.warning(f"Web search failed, falling back to LLM: {e}")

        # Fallback to LLM knowledge
        response = await self.llm_service.generate(
            prompt=f"Research and provide comprehensive information about: {description}",
            temperature=0.5,
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "research",
            "source": "llm_fallback",
        }

    async def _execute_writing_step(
        self, description: str, request: AgentRequest
    ) -> Dict:
        """Execute a creative writing step."""
        response = await self.llm_service.generate(
            prompt=description,
            temperature=0.8,  # Higher creativity
            max_tokens=2000,
        )

        return {
            "step_number": 1,
            "success": True,
            "result": response,
            "type": "writing",
        }

    async def _attempt_recovery(
        self, step: Dict, step_result: Dict, request: AgentRequest
    ) -> Optional[Dict]:
        """Attempt to recover from a failed step."""
        logger.info(f"Attempting recovery for step {step.get('step_number')}")

        recovery_prompt = f"""The following step failed:
Step: {step.get("description")}
Error: {step_result.get("error")}

Suggest an alternative approach or workaround.
Be concise and practical."""

        try:
            recovery_suggestion = await self.llm_service.generate(
                prompt=recovery_prompt, temperature=0.4
            )

            # Create recovery step
            recovery_step = {
                "step_number": step.get("step_number"),
                "description": recovery_suggestion,
                "action_type": "reasoning",
                "is_recovery": True,
            }

            return await self._execute_step(recovery_step, request)
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return None

    async def _synthesize_response(
        self, original_prompt: str, results: List[Dict]
    ) -> str:
        """Synthesize final response from all step results."""
        if len(results) == 1:
            return results[0].get("result", "Task completed")

        synthesis_prompt = f"""Synthesize the following step results into a coherent final response.

Original Task: {original_prompt}

Step Results:
{json.dumps(results, default=str, indent=2)}

Provide a comprehensive but concise response that addresses the original task.
Highlight key findings, code outputs, or insights where relevant."""

        try:
            response = await self.llm_service.generate(
                prompt=synthesis_prompt, temperature=0.5
            )
            return response
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to concatenating results
            return "\n\n".join(
                [r.get("result", "") for r in results if r.get("result")]
            )

    async def _analyze_content(self, request: AgentRequest) -> Dict[str, Any]:
        """Analyze content and provide insights."""
        prompt = request.prompt

        analysis_prompt = f"""Analyze the following content and provide insights:

{prompt}

Provide:
1. Key points or findings
2. Patterns or trends
3. Insights or implications
4. Recommendations (if applicable)"""

        response = await self.llm_service.generate(
            prompt=analysis_prompt, temperature=0.4
        )

        return {"result": response, "type": "analysis", "success": True}

    async def _generate_code(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate code for the task."""
        return await self._execute_code_step(request.prompt, request)

    async def _research_topic(self, request: AgentRequest) -> Dict[str, Any]:
        """Research a topic comprehensively."""
        prompt = request.prompt

        # Multi-step research
        research_plan = [
            {
                "step_number": 1,
                "description": f"Search for current information about: {prompt}",
                "action_type": "research",
            },
            {
                "step_number": 2,
                "description": f"Analyze and synthesize findings about: {prompt}",
                "action_type": "reasoning",
            },
        ]

        results = []
        for step in research_plan:
            result = await self._execute_step(step, request)
            results.append(result)

        final_response = await self._synthesize_response(prompt, results)

        return {
            "result": final_response,
            "findings": results,
            "type": "research",
            "success": True,
        }

    async def _creative_writing(self, request: AgentRequest) -> Dict[str, Any]:
        """Generate creative writing."""
        return await self._execute_writing_step(request.prompt, request)

    async def _solve_problem(self, request: AgentRequest) -> Dict[str, Any]:
        """Solve a complex problem."""
        prompt = request.prompt

        problem_solving_prompt = f"""Solve the following problem step by step:

{prompt}

Approach:
1. Understand the problem clearly
2. Break it down into components
3. Solve each component
4. Combine solutions
5. Verify the answer

Show your work and reasoning clearly."""

        response = await self.llm_service.generate(
            prompt=problem_solving_prompt, temperature=0.5
        )

        return {"result": response, "type": "problem_solving", "success": True}

    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        import re

        # Try to extract code block
        code_block_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        code_block_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code block, return the whole response
        return response.strip()

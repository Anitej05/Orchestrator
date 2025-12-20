"""
Browser Agent - Planner

Decomposes tasks into executable steps.
"""

import logging
import json
from typing import List
from .state import AgentMemory, Subtask
from .llm import LLMClient

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def create_initial_plan(self, task: str) -> List[Subtask]:
        """Break down the main task into subtasks"""
        prompt = f"""You are an expert planning agent. Break down this browser automation task into logical, sequential subtasks.

MAIN TASK: {task}

GUIDELINES:
1. **BE CONCISE**. Create the MINIMUM number of steps possible.
2. Merge navigation and waiting.
3. Merge search and interaction.
4. Merge analysis, extraction, and verification into a single "Analyze" or "Extract" step.
5. For "Go to X and describe Y", it should be just 2 steps: "Navigate to X", "Analyze Y".

Respond with valid JSON only:
{{
    "subtasks": [
        "Navigate to https://www.google.com",
        "Analyze the doodle image and description"
    ]
}}
"""
        response = await self.llm.call_llm_direct(prompt)
        if not response:
            logger.error("Failed to generate plan.")
            return []

        try:
            # clean json
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                steps = data.get("subtasks", [])
                return [
                    Subtask(id=i+1, description=desc, status="pending") 
                    for i, desc in enumerate(steps)
                ]
        except Exception as e:
            logger.error(f"Plan parsing failed: {e}")
        
        # Fallback plan
        return [Subtask(id=1, description=f"Execute task: {task}", status="pending")]

    async def update_plan(self, memory: AgentMemory, last_result: str):
        """Optional: Update plan based on new findings (Not fully implemented yet)"""
        pass

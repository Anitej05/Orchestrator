"""
Browser Agent - Planner

Decomposes tasks into executable steps and handles dynamic replanning.
"""

import logging
import json
import re
from typing import List, Optional, Tuple
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

    async def update_plan(self, memory: AgentMemory, failure_context: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Intelligently replan based on task progress and learned context.
        
        Args:
            memory: Current agent memory with task history
            failure_context: Optional context about why replanning was triggered
            
        Returns:
            Tuple of (should_update: bool, new_subtasks: List[str])
        """
        # Categorize subtasks by status
        completed = [t for t in memory.plan if t.status == "completed"]
        failed = [t for t in memory.plan if t.status == "failed"]
        pending = [t for t in memory.plan if t.status in ["pending", "active"]]
        
        # Only replan if we have failures or significant observations
        if not failed and not memory.observations:
            logger.info("📋 No replanning needed - no failures or new observations")
            return False, []
        
        # Build context from completed work
        completed_summary = "\n".join([
            f"  ✅ {t.description}" + (f" → Result: {t.result}" if t.result else "")
            for t in completed
        ]) or "  (none)"
        
        # Build context from failures with reasons
        failed_summary = "\n".join([
            f"  ❌ {t.description}" + (f" → Reason: {t.result}" if t.result else "")
            for t in failed
        ]) or "  (none)"
        
        # Build observations context
        observations_summary = "\n".join([
            f"  • {key}: {value}"
            for key, value in memory.observations.items()
        ]) or "  (none)"
        
        # Pending tasks
        pending_summary = "\n".join([
            f"  ⏳ {t.description}"
            for t in pending
        ]) or "  (none)"
        
        prompt = f"""You are an expert replanning agent for browser automation. Analyze the current task state and decide if the remaining plan should be modified.

═══════════════════════════════════════════════════════════════════════════════
ORIGINAL TASK
═══════════════════════════════════════════════════════════════════════════════
{memory.task}

═══════════════════════════════════════════════════════════════════════════════
TASK PROGRESS
═══════════════════════════════════════════════════════════════════════════════

COMPLETED SUBTASKS:
{completed_summary}

FAILED SUBTASKS:
{failed_summary}

PENDING SUBTASKS:
{pending_summary}

OBSERVATIONS LEARNED:
{observations_summary}

{f'''
═══════════════════════════════════════════════════════════════════════════════
FAILURE CONTEXT
═══════════════════════════════════════════════════════════════════════════════
{failure_context}
''' if failure_context else ''}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Analyze the situation and decide:
1. Should the remaining plan be modified? (Consider: Can failed tasks be approached differently? Are pending tasks still relevant?)
2. If yes, provide a revised list of subtasks to replace the pending ones.

GUIDELINES:
- If a task failed due to an element not found, consider adding a scroll or wait step first
- If a task failed due to navigation issues, consider breaking it into smaller steps
- If we learned something new (observations), use that knowledge in the revised plan
- Keep the plan MINIMAL - only essential steps
- DO NOT repeat already completed tasks

Respond with valid JSON:
{{
    "should_replan": true/false,
    "reasoning": "Brief explanation of your decision",
    "new_subtasks": [
        "Step 1 description",
        "Step 2 description"
    ]
}}

If should_replan is false, new_subtasks can be empty.
"""

        try:
            response = await self.llm.call_llm_direct(prompt)
            if not response:
                logger.warning("Replanning LLM returned no response")
                return False, []
            
            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                should_replan = data.get("should_replan", False)
                reasoning = data.get("reasoning", "")
                new_subtasks = data.get("new_subtasks", [])
                
                if should_replan and new_subtasks:
                    logger.info(f"📋 Replanning decision: YES")
                    logger.info(f"📋 Reasoning: {reasoning}")
                    logger.info(f"📋 New subtasks: {new_subtasks}")
                    return True, new_subtasks
                else:
                    logger.info(f"📋 Replanning decision: NO - {reasoning}")
                    return False, []
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse replanning response: {e}")
        except Exception as e:
            logger.error(f"Replanning failed: {e}")
        
        return False, []
    
    async def should_replan_after_failure(self, memory: AgentMemory, failed_action: str, error_msg: str) -> bool:
        """
        Quick check if we should trigger a full replan after a failure.
        
        Returns True if the failure pattern suggests replanning would help.
        """
        # Count consecutive failures on similar actions
        recent_failures = 0
        for entry in reversed(memory.history[-5:]):
            result = entry.get('result', {})
            if not result.get('success', True):
                recent_failures += 1
            else:
                break
        
        # Trigger replan if:
        # 1. Multiple consecutive failures (3+)
        # 2. Navigation or click failures (likely need different approach)
        if recent_failures >= 3:
            logger.info(f"📋 Triggering replan: {recent_failures} consecutive failures")
            return True
        
        # Check for specific failure patterns that benefit from replanning
        replan_triggers = [
            "not found",
            "timeout",
            "element not visible",
            "cannot scroll",
            "navigation failed"
        ]
        
        error_lower = error_msg.lower()
        for trigger in replan_triggers:
            if trigger in error_lower:
                logger.info(f"📋 Triggering replan: failure pattern '{trigger}'")
                return True
        
        return False


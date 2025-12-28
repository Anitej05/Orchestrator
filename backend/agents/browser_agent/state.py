"""
Browser Agent - State Management

Tracks the agent's memory, plan, and progress.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import time

class Subtask(BaseModel):
    """A single unit of work in the plan"""
    id: int
    description: str
    status: Literal["pending", "active", "completed", "failed"] = "pending"
    reasoning: Optional[str] = None
    result: Optional[str] = None

class AgentMemory(BaseModel):
    """The agent's long-term memory and state"""
    task: str
    plan: List[Subtask] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    observations: Dict[str, Any] = Field(default_factory=dict) # Key facts learned
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    extracted_items: List[Dict[str, Any]] = Field(default_factory=list)  # Accumulate multiple extractions
    
    def get_active_subtask(self) -> Optional[Subtask]:
        """Get the current active subtask"""
        for task in self.plan:
            if task.status == "active":
                return task
        # If none active, find first pending
        for task in self.plan:
            if task.status == "pending":
                task.status = "active"
                return task
        return None

    def mark_completed(self, subtask_id: int, result: str):
        """Mark a subtask as complete"""
        for task in self.plan:
            if task.id == subtask_id:
                task.status = "completed"
                task.result = result
                break
    
    def mark_failed(self, subtask_id: int, error: str):
        """Mark a subtask as failed"""
        for task in self.plan:
            if task.id == subtask_id:
                task.status = "failed"
                task.result = error
                break

    def add_observation(self, key: str, value: Any):
        """Remember a key fact"""
        self.observations[key] = value

    def update_plan(self, new_subtasks: List[str]):
        """Dynamically update the remaining plan"""
        # Find index of first pending/active task (keep completed/failed)
        start_idx = len(self.plan)
        for i, task in enumerate(self.plan):
            if task.status in ["pending", "active"]: # overwrite active too if we are replanning
                start_idx = i
                break
        
        # Keep old completed tasks
        kept_tasks = self.plan[:start_idx]
        
        # Create new tasks starting from last ID + 1
        last_id = kept_tasks[-1].id if kept_tasks else 0
        new_task_objs = []
        for i, desc in enumerate(new_subtasks):
            new_task_objs.append(Subtask(
                id=last_id + 1 + i,
                description=desc,
                status="pending"
            ))
            
        self.plan = kept_tasks + new_task_objs
        return len(new_task_objs)

    def to_prompt_context(self) -> str:
        """Format state for LLM prompt"""
        plan_str = "\n".join([
            f"{'[x]' if t.status == 'completed' else '[ ]'} {t.id}. {t.description} ({t.status})"
            for t in self.plan
        ])
        
        obs_str = "\n".join([f"- {k}: {v}" for k, v in self.observations.items()])
        
        return f"""
CURRENT PLAN:
{plan_str if plan_str else "No plan yet."}

KEY OBSERVATIONS:
{obs_str if obs_str else "None yet."}
"""

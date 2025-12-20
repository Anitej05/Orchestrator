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

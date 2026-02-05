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
    action_history: List[Dict[str, Any]] = Field(default_factory=list)  # Complete history of all actions taken
    
    # CMS Integration
    archived_blocks: List[str] = Field(default_factory=list) # IDs of archived history blocks
    active_content_id: Optional[str] = None # ID of the currently loaded large page in CMS
    active_content_summary: Optional[str] = None # Summary of the current large page
    
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

    def safe_add_extracted(self, data: Dict[str, Any]):
        """Add extracted data without overwriting existing keys.
        
        For structured_info, accumulates into a list to preserve all findings.
        """
        for key, value in data.items():
            if key == 'structured_info':
                # Accumulate structured info into a list
                if 'structured_items' not in self.extracted_data:
                    self.extracted_data['structured_items'] = []
                self.extracted_data['structured_items'].append(value)
            elif key not in self.extracted_data:
                # Only add if key doesn't exist (prevent overwriting)
                self.extracted_data[key] = value
            elif isinstance(self.extracted_data[key], list):
                # Append to existing list
                self.extracted_data[key].append(value)
        
        # Always append to extracted_items for traceability
        self.extracted_items.append(data)

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
        """Format state for LLM prompt - includes ALL saved info for stateful execution"""
        plan_str = "\n".join([
            f"{'[x]' if t.status == 'completed' else '[ ]'} {t.id}. {t.description} ({t.status})"
            for t in self.plan
        ])
        
        obs_str = "\n".join([f"- {k}: {v}" for k, v in self.observations.items()])
        
        # Include previously saved info so agent can reference it
        saved_info_str = ""
        if self.extracted_items:
            saved_items = []
            for item in self.extracted_items:
                if 'structured_info' in item:
                    s = item['structured_info']
                    verified = "✓" if s.get('verified', False) else "?"
                    saved_items.append(f"  [{verified}] {s.get('key', 'unknown')}: {s.get('value', '')}")
                elif item.get('fallback_capture'):
                    # Fallback captured data
                    s = item.get('structured_info', {})
                    patterns = s.get('extracted_patterns', {})
                    if patterns.get('prices_found'):
                        saved_items.append(f"  [auto] prices: {patterns['prices_found'][:3]}")
                    if patterns.get('potential_products'):
                        saved_items.append(f"  [auto] products: {patterns['potential_products'][:3]}")
            
            if saved_items:
                saved_info_str = "\n".join(saved_items)
        
        # Get complete action history
        action_history_str = self.format_action_history()
        
        return f"""
{action_history_str}

CURRENT PLAN:
{plan_str if plan_str else "No plan yet."}

KEY OBSERVATIONS:
{obs_str if obs_str else "None yet."}

PREVIOUSLY SAVED DATA (use this info - don't re-extract what you already have!):
{saved_info_str if saved_info_str else "Nothing saved yet. Use save_info when you find important data."}
"""

    def get_saved_summary(self) -> str:
        """Get a short summary of all saved data for quick reference"""
        if not self.extracted_items:
            return "No data saved yet"
        
        summary = []
        for item in self.extracted_items:
            if 'structured_info' in item:
                s = item['structured_info']
                summary.append(f"{s.get('key', '?')}: {str(s.get('value', ''))[:50]}")
        
        return "; ".join(summary) if summary else "No structured data"

    def add_action(self, step: int, url: str, title: str, goal: str, 
                   reasoning: str, action_type: str, target: str, 
                   result: str, error: str = None, stuck: bool = False,
                   mode: str = "text"):
        """Record an action in the complete history"""
        self.action_history.append({
            "step": step,
            "url": url[:100] if url else "",
            "title": title[:50] if title else "",
            "goal": goal[:100] if goal else "",
            "stuck": stuck,
            "mode": mode,
            "reasoning": reasoning[:200] if reasoning else "",
            "action_type": action_type,
            "target": target[:100] if target else "",
            "result": result[:100] if result else "",
            "error": error[:100] if error else None
        })
    
    def format_action_history(self) -> str:
        """Format complete action history for LLM prompt"""
        if not self.action_history:
            return "No actions taken yet."
        
        lines = []
        lines.append("=" * 60)
        lines.append("📜 COMPLETE ACTION HISTORY")
        lines.append("=" * 60)
        lines.append("")
        
        for entry in self.action_history:
            step = entry.get("step", "?")
            url = entry.get("url", "")
            title = entry.get("title", "")
            
            # Step header
            lines.append(f"Step {step} | {url[:50]}{'...' if len(url) > 50 else ''} | \"{title}\"")
            
            # Goal
            if entry.get("goal"):
                lines.append(f"  🎯 Goal: {entry['goal']}")
            
            # Stuck
            if entry.get("stuck"):
                lines.append(f"  ⚠️ Stuck: Yes")
            
            # Reasoning
            if entry.get("reasoning"):
                lines.append(f"  💭 Reasoning: {entry['reasoning']}")
            
            # Action
            action_type = entry.get("action_type", "unknown")
            target = entry.get("target", "")
            lines.append(f"  ⚡ Action: {action_type} → {target}")
            
            # Result
            result = entry.get("result", "")
            if "success" in result.lower() or result.startswith("✅"):
                lines.append(f"  ✅ Result: {result}")
            else:
                lines.append(f"  📋 Result: {result}")
            
            # Error
            if entry.get("error"):
                lines.append(f"  ❌ Error: {entry['error']}")
            
            lines.append("")  # Empty line between steps
        
        lines.append("=" * 60)
        return "\n".join(lines)


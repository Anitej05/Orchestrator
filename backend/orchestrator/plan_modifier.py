"""
Plan Modifier Module

Handles incremental plan updates instead of full plan recreation.
Provides functions to:
1. Add new tasks to existing plan
2. Update specific tasks in existing plan
3. Remove tasks from plan
4. Track all modifications with audit trail
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from schemas import (
    PlannedTask, 
    TaskAgentPair, 
    AgentCard,
    PlanModification,
    ExecutionStep,
)

logger = logging.getLogger(__name__)

class PlanModifier:
    """Handles incremental modifications to execution plans."""
    
    @staticmethod
    def add_task_to_plan(
        current_plan: List[List[PlannedTask]],
        task_agent_pairs: List[TaskAgentPair],
        new_task_description: str,
        suggested_agent: AgentCard,
        user_instruction: str,
        new_task_name: Optional[str] = None
    ) -> Tuple[List[List[PlannedTask]], List[TaskAgentPair], PlanModification]:
        """
        Adds a NEW task to the existing plan WITHOUT clearing previous tasks.
        
        Args:
            current_plan: Current execution plan (list of batches)
            task_agent_pairs: Current task-agent assignments
            new_task_description: Description of the new task
            suggested_agent: The agent to handle this task
            user_instruction: The user's actual request
            new_task_name: Optional custom name for the task
            
        Returns:
            Tuple of (updated_plan, updated_pairs, modification_record)
        """
        
        # Generate task name if not provided
        if not new_task_name:
            existing_task_names = [pair.task_name for pair in task_agent_pairs]
            base_name = "additional_task"
            counter = 1
            task_name = f"{base_name}_{counter}"
            while task_name in existing_task_names:
                counter += 1
                task_name = f"{base_name}_{counter}"
        else:
            task_name = new_task_name
        
        logger.info(f"Adding new task to plan: {task_name}")
        
        # Create new TaskAgentPair
        new_pair = TaskAgentPair(
            task_name=task_name,
            task_description=new_task_description,
            primary=suggested_agent,
            fallbacks=[]
        )
        
        # Add to task_agent_pairs (PRESERVE old ones)
        updated_pairs = task_agent_pairs + [new_pair]
        
        # Create PlannedTask for the plan
        import uuid
        planned_task = PlannedTask(
            task_name=task_name,
            task_description=new_task_description,
            primary=ExecutionStep(
                id=str(uuid.uuid4()),
                http_method=suggested_agent.endpoints[0].http_method if suggested_agent.endpoints else "POST",
                endpoint=suggested_agent.endpoints[0].endpoint if suggested_agent.endpoints else "",
                payload={}
            ),
            fallbacks=[]
        )
        
        # Add to plan (PRESERVE old batches, add new task to last batch or create new batch)
        updated_plan = [batch.copy() for batch in current_plan] if current_plan else []
        
        if updated_plan:
            # Add to last batch
            updated_plan[-1] = updated_plan[-1] + [planned_task]
        else:
            # Create first batch
            updated_plan = [[planned_task]]
        
        # Record modification
        modification = PlanModification(
            modification_type='add_task',
            affected_tasks=[task_name],
            reasoning=f"User requested to add: {user_instruction}",
            timestamp=datetime.now().isoformat(),
            user_instruction=user_instruction
        )
        
        logger.info(f"✅ Added new task '{task_name}' to plan. New plan has {len(updated_plan)} batches")
        
        return updated_plan, updated_pairs, modification
    
    @staticmethod
    def update_specific_task(
        current_plan: List[List[PlannedTask]],
        task_agent_pairs: List[TaskAgentPair],
        task_to_update: str,
        updated_description: str,
        update_reason: str,
        user_instruction: str
    ) -> Tuple[List[List[PlannedTask]], List[TaskAgentPair], Optional[PlanModification]]:
        """
        Updates a SPECIFIC existing task in the plan WITHOUT affecting others.
        
        Args:
            current_plan: Current execution plan
            task_agent_pairs: Current task-agent assignments
            task_to_update: Name of the task to update
            updated_description: New task description
            update_reason: Why the task is being updated
            user_instruction: The user's actual request
            
        Returns:
            Tuple of (updated_plan, updated_pairs, modification_record) or None if task not found
        """
        
        # Find the task in task_agent_pairs
        target_pair_idx = None
        for idx, pair in enumerate(task_agent_pairs):
            if pair.task_name == task_to_update:
                target_pair_idx = idx
                break
        
        if target_pair_idx is None:
            logger.warning(f"Task '{task_to_update}' not found in plan")
            return None
        
        logger.info(f"Updating task in plan: {task_to_update}")
        
        # Update in task_agent_pairs
        updated_pairs = task_agent_pairs.copy()
        old_description = updated_pairs[target_pair_idx].task_description
        updated_pairs[target_pair_idx].task_description = updated_description
        
        # Update in task_plan
        updated_plan = []
        for batch in current_plan:
            updated_batch = []
            for task in batch:
                if task.task_name == task_to_update:
                    # Update the task description
                    task.task_description = updated_description
                updated_batch.append(task)
            updated_plan.append(updated_batch)
        
        # Record modification
        modification = PlanModification(
            modification_type='update_task',
            affected_tasks=[task_to_update],
            reasoning=f"User requested: {update_reason}. Original: '{old_description}' → Updated: '{updated_description}'",
            timestamp=datetime.now().isoformat(),
            user_instruction=user_instruction
        )
        
        logger.info(f"✅ Updated task '{task_to_update}'. Reason: {update_reason}")
        
        return updated_plan, updated_pairs, modification
    
    @staticmethod
    def remove_task_from_plan(
        current_plan: List[List[PlannedTask]],
        task_agent_pairs: List[TaskAgentPair],
        task_to_remove: str,
        user_instruction: str
    ) -> Tuple[List[List[PlannedTask]], List[TaskAgentPair], Optional[PlanModification]]:
        """
        Removes a task from the plan.
        
        Args:
            current_plan: Current execution plan
            task_agent_pairs: Current task-agent assignments
            task_to_remove: Name of the task to remove
            user_instruction: The user's actual request
            
        Returns:
            Tuple of (updated_plan, updated_pairs, modification_record) or None if task not found
        """
        
        # Find and remove from task_agent_pairs
        updated_pairs = [pair for pair in task_agent_pairs if pair.task_name != task_to_remove]
        
        if len(updated_pairs) == len(task_agent_pairs):
            logger.warning(f"Task '{task_to_remove}' not found in pairs")
            return None
        
        logger.info(f"Removing task from plan: {task_to_remove}")
        
        # Remove from task_plan
        updated_plan = []
        for batch in current_plan:
            updated_batch = [task for task in batch if task.task_name != task_to_remove]
            if updated_batch:  # Only add non-empty batches
                updated_plan.append(updated_batch)
        
        # Record modification
        modification = PlanModification(
            modification_type='remove_task',
            affected_tasks=[task_to_remove],
            reasoning=f"User requested removal: {user_instruction}",
            timestamp=datetime.now().isoformat(),
            user_instruction=user_instruction
        )
        
        logger.info(f"✅ Removed task '{task_to_remove}' from plan")
        
        return updated_plan, updated_pairs, modification
    
    @staticmethod
    def build_context_summary(
        messages: List[Any],
        current_plan: List[List[PlannedTask]],
        task_agent_pairs: List[TaskAgentPair],
        completed_tasks: List[Dict[str, Any]],
        modifications: List[PlanModification]
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive context summary for orchestrator decisions.
        
        Returns dictionary with:
        - Conversation timeline
        - User preferences/patterns
        - Current plan state
        - Execution progress
        - Previous modifications
        """
        
        # Extract conversation summary (last 10 messages)
        recent_messages = []
        for msg in messages[-10:]:
            role = "user" if hasattr(msg, 'type') and msg.type == "human" else "assistant"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            recent_messages.append({"role": role, "content": content[:200]})  # Truncate for brevity
        
        # Current plan summary
        current_tasks = []
        for batch_idx, batch in enumerate(current_plan):
            for task in batch:
                current_tasks.append({
                    'task_name': task.task_name,
                    'description': task.task_description,
                    'batch': batch_idx
                })
        
        # Execution progress
        completed_task_names = [t.get('task_name', '') for t in completed_tasks]
        remaining_tasks = [t['task_name'] for t in current_tasks if t['task_name'] not in completed_task_names]
        
        # Modification timeline
        mod_timeline = [
            {
                'type': mod.modification_type,
                'tasks': mod.affected_tasks,
                'reason': mod.reasoning[:100],  # Truncate
                'time': mod.timestamp
            }
            for mod in modifications[-5:]  # Last 5 modifications
        ]
        
        return {
            'conversation_length': len(messages),
            'recent_messages': recent_messages,
            'current_plan': {
                'total_tasks': len(current_tasks),
                'batches': len(current_plan),
                'tasks': current_tasks
            },
            'execution_progress': {
                'completed': len(completed_task_names),
                'remaining': len(remaining_tasks),
                'remaining_tasks': remaining_tasks
            },
            'modification_history': mod_timeline
        }

def preserve_and_merge_plan(
    old_plan: List[List[PlannedTask]],
    new_task_agent_pairs: List[TaskAgentPair],
    modification_type: str
) -> List[List[PlannedTask]]:
    """
    Helper to preserve existing plan when merging new tasks.
    """
    
    if not old_plan or not new_task_agent_pairs:
        return old_plan
    
    # Get existing task names
    existing_task_names = set()
    for batch in old_plan:
        for task in batch:
            existing_task_names.add(task.task_name)
    
    # If this is a pure addition (not replacement), preserve old plan
    if modification_type == 'add_task':
        return old_plan
    
    # For updates, return the old plan - will be modified by caller
    return old_plan


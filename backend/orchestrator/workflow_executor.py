"""
Workflow Executor - Re-runs saved workflows with new inputs
"""
import uuid
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Executes a saved workflow by modifying the original prompt with new inputs
    and running it through the orchestrator.
    """
    
    def __init__(self, workflow_blueprint: Dict[str, Any]):
        """
        Initialize with workflow blueprint.
        
        Args:
            workflow_blueprint: The saved workflow structure containing:
                - original_prompt: The template prompt
                - task_agent_pairs: Saved agent assignments
                - state: Original state snapshot
        """
        self.blueprint = workflow_blueprint
        self.original_prompt = workflow_blueprint.get("original_prompt", "")
        self.task_agent_pairs = workflow_blueprint.get("task_agent_pairs", [])
        
    def build_prompt_with_inputs(self, inputs: Dict[str, Any]) -> str:
        """
        Replace placeholders in the original prompt with new input values.
        
        Supports:
        - {variable_name} - Simple replacement
        - {{variable_name}} - Double brace replacement
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Modified prompt string
        """
        modified_prompt = self.original_prompt
        
        # Replace both single and double brace patterns
        for key, value in inputs.items():
            # Single braces
            modified_prompt = modified_prompt.replace(f"{{{key}}}", str(value))
            # Double braces
            modified_prompt = modified_prompt.replace(f"{{{{{key}}}}}", str(value))
            
        logger.info(f"Built prompt with inputs: {list(inputs.keys())}")
        return modified_prompt
    
    def get_execution_config(self) -> Dict[str, Any]:
        """
        Get configuration for executing this workflow.
        
        Returns:
            Config dict with workflow metadata
        """
        return {
            "workflow_mode": True,
            "reuse_task_agent_pairs": self.task_agent_pairs,
            "skip_agent_search": bool(self.task_agent_pairs),
        }
    
    async def execute(self, inputs: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow with new inputs.
        
        This method prepares the inputs and config, but the actual orchestration
        happens via the /ws/chat or /ws/workflow/{id}/execute endpoints which
        stream results in real-time.
        
        Args:
            inputs: New input values for workflow execution
            owner: User/owner information
            
        Returns:
            Execution metadata (thread_id, prompt, config)
        """
        # Build modified prompt
        modified_prompt = self.build_prompt_with_inputs(inputs)
        
        # Generate new thread ID for this execution
        thread_id = str(uuid.uuid4())
        
        # Prepare execution config
        config = self.get_execution_config()
        
        logger.info(f"Prepared workflow execution - Thread: {thread_id}")
        
        return {
            "thread_id": thread_id,
            "prompt": modified_prompt,
            "config": config,
            "owner": owner,
            "inputs": inputs,
            "timestamp": datetime.utcnow().isoformat()
        }

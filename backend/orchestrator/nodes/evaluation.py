"""
Evaluation Nodes - Response evaluation and user interaction.

Contains nodes that evaluate agent responses and handle user interaction:
- evaluate_agent_response: Critically evaluates task results
- ask_user: Formats questions for user and prepares final response
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from orchestrator.state import State

logger = logging.getLogger("AgentOrchestrator")


class AgentResponseEvaluation(BaseModel):
    """Schema for evaluating an agent's response post-flight."""
    status: str = Field(
        ..., description="Either 'complete' or 'user_input_required'."
    )
    question: Optional[str] = Field(
        None, description="The clarifying question to ask the user if the result is vague."
    )


def evaluate_agent_response(state: State):
    """
    Critically evaluates the result of the last executed task to ensure it is
    logically correct and satisfies the user's intent before proceeding.
    
    Evaluation criteria:
    - Result completeness
    - Error detection
    - User intent alignment
    - Need for clarification
    
    Returns:
    - eval_status: 'complete', 'user_input_required', or 'failed'
    - replan_reason: If task failed and replanning is needed
    - pending_user_input: If user clarification is needed
    - question_for_user: The clarification question
    
    NOTE: Full implementation remains in graph.py due to LLM client dependencies.
    """
    raise NotImplementedError("Use graph.evaluate_agent_response - LLM implementation in main module")


def ask_user(state: State):
    """
    Formats the question for the user and prepares it as the final response.
    This is a terminal node that ends the graph's execution for the current run.
    
    Sets:
    - final_response: The formatted question
    - pending_user_input: True
    
    Used for:
    - Plan approval questions
    - Clarification requests
    - Parameter gathering
    
    NOTE: Full implementation remains in graph.py.
    """
    raise NotImplementedError("Use graph.ask_user - implementation in main module")


__all__ = [
    'AgentResponseEvaluation',
    'evaluate_agent_response',
    'ask_user',
]

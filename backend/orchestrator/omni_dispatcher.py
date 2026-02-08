"""
OMNI-DISPATCHER Integration

The Brain-Hands cycle that powers the orchestrator.

Architecture:
- Brain analyzes state and decides next action
- Hands executes the action and returns results
- Cycle continues until Brain decides to finish

This module provides the glue between Brain, Hands, and the existing graph.
"""

import logging
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig

from .brain import Brain
from .hands import Hands

logger = logging.getLogger(__name__)


brain = Brain()
hands = Hands()


async def omni_dispatch(
    state: Dict[str, Any], config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Main entry point for the OMNI-DISPATCHER system.

    Executes one iteration of the Brain-Hands cycle.
    Supports human-in-the-loop: pauses when approval is required.

    Args:
        state: Current orchestrator state
        config: RunnableConfig with thread_id, user_id, etc.

    Returns:
        Updated state after one Brain-Hands cycle
    """
    logger.info("Starting OMNI-DISPATCHER cycle")
    
    # === Check for pending approval (human-in-the-loop) ===
    if state.get("pending_approval"):
        logger.info("⏸️ PAUSED: Waiting for user approval")
        # Don't execute anything - return current state and wait for approval
        return state

    # Phase 1: Brain thinks and decides
    logger.debug("Phase 1: Brain thinking...")
    brain_updates = await brain.think(state, config)
    updated_state = {**state, **brain_updates}

    # Check if Brain decided to finish or skip
    decision = brain_updates.get("decision") or {}
    if decision.get("action_type") == "finish":
        logger.info("Brain decided to finish. Terminating cycle.")
        return updated_state
    if decision.get("action_type") == "skip":
        logger.info("Brain decided to skip. Continuing cycle.")
        return updated_state
    
    # === Check if this decision requires approval ===
    if updated_state.get("pending_approval"):
        logger.info("⏸️ ACTION REQUIRES APPROVAL: Pausing execution")
        return updated_state  # Don't execute, wait for approval

    # Phase 2: Hands executes the decision
    logger.debug("Phase 2: Hands executing...")
    execution_updates = await hands.execute(updated_state, config)
    final_state = {**updated_state, **execution_updates}

    # Handle skip in hands execution
    if (
        execution_updates.get("execution_result", {}).get("action_id") or ""
    ).startswith("skip"):
        logger.info("Hands skipped execution. Continuing cycle.")
        return final_state

    logger.info("OMNI-DISPATCHER cycle complete")

    return final_state


def should_continue(state: Dict[str, Any]) -> str:
    """
    Route decision: continue cycle, pause for approval, or finish?

    Returns:
        "continue" for another cycle
        "approval" when waiting for user approval
        "finish" to end
    """
    # Check for pending approval first
    if state.get("pending_approval"):
        return "approval"
    
    decision = state.get("decision") or {}

    if decision.get("action_type") == "finish":
        return "finish"

    if state.get("final_response"):
        return "finish"

    return "continue"


async def omni_brain_node(
    state: Dict[str, Any], config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Graph node wrapper for Brain.

    This can be directly used in LangGraph workflows.
    """
    return await brain.think(state, config)


async def omni_hands_node(
    state: Dict[str, Any], config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Graph node wrapper for Hands.

    This can be directly used in LangGraph workflows.
    """
    return await hands.execute(state, config)


def omni_route_condition(state: Dict[str, Any]) -> str:
    """
    Conditional routing for the OMNI-DISPATCHER graph.
    Routes: "hands" | "approval" | "finish" | "brain"
    """
    # Check for pending approval first
    if state.get("pending_approval"):
        return "approval"
    
    decision = state.get("decision") or {}
    action_type = decision.get("action_type", "")
    final_res = state.get("final_response")

    if action_type == "finish" or final_res:
        return "finish"
    
    # Skip action should loop back to Brain for another thinking cycle
    # This happens during initialization when todo_list is being set up
    if action_type == "skip":
        return "brain"

    return "hands"


def approve_pending_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approve a pending action - clears approval flags and prepares for execution.
    
    Call this when user approves the pending_decision.
    Returns state updates to apply.
    """
    if not state.get("pending_approval"):
        return {}
    
    pending_decision = state.get("pending_decision", {})
    logger.info(f"✅ User APPROVED action: {pending_decision.get('approval_reason', 'Unknown')}")
    
    return {
        "pending_approval": False,
        "pending_decision": None,
        # Re-apply decision but without requires_approval so it executes
        "decision": {
            **pending_decision,
            "requires_approval": False,
        }
    }


def reject_pending_action(state: Dict[str, Any], reason: str = "User rejected") -> Dict[str, Any]:
    """
    Reject a pending action - clears approval flags and sets skip decision.
    
    Call this when user rejects the pending_decision.
    Returns state updates to apply.
    """
    if not state.get("pending_approval"):
        return {}
    
    pending_decision = state.get("pending_decision", {})
    logger.info(f"❌ User REJECTED action: {pending_decision.get('approval_reason', 'Unknown')} - {reason}")
    
    return {
        "pending_approval": False,
        "pending_decision": None,
        "decision": {
            "action_type": "skip",
            "reasoning": f"User rejected: {reason}",
        }
    }


__all__ = [
    "brain",
    "hands",
    "omni_dispatch",
    "omni_brain_node",
    "omni_hands_node",
    "omni_route_condition",
    "should_continue",
    "approve_pending_action",
    "reject_pending_action",
]

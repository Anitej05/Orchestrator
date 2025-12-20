"""
Orchestrator Routing Functions

Conditional routing functions used by the LangGraph StateGraph.
These functions determine the next node to execute based on the current state.
"""

import logging
from orchestrator.state import State

logger = logging.getLogger("AgentOrchestrator")


def route_after_search(state: State):
    """Route after agent directory search based on whether agents were found."""
    if state.get("parsing_error_feedback"):
        if state.get("parse_retry_count", 0) >= 3:
            logger.warning("Max parse retries reached. Asking user for clarification.")
            return "ask_user"
        else:
            logger.info("Retrying parse_prompt.")
            return "parse_prompt"
    return "rank_agents"


def route_after_approval(state: State):
    """Routes after plan approval checkpoint."""
    approval_required = state.get("approval_required")
    pending_user_input = state.get("pending_user_input")
    
    logger.info(f"=== ROUTING AFTER APPROVAL: approval_required={approval_required}, pending_user_input={pending_user_input} ===")
    
    if approval_required and pending_user_input:
        logger.info("Routing to ask_user for plan approval.")
        return "ask_user"
    else:
        logger.info(f"Plan approved or no approval needed. Routing to validate_plan_for_execution.")
        return "validate_plan_for_execution"


def route_after_validation(state: State):
    """Simple routing after validation."""
    replan_reason = state.get("replan_reason")
    pending_user_input = state.get("pending_user_input")
    
    print(f"!!! ROUTER AFTER VALIDATION: replan={replan_reason}, pending={pending_user_input} !!!")
    
    if replan_reason:
        logger.info("Replan needed. Routing back to plan_execution.")
        return "plan_execution"
    if pending_user_input:
        logger.info("Pending user input. Routing to ask_user.")
        return "ask_user"
    else:
        logger.info("Plan is valid. Routing to execute_batch.")
        return "execute_batch"


def route_after_load_history(state: State):
    """Route after loading history: skip orchestration for pre-approved workflows."""
    plan_approved = state.get("plan_approved", False)
    has_plan = state.get("task_plan") and len(state.get("task_plan", [])) > 0
    needs_approval = state.get("needs_approval", False)
    
    print(f"!!! ROUTE AFTER LOAD HISTORY: plan_approved={plan_approved}, has_plan={has_plan}, needs_approval={needs_approval} !!!")
    
    if plan_approved and has_plan:
        print("!!! PRE-APPROVED WORKFLOW DETECTED = SKIP TO VALIDATION !!!")
        logger.info("Pre-approved workflow detected. Skipping orchestration and jumping to validation.")
        return "validate_plan_for_execution"
    
    print("!!! NO PRE-APPROVED PLAN = PROCEED TO ANALYSIS !!!")
    logger.info("No pre-approved plan. Proceeding to analyze_request.")
    return "analyze_request"


def route_after_analysis(state: State):
    """Route based on whether we have an existing plan or need to create one."""
    has_plan = state.get("task_plan") and len(state.get("task_plan", [])) > 0
    planning_mode = state.get("planning_mode", False)
    needs_complex = state.get("needs_complex_processing")
    plan_approved = state.get("plan_approved", False)
    uploaded_files_count = len(state.get("uploaded_files", []))
    
    logger.info(f"🔍 ROUTE_AFTER_ANALYSIS: uploaded_files count = {uploaded_files_count}")
    print(f"!!! ROUTE AFTER ANALYSIS: has_plan={has_plan}, planning_mode={planning_mode}, needs_complex={needs_complex}, plan_approved={plan_approved}, uploaded_files={uploaded_files_count} !!!")
    
    if has_plan and not planning_mode:
        print("!!! HAS PLAN + NO PLANNING MODE = SKIP TO VALIDATION !!!")
        logger.info("Plan exists and planning mode is off. Skipping to validation.")
        return "validate_plan_for_execution"
    
    if needs_complex:
        if state.get("uploaded_files"):
            return "preprocess_files"
        else:
            return "parse_prompt"
    else:
        return "generate_final_response"


def route_after_parse(state: State):
    """Route after parsing: handle direct responses, no tasks, or proceed to search."""
    # OPTIMIZATION: Short-circuit for direct responses (chitchat)
    if state.get('needs_complex_processing') is False and state.get('final_response'):
        logger.info("Direct response available. Short-circuiting to save_history.")
        return "save_history"
    
    if not state.get('parsed_tasks'):
        logger.warning("No tasks were parsed from prompt. Routing to ask_user for clarification.")
        return "ask_user"
    return "agent_directory_search"


def should_continue_or_finish(state: State):
    """REACTIVE ROUTER: Runs after execution and evaluation to decide the next step."""
    pending = state.get("pending_user_input")
    pending_confirmation = state.get("pending_confirmation")
    task_plan = state.get('task_plan')
    eval_status = state.get("eval_status")
    replan_reason = state.get("replan_reason")
    
    print(f"!!! SHOULD_CONTINUE_OR_FINISH: eval_status={eval_status}, pending={pending}, pending_confirmation={pending_confirmation}, replan_reason={replan_reason}, task_plan_length={len(task_plan) if task_plan else 0} !!!")
    logger.info(f"Reactive Router: eval_status={eval_status}, pending={pending}, pending_confirmation={pending_confirmation}, replan={bool(replan_reason)}, plan_length={len(task_plan) if task_plan else 0}")
    
    # CANVAS CONFIRMATION FIX: If confirmation is pending, go to generate_final_response to show canvas
    if pending_confirmation:
        print(f"!!! ROUTING TO GENERATE_FINAL_RESPONSE (pending_confirmation=True) !!!")
        logger.info("Routing to generate_final_response to display canvas with confirmation button")
        return "generate_final_response"
    
    # REACTIVE LOOP: If task failed, trigger auto-replan
    if eval_status == "failed" and replan_reason:
        print(f"!!! ROUTING TO PLAN_EXECUTION (auto-replan) !!!")
        logger.info("Task failed. Routing to plan_execution for auto-replan.")
        return "plan_execution"
    
    if pending:
        print(f"!!! ROUTING TO ASK_USER (pending_user_input=True) !!!")
        logger.info("Routing to ask_user due to pending_user_input")
        return "ask_user"
    
    if not task_plan:
        print(f"!!! ROUTING TO GENERATE_FINAL_RESPONSE (plan complete) !!!")
        logger.info("Execution plan is complete. Routing to generate_final_response.")
        return "generate_final_response"
    else:
        print(f"!!! ROUTING TO VALIDATE (more batches) !!!")
        logger.info("Plan has more batches. Routing back to validation for the next batch.")
        return "validate_plan_for_execution"


def route_after_plan_creation(state: State):
    """Stop for approval if planning mode, otherwise continue to execution."""
    planning_mode = state.get("planning_mode", False)
    plan_approved = state.get("plan_approved", False)
    
    print(f"!!! ROUTE AFTER PLAN: planning_mode={planning_mode}, plan_approved={plan_approved} !!!")
    logger.info(f"=== ROUTE AFTER PLAN: planning_mode={planning_mode}, plan_approved={plan_approved} ===")
    
    if planning_mode:
        print("!!! PLANNING MODE ON - PAUSING FOR APPROVAL !!!")
        logger.info("=== ROUTING: Planning mode ON. Pausing for approval ===")
        return "save_history"
    else:
        print("!!! CONTINUING TO VALIDATION !!!")
        logger.info("=== ROUTING: Planning mode OFF. Continuing to validation ===")
        return "validate_plan_for_execution"


def route_after_execute_batch(state: State):
    """Check if we're waiting for canvas confirmation, otherwise proceed to evaluation."""
    pending_confirmation = state.get("pending_confirmation", False)
    canvas_confirmation_action = state.get("canvas_confirmation_action")
    
    if canvas_confirmation_action:
        logger.info(f"🔄 Canvas confirmation received: {canvas_confirmation_action}. Routing to execute_confirmed_task.")
        return "execute_confirmed_task"
    
    if pending_confirmation:
        logger.info("⏸️ Waiting for canvas confirmation. Routing to generate_final_response to display canvas.")
        return "generate_final_response"
    
    return "evaluate_agent_response"


def route_after_ask_user(state: State):
    """Routes after ask_user based on context - approval vs clarification."""
    plan_approved = state.get("plan_approved")
    pending_user_input = state.get("pending_user_input")
    
    logger.info(f"=== ROUTING AFTER ASK_USER: plan_approved={plan_approved}, pending_user_input={pending_user_input} ===")
    
    if pending_user_input:
        logger.info("=== ROUTING: Workflow paused for user input. Routing to save_history (end) ===")
        return "save_history"
    
    if plan_approved:
        logger.info("=== ROUTING: Plan approved, continuing to validation (ONE TIME ONLY) ===")
        return "validate_plan_for_execution"
    
    logger.info("=== ROUTING: Default route to save_history (end) ===")
    return "save_history"

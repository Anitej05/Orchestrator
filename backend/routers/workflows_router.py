"""
Workflows Router - Handles workflow, schedule, and webhook endpoints.

Extracted from main.py to improve code organization and maintainability.
This is one of the larger routers due to the comprehensive workflow management features.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, Request, Body, Query
from sqlalchemy.orm import Session

from database import SessionLocal
from models import Workflow, WorkflowExecution, WorkflowSchedule, WorkflowWebhook, UserThread
from schemas import PlanResponse

router = APIRouter(tags=["Workflows"])
logger = logging.getLogger("uvicorn.error")

CONVERSATION_HISTORY_DIR = "conversation_history"


# --- Shared State (will be injected from main.py) ---
checkpointer = None


def set_checkpointer(cp):
    """Called by main.py to inject the shared checkpointer."""
    global checkpointer
    checkpointer = cp


# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Models ---
class ScheduleWorkflowRequest(BaseModel):
    """Model for scheduling workflow requests"""
    cron_expression: str
    input_template: Dict[str, Any] = {}


class UpdateScheduleRequest(BaseModel):
    """Model for updating schedule requests"""
    is_active: Optional[bool] = None
    cron_expression: Optional[str] = None
    input_template: Optional[Dict[str, Any]] = None


# --- Workflow CRUD ---
@router.post("/api/workflows")
async def save_workflow(request: Request, thread_id: str, name: str, description: str = "", db: Session = Depends(get_db)):
    """Save conversation as reusable workflow"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{thread_id}.json")
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    workflow_id = str(uuid.uuid4())
    
    task_agent_pairs = history.get("task_agent_pairs", [])
    original_prompt = history.get("original_prompt", "")
    
    if not original_prompt:
        messages = history.get("messages", [])
        if messages and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict) and first_msg.get("type") == "user":
                original_prompt = first_msg.get("content", "")
    
    if not task_agent_pairs or not original_prompt:
        if checkpointer:
            try:
                config = {"configurable": {"thread_id": thread_id}}
                checkpoint = checkpointer.get(config)
                if checkpoint:
                    checkpoint_state = checkpoint.get("values", {})
                    task_agent_pairs = checkpoint_state.get("task_agent_pairs", task_agent_pairs)
                    original_prompt = checkpoint_state.get("original_prompt", original_prompt)
                    history = {**history, **checkpoint_state}
            except Exception as e:
                logger.warning(f"Could not load from checkpointer: {e}")
    
    task_plan = history.get("task_plan", []) or history.get("plan", [])
    
    blueprint = {
        "workflow_id": workflow_id,
        "thread_id": thread_id,
        "original_prompt": original_prompt,
        "task_agent_pairs": task_agent_pairs,
        "task_plan": task_plan,
        "parsed_tasks": history.get("parsed_tasks", []),
        "candidate_agents": history.get("candidate_agents", {}),
        "user_expectations": history.get("user_expectations"),
        "completed_tasks": history.get("completed_tasks", []),
        "final_response": history.get("final_response"),
        "created_at": datetime.utcnow().isoformat()
    }
    
    plan_graph = None
    if task_plan:
        plan_graph = {"nodes": [], "edges": [], "tasks": task_plan}
    
    workflow = Workflow(
        workflow_id=workflow_id,
        user_id=user_id,
        name=name,
        description=description or blueprint.get("original_prompt", "")[:200],
        blueprint=blueprint,
        plan_graph=plan_graph
    )
    db.add(workflow)
    db.commit()
    
    logger.info(f"Workflow '{name}' ({workflow_id}) saved by user {user_id}")
    return {
        "workflow_id": workflow_id,
        "name": name,
        "description": workflow.description,
        "status": "saved",
        "task_count": len(blueprint.get("task_agent_pairs", [])),
        "created_at": blueprint["created_at"]
    }


@router.get("/api/workflows")
async def list_workflows(request: Request, db: Session = Depends(get_db)):
    """List user's workflows"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflows = db.query(Workflow).filter_by(user_id=user_id, status='active').all()
    
    result = []
    for w in workflows:
        blueprint = w.blueprint or {}
        task_count = len(blueprint.get("task_agent_pairs", []))
        
        estimated_cost = 0.0
        completed_tasks = blueprint.get("completed_tasks", [])
        for task in completed_tasks:
            if isinstance(task, dict):
                estimated_cost += task.get("cost", 0.0)
        
        active_schedules = db.query(WorkflowSchedule).filter_by(
            workflow_id=w.workflow_id, 
            is_active=True
        ).all()
        schedule_count = len(active_schedules)
        
        next_scheduled_run = None
        if active_schedules:
            next_schedule = min(active_schedules, key=lambda s: s.next_run_at or datetime.utcnow())
            if next_schedule.next_run_at:
                next_scheduled_run = next_schedule.next_run_at.isoformat()
        
        result.append({
            "workflow_id": w.workflow_id,
            "workflow_name": w.name,
            "workflow_description": w.description or "No description",
            "created_at": w.created_at.isoformat(),
            "updated_at": w.updated_at.isoformat() if w.updated_at else w.created_at.isoformat(),
            "task_count": task_count,
            "estimated_cost": estimated_cost,
            "has_plan_graph": w.plan_graph is not None,
            "is_public": False,
            "active_schedules": schedule_count,
            "next_scheduled_run": next_scheduled_run
        })
    
    return result


@router.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Get workflow details"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "workflow_id": workflow.workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "blueprint": workflow.blueprint,
        "plan_graph": workflow.plan_graph,
        "created_at": workflow.created_at.isoformat()
    }


@router.post("/api/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, request: Request, inputs: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    """Execute workflow by re-running the saved plan (no re-planning)"""
    from auth import get_user_from_request
    from langgraph.checkpoint.memory import MemorySaver
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    new_thread_id = str(uuid.uuid4())
    
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        user_id=user_id,
        inputs=inputs,
        status='running'
    )
    db.add(execution)
    db.commit()
    
    blueprint = workflow.blueprint
    task_plan = blueprint.get("task_plan", [])
    task_agent_pairs = blueprint.get("task_agent_pairs", [])
    original_prompt = blueprint.get("original_prompt", "")
    
    if not task_plan or not task_agent_pairs:
        raise HTTPException(status_code=400, detail="Workflow has no saved execution plan")
    
    memory = MemorySaver()
    
    initial_state = {
        "thread_id": new_thread_id,
        "original_prompt": original_prompt,
        "task_plan": task_plan,
        "task_agent_pairs": task_agent_pairs,
        "messages": [{"type": "user", "content": original_prompt}],
        "status": "executing",
        "planning_mode": False,
        "plan_approved": True,
        "completed_tasks": [],
        "task_events": []
    }
    
    config = {"configurable": {"thread_id": new_thread_id}}
    memory.put(config, {"values": initial_state, "next": ["execute_batch"]})
    
    logger.info(f"Pre-seeded workflow execution {workflow_id} as thread {new_thread_id}")
    return {
        "execution_id": execution_id, 
        "thread_id": new_thread_id, 
        "status": "running", 
        "task_count": len(task_agent_pairs),
        "message": "Connect to /ws/chat with this thread_id - execution will start automatically"
    }


@router.post("/api/workflows/{workflow_id}/create-conversation")
async def create_workflow_conversation(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Create a new conversation pre-seeded with the workflow's saved plan."""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub") or user.get("user_id") or user.get("id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Unable to determine user identity")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    new_thread_id = str(uuid.uuid4())
    blueprint = workflow.blueprint
    
    if isinstance(blueprint, str):
        try:
            blueprint = json.loads(blueprint)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid workflow blueprint format")
    
    task_plan = blueprint.get("task_plan", [])
    task_agent_pairs = blueprint.get("task_agent_pairs", [])
    original_prompt = blueprint.get("original_prompt", "")
    
    if not task_agent_pairs:
        raise HTTPException(status_code=400, detail="Workflow has no task agent pairs")
    
    user_thread = UserThread(
        thread_id=new_thread_id,
        user_id=user_id,
        title=f"From: {workflow.name}"
    )
    db.add(user_thread)
    db.commit()
    
    history_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{new_thread_id}.json")
    try:
        os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)
        conversation_json = {
            "thread_id": new_thread_id,
            "original_prompt": original_prompt,
            "task_agent_pairs": task_agent_pairs,
            "task_plan": task_plan,
            "messages": [],
            "completed_tasks": [],
            "final_response": None,
            "pending_user_input": True,
            "question_for_user": "Review the execution plan and approve to proceed.",
            "needs_approval": True,
            "approval_required": True,
            "plan_approved": False,
            "status": "planning_complete",
            "metadata": {"from_workflow": workflow_id, "workflow_name": workflow.name},
            "uploaded_files": []
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(conversation_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save conversation JSON file: {e}")
    
    return {
        "thread_id": new_thread_id,
        "workflow_id": workflow_id,
        "original_prompt": original_prompt,
        "task_agent_pairs": task_agent_pairs,
        "task_plan": task_plan,
        "task_count": len(task_agent_pairs),
        "status": "planning_complete",
        "message": "Plan loaded. Review and run from the input box."
    }


# --- Schedule Endpoints ---
@router.post("/api/workflows/{workflow_id}/schedule")
async def schedule_workflow(workflow_id: str, body: ScheduleWorkflowRequest, request: Request, db: Session = Depends(get_db)):
    """Schedule workflow execution with cron expression"""
    from auth import get_user_from_request
    from services.workflow_scheduler import get_scheduler
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if not workflow.blueprint:
        raise HTTPException(status_code=400, detail="Workflow has no blueprint data.")
    
    has_task_plan = workflow.blueprint.get("task_plan") and len(workflow.blueprint.get("task_plan", [])) > 0
    has_task_agent_pairs = workflow.blueprint.get("task_agent_pairs") and len(workflow.blueprint.get("task_agent_pairs", [])) > 0
    
    if not has_task_plan and not has_task_agent_pairs:
        raise HTTPException(status_code=400, detail="Workflow has no saved execution plan.")
    
    schedule_id = str(uuid.uuid4())
    schedule = WorkflowSchedule(
        schedule_id=schedule_id,
        workflow_id=workflow_id,
        user_id=user_id,
        cron_expression=body.cron_expression,
        input_template=body.input_template
    )
    db.add(schedule)
    db.commit()
    
    try:
        scheduler = get_scheduler()
        scheduler.add_schedule(
            schedule_id=schedule_id,
            workflow_id=workflow_id,
            cron_expression=body.cron_expression,
            input_template=body.input_template,
            user_id=user_id,
            db_session_factory=SessionLocal
        )
        return {"schedule_id": schedule_id, "status": "scheduled", "cron": body.cron_expression}
    except Exception as e:
        db.delete(schedule)
        db.commit()
        raise HTTPException(status_code=400, detail=f"Invalid cron expression: {str(e)}")


@router.get("/api/schedules")
async def list_schedules(request: Request, db: Session = Depends(get_db)):
    """List all workflow schedules for the authenticated user"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedules = db.query(WorkflowSchedule).filter_by(user_id=user_id).order_by(WorkflowSchedule.created_at.desc()).all()
    
    result = []
    for schedule in schedules:
        workflow = db.query(Workflow).filter_by(workflow_id=schedule.workflow_id).first()
        
        next_run = None
        if schedule.is_active:
            try:
                from apscheduler.triggers.cron import CronTrigger
                from datetime import timezone
                parts = schedule.cron_expression.split()
                if len(parts) == 5:
                    minute, hour, day, month, day_of_week = parts
                    trigger = CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week, timezone='UTC')
                    now = datetime.now(timezone.utc)
                    next_run = trigger.get_next_fire_time(None, now)
                    if next_run:
                        next_run = next_run.isoformat()
            except Exception as e:
                logger.error(f"Failed to calculate next run time: {str(e)}")
        
        result.append({
            "schedule_id": schedule.schedule_id,
            "workflow_id": schedule.workflow_id,
            "workflow_name": workflow.name if workflow else "Unknown",
            "cron_expression": schedule.cron_expression,
            "input_template": schedule.input_template,
            "is_active": schedule.is_active,
            "last_run_at": schedule.last_run_at.isoformat() if schedule.last_run_at else None,
            "next_run_at": next_run,
            "created_at": schedule.created_at.isoformat()
        })
    
    return {"schedules": result, "count": len(result)}


@router.get("/api/schedules/{schedule_id}/executions")
async def get_schedule_executions(schedule_id: str, request: Request, db: Session = Depends(get_db)):
    """Get execution history for a specific schedule"""
    from auth import get_user_from_request
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(schedule_id=schedule_id, user_id=user_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    executions = db.query(WorkflowExecution).filter_by(
        workflow_id=schedule.workflow_id,
        user_id=user_id
    ).order_by(WorkflowExecution.started_at.desc()).limit(50).all()
    
    result = []
    for execution in executions:
        result.append({
            "execution_id": execution.execution_id,
            "status": execution.status,
            "inputs": execution.inputs,
            "outputs": execution.outputs,
            "error": execution.error,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration_ms": int((execution.completed_at - execution.started_at).total_seconds() * 1000) if execution.completed_at and execution.started_at else None
        })
    
    return {"executions": result, "count": len(result)}


@router.patch("/api/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, body: UpdateScheduleRequest, request: Request, db: Session = Depends(get_db)):
    """Update a workflow schedule (pause/resume, change cron, update inputs)"""
    from auth import get_user_from_request
    from services.workflow_scheduler import get_scheduler
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(schedule_id=schedule_id, user_id=user_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    scheduler = get_scheduler()
    
    if body.is_active is not None:
        schedule.is_active = body.is_active
        if body.is_active:
            try:
                scheduler.add_schedule(
                    schedule_id=schedule.schedule_id,
                    workflow_id=schedule.workflow_id,
                    cron_expression=schedule.cron_expression,
                    input_template=schedule.input_template or {},
                    user_id=schedule.user_id,
                    db_session_factory=SessionLocal
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to resume schedule: {str(e)}")
        else:
            try:
                scheduler.remove_schedule(schedule_id)
            except Exception as e:
                logger.error(f"Failed to pause schedule: {str(e)}")
    
    if body.cron_expression is not None:
        schedule.cron_expression = body.cron_expression
        if schedule.is_active:
            try:
                scheduler.add_schedule(
                    schedule_id=schedule.schedule_id,
                    workflow_id=schedule.workflow_id,
                    cron_expression=body.cron_expression,
                    input_template=schedule.input_template or {},
                    user_id=schedule.user_id,
                    db_session_factory=SessionLocal
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid cron expression: {str(e)}")
    
    if body.input_template is not None:
        schedule.input_template = body.input_template
    
    db.commit()
    
    return {
        "schedule_id": schedule.schedule_id,
        "is_active": schedule.is_active,
        "cron_expression": schedule.cron_expression,
        "input_template": schedule.input_template,
        "status": "updated"
    }


@router.post("/api/admin/reload-schedules", tags=["Admin"])
async def reload_schedules(db: Session = Depends(get_db)):
    """Reload all active schedules from database into the scheduler."""
    from services.workflow_scheduler import get_scheduler
    scheduler = get_scheduler()
    
    try:
        scheduler.load_active_schedules(db)
        jobs = scheduler.scheduler.get_jobs()
        job_details = [{"job_id": job.id, "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None} for job in jobs]
        return {"status": "success", "jobs_loaded": len(jobs), "jobs": job_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload schedules: {str(e)}")


@router.delete("/api/workflows/{workflow_id}/schedule/{schedule_id}")
async def delete_schedule(workflow_id: str, schedule_id: str, request: Request, db: Session = Depends(get_db)):
    """Delete a workflow schedule"""
    from auth import get_user_from_request
    from services.workflow_scheduler import get_scheduler
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    schedule = db.query(WorkflowSchedule).filter_by(schedule_id=schedule_id, workflow_id=workflow_id, user_id=user_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    try:
        scheduler = get_scheduler()
        scheduler.remove_schedule(schedule_id)
    except Exception as e:
        logger.error(f"Failed to remove schedule from scheduler: {str(e)}")
    
    db.delete(schedule)
    db.commit()
    
    return {"status": "deleted"}


# --- Webhook Endpoints ---
@router.post("/api/workflows/{workflow_id}/webhook")
async def create_webhook(workflow_id: str, request: Request, db: Session = Depends(get_db)):
    """Create webhook trigger for workflow"""
    from auth import get_user_from_request
    import secrets
    
    user = get_user_from_request(request)
    user_id = user.get("sub")
    
    workflow = db.query(Workflow).filter_by(workflow_id=workflow_id, user_id=user_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    webhook_id = str(uuid.uuid4())
    webhook_token = secrets.token_urlsafe(32)
    
    webhook = WorkflowWebhook(
        webhook_id=webhook_id,
        workflow_id=workflow_id,
        user_id=user_id,
        webhook_token=webhook_token
    )
    db.add(webhook)
    db.commit()
    
    return {"webhook_id": webhook_id, "webhook_url": f"/webhooks/{webhook_id}", "webhook_token": webhook_token}


@router.post("/webhooks/{webhook_id}")
async def trigger_webhook(webhook_id: str, payload: Dict[str, Any], webhook_token: str = Query(...), db: Session = Depends(get_db)):
    """Trigger workflow via webhook - executes asynchronously"""
    webhook = db.query(WorkflowWebhook).filter_by(webhook_id=webhook_id, is_active=True).first()
    
    if not webhook or webhook.webhook_token != webhook_token:
        raise HTTPException(status_code=404, detail="Invalid webhook")
    
    workflow = db.query(Workflow).filter_by(workflow_id=webhook.workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    execution_id = str(uuid.uuid4())
    execution = WorkflowExecution(
        execution_id=execution_id,
        workflow_id=webhook.workflow_id,
        user_id=webhook.user_id,
        inputs=payload,
        status='queued',
        started_at=datetime.utcnow()
    )
    db.add(execution)
    db.commit()
    
    from services.workflow_scheduler import get_scheduler
    scheduler = get_scheduler()
    asyncio.create_task(
        scheduler._async_execute_workflow(
            execution_id, workflow.workflow_id, workflow.blueprint, payload, webhook.user_id
        )
    )
    
    return {"execution_id": execution_id, "status": "running", "message": "Workflow execution started"}


# --- Plan Endpoint ---
@router.get("/api/plan/{thread_id}", response_model=PlanResponse)
async def get_agent_plan(thread_id: str):
    """Retrieves the markdown execution plan for a given conversation thread."""
    plan_dirs = ["agent_plans"]
    file_path = None
    
    for plan_dir in plan_dirs:
        temp_path = os.path.join(plan_dir, f"{thread_id}-plan.md")
        if os.path.exists(temp_path):
            file_path = temp_path
            break

    if not file_path:
        raise HTTPException(status_code=404, detail=f"Plan file not found for thread_id: {thread_id}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return PlanResponse(thread_id=thread_id, content=content)
    except Exception as e:
        logger.error(f"Error reading plan file for thread_id {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Error reading the plan file.")

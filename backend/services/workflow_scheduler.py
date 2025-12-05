"""
Workflow Scheduler Service
Manages automated execution of workflows using APScheduler
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError
import asyncio
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class WorkflowScheduler:
    """Manages scheduled workflow executions using APScheduler"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone='UTC')
        self.scheduler.start()
        logger.info("Workflow scheduler initialized")
    
    def add_schedule(self, schedule_id: str, workflow_id: str, cron_expression: str, 
                     input_template: Dict, user_id: str, db_session_factory):
        """Add a scheduled workflow to the scheduler"""
        try:
            # Parse cron expression (format: "minute hour day month day_of_week")
            parts = cron_expression.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid cron expression: {cron_expression}. Expected 5 parts.")
            
            minute, hour, day, month, day_of_week = parts
            
            # Create cron trigger
            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                timezone='UTC'
            )
            
            # Add job to scheduler
            self.scheduler.add_job(
                func=self._execute_scheduled_workflow,
                trigger=trigger,
                id=schedule_id,
                args=[schedule_id, workflow_id, input_template, user_id, db_session_factory],
                replace_existing=True,
                name=f"Workflow {workflow_id} Schedule"
            )
            
            logger.info(f"Added schedule {schedule_id} for workflow {workflow_id} with cron: {cron_expression}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add schedule {schedule_id}: {str(e)}")
            raise
    
    def remove_schedule(self, schedule_id: str):
        """Remove a scheduled workflow"""
        try:
            self.scheduler.remove_job(schedule_id)
            logger.info(f"Removed schedule {schedule_id}")
            return True
        except JobLookupError:
            logger.warning(f"Schedule {schedule_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to remove schedule {schedule_id}: {str(e)}")
            raise
    
    def _execute_scheduled_workflow(self, schedule_id: str, workflow_id: str, 
                                    input_template: Dict, user_id: str, db_session_factory):
        """Execute a workflow as part of a scheduled job (runs in thread pool)"""
        try:
            logger.info(f"Executing scheduled workflow {workflow_id} (schedule: {schedule_id})")
            
            # Create database session
            db = db_session_factory()
            
            try:
                from models import Workflow, WorkflowExecution, WorkflowSchedule
                import uuid
                
                # Get workflow
                workflow = db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
                if not workflow:
                    logger.error(f"Workflow {workflow_id} not found")
                    return
                
                # Create execution record
                execution_id = str(uuid.uuid4())
                execution = WorkflowExecution(
                    execution_id=execution_id,
                    workflow_id=workflow_id,
                    user_id=user_id,
                    status='queued',
                    inputs=input_template,
                    started_at=datetime.utcnow()
                )
                db.add(execution)
                
                # Update schedule last_run_at
                schedule = db.query(WorkflowSchedule).filter(
                    WorkflowSchedule.schedule_id == schedule_id
                ).first()
                if schedule:
                    schedule.last_run_at = datetime.utcnow()
                
                db.commit()
                
                # Execute workflow synchronously (run async code in new event loop)
                import asyncio
                try:
                    # Create a new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        self._async_execute_workflow(
                            execution_id, workflow_id, workflow.blueprint, input_template, user_id
                        )
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Error running async execution: {str(e)}", exc_info=True)
                
                logger.info(f"Completed execution {execution_id} for scheduled workflow {workflow_id}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to execute scheduled workflow {workflow_id}: {str(e)}", exc_info=True)
    
    async def _async_execute_workflow(self, execution_id: str, workflow_id: str, 
                                      blueprint: Dict, inputs: Dict, user_id: str):
        """Actually execute the workflow asynchronously using the orchestrator"""
        try:
            from database import SessionLocal
            from models import WorkflowExecution, WorkflowSchedule, UserThread
            from orchestrator.graph import graph
            import uuid
            import os
            import json
            
            db = SessionLocal()
            
            try:
                # Update status to running
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.execution_id == execution_id
                ).first()
                if execution:
                    execution.status = 'running'
                    execution.started_at = datetime.utcnow()
                    db.commit()
                
                # Get the saved plan from blueprint
                task_plan = blueprint.get("task_plan", [])
                task_agent_pairs = blueprint.get("task_agent_pairs", [])
                original_prompt = blueprint.get("original_prompt", "")
                
                if not task_agent_pairs:
                    raise Exception("Workflow has no saved task agent pairs")
                
                # Check if schedule already has a conversation thread
                schedule = db.query(WorkflowSchedule).filter(
                    WorkflowSchedule.workflow_id == workflow_id,
                    WorkflowSchedule.user_id == user_id,
                    WorkflowSchedule.is_active == True
                ).first()
                
                thread_id = None
                if schedule and schedule.conversation_thread_id:
                    # Reuse existing conversation thread
                    thread_id = schedule.conversation_thread_id
                    logger.info(f"Reusing conversation thread {thread_id} for scheduled workflow {workflow_id}")
                else:
                    # Create a new conversation thread for this schedule
                    thread_id = str(uuid.uuid4())
                    
                    # Get workflow name for conversation title
                    from models import Workflow
                    workflow = db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
                    workflow_name = workflow.name if workflow else "Scheduled Workflow"
                    
                    # Create UserThread record
                    user_thread = UserThread(
                        thread_id=thread_id,
                        user_id=user_id,
                        title=f"Scheduled: {workflow_name}"
                    )
                    db.add(user_thread)
                    
                    # Store thread_id in schedule for future runs
                    if schedule:
                        schedule.conversation_thread_id = thread_id
                    
                    db.commit()
                    logger.info(f"Created new conversation thread {thread_id} for scheduled workflow {workflow_id}")
                
                # Save/update conversation JSON file
                conversation_history_dir = "conversation_history"
                os.makedirs(conversation_history_dir, exist_ok=True)
                history_path = os.path.join(conversation_history_dir, f"{thread_id}.json")
                
                # Load existing conversation if it exists (to preserve message history)
                existing_messages = []
                if os.path.exists(history_path):
                    try:
                        with open(history_path, "r", encoding="utf-8") as f:
                            existing_conv = json.load(f)
                            existing_messages = existing_conv.get("messages", [])
                    except Exception as e:
                        logger.warning(f"Could not load existing conversation: {e}")
                
                # Add execution start message
                execution_start_message = {
                    "type": "system",
                    "content": f"Scheduled execution started at {datetime.utcnow().isoformat()}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                existing_messages.append(execution_start_message)
                
                # Pre-seed conversation with plan_approved=True for automatic execution
                conversation_json = {
                    "thread_id": thread_id,
                    "original_prompt": original_prompt,
                    "task_agent_pairs": task_agent_pairs,
                    "task_plan": task_plan,
                    "messages": existing_messages,
                    "completed_tasks": [],
                    "final_response": None,
                    "pending_user_input": False,
                    "needs_approval": False,
                    "plan_approved": True,  # Auto-approve for scheduled execution
                    "status": "executing",
                    "metadata": {
                        "from_workflow": workflow_id,
                        "execution_id": execution_id,
                        "scheduled": True,
                        "last_execution": datetime.utcnow().isoformat()
                    },
                    "uploaded_files": []
                }
                
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(conversation_json, f, ensure_ascii=False, indent=2)
                
                # Execute the workflow through the orchestrator graph
                initial_state = {
                    "thread_id": thread_id,
                    "original_prompt": original_prompt,
                    "task_plan": task_plan,
                    "task_agent_pairs": task_agent_pairs,
                    "user_inputs": inputs,
                    "messages": existing_messages,
                    "status": "executing",
                    "planning_mode": False,
                    "plan_approved": True,  # Skip approval - execute directly
                    "needs_approval": False,
                    "completed_tasks": [],
                    "task_events": [],
                    "owner": {"user_id": user_id}
                }
                
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": str(uuid.uuid4())
                    }
                }
                
                # Execute the workflow through the orchestrator graph
                final_state = None
                completed_tasks = []
                
                async for event in graph.astream(
                    initial_state,
                    config,
                    stream_mode="values"
                ):
                    final_state = event
                    
                    # Collect completed tasks
                    if event.get("completed_tasks"):
                        completed_tasks = event.get("completed_tasks", [])
                    
                    # Log progress
                    if event.get("status"):
                        logger.info(f"Execution {execution_id} status: {event.get('status')}")
                
                # Extract results from final state
                final_response = None
                if final_state:
                    messages = final_state.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        if isinstance(last_message, dict):
                            final_response = last_message.get("content", "")
                        else:
                            final_response = str(last_message)
                    
                    if not final_response:
                        final_response = final_state.get("final_response", "")
                    
                    # Update completed tasks
                    completed_tasks = final_state.get("completed_tasks", completed_tasks)
                
                # Update conversation JSON with final results
                # Use get_serializable_state to properly serialize messages and other complex objects
                from orchestrator.graph import get_serializable_state
                
                # Get messages from final state or use existing
                raw_messages = final_state.get("messages", existing_messages) if final_state else existing_messages
                
                # Create a minimal state dict for serialization
                temp_state = {
                    "messages": raw_messages,
                    "completed_tasks": completed_tasks,
                    "final_response": final_response or "Workflow completed",
                    "task_agent_pairs": final_state.get("task_agent_pairs", []) if final_state else [],
                    "task_plan": final_state.get("task_plan", []) if final_state else [],
                    "original_prompt": conversation_json.get("metadata", {}).get("original_prompt", ""),
                }
                
                # Serialize the state properly
                serialized = get_serializable_state(temp_state, thread_id)
                
                # Update conversation JSON with serialized data
                conversation_json["messages"] = serialized.get("messages", [])
                conversation_json["completed_tasks"] = serialized.get("metadata", {}).get("completed_tasks", completed_tasks)
                conversation_json["final_response"] = final_response or "Workflow completed"
                conversation_json["status"] = "completed"
                conversation_json["metadata"]["completed_at"] = datetime.utcnow().isoformat()
                
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(conversation_json, f, ensure_ascii=False, indent=2)
                
                # Update execution with results
                if execution:
                    execution.status = 'completed'
                    execution.outputs = {
                        "response": final_response or "Workflow completed",
                        "completed_tasks": completed_tasks,
                        "thread_id": thread_id
                    }
                    execution.completed_at = datetime.utcnow()
                    db.commit()
                
                logger.info(f"Scheduled workflow execution {execution_id} completed successfully. Results saved to thread {thread_id}")
                
            except Exception as e:
                # Update execution with error
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.execution_id == execution_id
                ).first()
                if execution:
                    execution.status = 'failed'
                    execution.error = str(e)
                    execution.completed_at = datetime.utcnow()
                    db.commit()
                
                logger.error(f"Scheduled workflow execution {execution_id} failed: {str(e)}", exc_info=True)
                raise
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Critical error in async workflow execution {execution_id}: {str(e)}", exc_info=True)
    
    def load_active_schedules(self, db: Session):
        """Load all active schedules from database on startup"""
        try:
            from models import WorkflowSchedule
            
            active_schedules = db.query(WorkflowSchedule).filter(
                WorkflowSchedule.is_active == True
            ).all()
            
            from database import SessionLocal
            
            for schedule in active_schedules:
                try:
                    self.add_schedule(
                        schedule_id=schedule.schedule_id,
                        workflow_id=schedule.workflow_id,
                        cron_expression=schedule.cron_expression,
                        input_template=schedule.input_template or {},
                        user_id=schedule.user_id,
                        db_session_factory=SessionLocal
                    )
                except Exception as e:
                    logger.error(f"Failed to load schedule {schedule.schedule_id}: {str(e)}")
            
            logger.info(f"Loaded {len(active_schedules)} active schedules")
            
        except Exception as e:
            logger.error(f"Failed to load active schedules: {str(e)}")
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown(wait=True)
        logger.info("Workflow scheduler shut down")

# Global scheduler instance
_scheduler: Optional[WorkflowScheduler] = None

def get_scheduler() -> WorkflowScheduler:
    """Get or create the global scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = WorkflowScheduler()
    return _scheduler

def init_scheduler(db: Session):
    """Initialize scheduler and load active schedules"""
    scheduler = get_scheduler()
    scheduler.load_active_schedules(db)
    return scheduler

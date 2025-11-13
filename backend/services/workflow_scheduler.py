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
                
                # Execute workflow asynchronously (queue it for background execution)
                asyncio.create_task(self._async_execute_workflow(
                    execution_id, workflow_id, workflow.blueprint, input_template, user_id
                ))
                
                logger.info(f"Queued execution {execution_id} for scheduled workflow {workflow_id}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to execute scheduled workflow {workflow_id}: {str(e)}", exc_info=True)
    
    async def _async_execute_workflow(self, execution_id: str, workflow_id: str, 
                                      blueprint: Dict, inputs: Dict, user_id: str):
        """Actually execute the workflow asynchronously"""
        try:
            from orchestrator.workflow_executor import WorkflowExecutor
            from orchestrator.graph import graph
            from database import SessionLocal
            from models import WorkflowExecution
            
            db = SessionLocal()
            
            try:
                # Update status to running
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.execution_id == execution_id
                ).first()
                if execution:
                    execution.status = 'running'
                    db.commit()
                
                # Execute workflow
                executor = WorkflowExecutor(blueprint)
                owner = {"user_id": user_id}
                execution_metadata = await executor.execute(inputs, owner)
                
                # Stream through orchestrator (collect results)
                final_output = ""
                async for event in graph.astream_events(
                    {"messages": [("user", execution_metadata["prompt"])]},
                    execution_metadata["config"],
                    version="v2"
                ):
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        content = event.get("data", {}).get("chunk", {}).content
                        if content:
                            final_output += content
                
                # Update execution with results
                if execution:
                    execution.status = 'completed'
                    execution.outputs = {"response": final_output}
                    execution.completed_at = datetime.utcnow()
                    db.commit()
                
                logger.info(f"Scheduled workflow execution {execution_id} completed successfully")
                
            except Exception as e:
                # Update execution with error
                if execution:
                    execution.status = 'failed'
                    execution.error = str(e)
                    execution.completed_at = datetime.utcnow()
                    db.commit()
                
                logger.error(f"Scheduled workflow execution {execution_id} failed: {str(e)}")
                
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

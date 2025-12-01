from database import SessionLocal
from models import WorkflowExecution, WorkflowSchedule

db = SessionLocal()

# Check schedules
schedules = db.query(WorkflowSchedule).all()
print(f'Total schedules: {len(schedules)}')
for s in schedules:
    print(f'Schedule {s.schedule_id[:8]}: workflow={s.workflow_id[:8]}, cron={s.cron_expression}, active={s.is_active}, last_run={s.last_run_at}')

# Check recent executions
executions = db.query(WorkflowExecution).order_by(WorkflowExecution.started_at.desc()).limit(10).all()
print(f'\nRecent executions: {len(executions)}')
for e in executions:
    print(f'Execution {e.execution_id[:8]}: workflow={e.workflow_id[:8] if e.workflow_id else "None"}, status={e.status}, started={e.started_at}')

db.close()

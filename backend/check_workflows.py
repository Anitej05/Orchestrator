from database import SessionLocal
from models import Workflow

db = SessionLocal()
workflows = db.query(Workflow).all()
print(f'Total workflows: {len(workflows)}')
for w in workflows[:5]:
    task_count = len(w.blueprint.get("task_agent_pairs", []))
    print(f'ID: {w.workflow_id}, Name: {w.name}, Tasks: {task_count}')
db.close()

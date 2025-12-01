from services.workflow_scheduler import get_scheduler
from database import SessionLocal

try:
    scheduler = get_scheduler()
    
    # Check if scheduler is running
    print(f"Scheduler running: {scheduler.scheduler.running}")
    print(f"Scheduler state: {scheduler.scheduler.state}")
    
    # List all jobs
    jobs = scheduler.scheduler.get_jobs()
    print(f"\nTotal jobs in scheduler: {len(jobs)}")
    for job in jobs:
        print(f"Job ID: {job.id}")
        print(f"  Name: {job.name}")
        print(f"  Next run: {job.next_run_time}")
        print(f"  Trigger: {job.trigger}")
        print()
    
    # Try to load schedules
    db = SessionLocal()
    print("Loading active schedules...")
    scheduler.load_active_schedules(db)
    db.close()
    
    # Check jobs again
    jobs = scheduler.scheduler.get_jobs()
    print(f"\nTotal jobs after loading: {len(jobs)}")
    for job in jobs:
        print(f"Job ID: {job.id}")
        print(f"  Next run: {job.next_run_time}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

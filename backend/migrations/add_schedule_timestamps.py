"""
Migration: Add last_run_at and next_run_at to workflow_schedules
"""
from sqlalchemy import text
from database import SessionLocal, engine

def upgrade():
    """Add new columns to workflow_schedules table"""
    db = SessionLocal()
    try:
        # Add last_run_at column
        db.execute(text("""
            ALTER TABLE workflow_schedules 
            ADD COLUMN IF NOT EXISTS last_run_at TIMESTAMP
        """))
        
        # Add next_run_at column
        db.execute(text("""
            ALTER TABLE workflow_schedules 
            ADD COLUMN IF NOT EXISTS next_run_at TIMESTAMP
        """))
        
        db.commit()
        print("✅ Migration completed: Added last_run_at and next_run_at columns")
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    upgrade()

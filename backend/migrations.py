"""
Database migration system for Agent Directory.

This file contains migration functions that can be executed to update the database schema.
Each migration should be idempotent (safe to run multiple times).
"""

from sqlalchemy import text, inspect
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class Migration:
    """Base class for database migrations."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def should_run(self, inspector, conn) -> bool:
        """Check if this migration should be executed."""
        raise NotImplementedError
    
    def execute(self, conn):
        """Execute the migration."""
        raise NotImplementedError

class AddRatingCountMigration(Migration):
    """Migration to add rating_count column to agents table."""
    
    def __init__(self):
        super().__init__(
            name="add_rating_count",
            description="Add rating_count column to agents table"
        )
    
    def should_run(self, inspector, conn) -> bool:
        if not inspector.has_table("agents"):
            return False
        
        columns = inspector.get_columns("agents")
        column_names = [col['name'] for col in columns]
        return 'rating_count' not in column_names
    
    def execute(self, conn):
        conn.execute(text("ALTER TABLE agents ADD COLUMN rating_count INTEGER DEFAULT 0 NOT NULL"))
        conn.commit()

# List of all migrations in order
MIGRATIONS = [
    AddRatingCountMigration(),
    # Add new migrations here as they are created
]

def run_all_migrations(engine):
    """
    Run all pending migrations.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        for migration in MIGRATIONS:
            if migration.should_run(inspector, conn):
                print(f"🔄 Running migration: {migration.description}")
                try:
                    migration.execute(conn)
                    print(f"✅ Completed migration: {migration.name}")
                except ProgrammingError as e:
                    print(f"⚠️ Warning: Migration '{migration.name}' failed: {e}")
                    logger.warning(f"Migration {migration.name} failed: {e}")
            else:
                print(f"✅ Migration already applied: {migration.name}")

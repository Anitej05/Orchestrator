"""Database cleanup: drop stale columns/tables and stamp migration."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from database import engine
from sqlalchemy import text, inspect

inspector = inspect(engine)
existing_tables = inspector.get_table_names()

def col_exists(table, col):
    if table not in existing_tables:
        return False
    return col in [c['name'] for c in inspector.get_columns(table)]

print("=" * 60)
print("DATABASE CLEANUP")
print("=" * 60)

with engine.begin() as conn:
    # 1. Drop stale columns from agents
    for col in ['rating', 'rating_count', 'image_url']:
        if col_exists('agents', col):
            conn.execute(text(f"ALTER TABLE agents DROP COLUMN {col}"))
            print(f"✅ Dropped agents.{col}")
        else:
            print(f"   agents.{col} already gone")

    # 2. Make agents.capabilities nullable (migration wanted this)
    conn.execute(text("ALTER TABLE agents ALTER COLUMN capabilities DROP NOT NULL"))
    print("✅ Made agents.capabilities nullable")

    # 3. Drop stale empty tables
    for table in ['connection_logs', 'user_connections']:
        if table in existing_tables:
            # Drop indexes first
            indexes = inspector.get_indexes(table)
            for idx in indexes:
                try:
                    conn.execute(text(f"DROP INDEX IF EXISTS {idx['name']}"))
                except Exception:
                    pass
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            print(f"✅ Dropped table {table}")
        else:
            print(f"   {table} already gone")

    # 4. Drop stale metrics columns (migration wanted these dropped)
    for table, col in [
        ('agent_usage_analytics', 'metrics'),
        ('conversation_analytics', 'metrics'),
        ('workflow_execution_analytics', 'metrics'),
    ]:
        if col_exists(table, col):
            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {col}"))
            print(f"✅ Dropped {table}.{col}")
        else:
            print(f"   {table}.{col} already gone")

    # 5. Stamp alembic to e7a71c4bf948 (the migration is now complete)
    conn.execute(text("UPDATE alembic_version SET version_num = 'e7a71c4bf948'"))
    print("✅ Stamped alembic_version to e7a71c4bf948")

print("\n" + "=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)

# Verify
print("\n--- VERIFICATION ---")
inspector2 = inspect(engine)
agents_cols = [c['name'] for c in inspector2.get_columns('agents')]
print(f"agents columns: {agents_cols}")
remaining_tables = sorted(inspector2.get_table_names())
print(f"tables: {remaining_tables}")
with engine.connect() as conn:
    result = conn.execute(text("SELECT version_num FROM alembic_version"))
    print(f"alembic version: {result.scalar()}")

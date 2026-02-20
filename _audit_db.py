"""Audit database schema: compare actual DB tables/columns vs models.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from database import engine
from sqlalchemy import inspect, text

inspector = inspect(engine)
tables = sorted(inspector.get_table_names())

print("=" * 70)
print("DATABASE SCHEMA AUDIT")
print("=" * 70)

print(f"\nTotal tables in DB: {len(tables)}")

# Known model tables from models.py
model_tables = {
    "agents", "agent_capabilities", "agent_endpoints", "endpoint_parameters",
    "user_threads", "workflows", "workflow_executions", "workflow_schedules",
    "workflow_webhooks", "agent_credentials",
    "user_connections", "connection_logs",  # Should be DROPPED by migration
    "conversation_plans", "conversation_search",
    "conversation_tags", "conversation_tag_assignments",
    "conversation_analytics", "agent_usage_analytics",
    "user_activity_summary", "workflow_execution_analytics",
}

# Alembic internal
alembic_tables = {"alembic_version"}

print("\n--- TABLES IN DB ---")
for t in tables:
    status = ""
    if t in alembic_tables:
        status = "(alembic internal)"
    elif t in model_tables:
        status = "(in models.py)"
    else:
        status = "⚠️  NOT IN MODELS.PY - candidate for removal"
    print(f"  {t:45s} {status}")

orphan_tables = [t for t in tables if t not in model_tables and t not in alembic_tables]
if orphan_tables:
    print(f"\n⚠️  {len(orphan_tables)} tables exist in DB but NOT in models.py")

# Check columns for agents table specifically (the problem table)
print("\n--- AGENTS TABLE COLUMNS ---")
cols = inspector.get_columns("agents")
for c in cols:
    print(f"  {c['name']:30s} {str(c['type']):20s} nullable={c.get('nullable')}")

# Check if rating columns still exist
rating_cols = [c for c in cols if 'rating' in c['name'].lower() or 'image_url' in c['name'].lower()]
if rating_cols:
    print(f"\n⚠️  STALE columns in agents: {[c['name'] for c in rating_cols]}")

# Check alembic version
print("\n--- ALEMBIC VERSION ---")
with engine.connect() as conn:
    result = conn.execute(text("SELECT version_num FROM alembic_version"))
    for row in result:
        print(f"  Current: {row[0]}")

# Check if user_connections and connection_logs still exist
print("\n--- TABLES THAT SHOULD BE DROPPED (per migration) ---")
for t in ["user_connections", "connection_logs"]:
    if t in tables:
        print(f"  ⚠️  {t} still exists in DB (migration never completed)")
    else:
        print(f"  ✅ {t} already dropped")

# Check for empty tables
print("\n--- TABLE ROW COUNTS ---")
with engine.connect() as conn:
    for t in tables:
        if t != "alembic_version":
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                count = result.scalar()
                marker = " (EMPTY)" if count == 0 else ""
                print(f"  {t:45s} {count:>6d} rows{marker}")
            except Exception as e:
                print(f"  {t:45s} ERROR: {e}")

print()

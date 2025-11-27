#!/usr/bin/env python3
from sqlalchemy import create_engine, text

db_url = 'postgresql://postgres:Mahesh*456@localhost:5432/agentdb'
engine = create_engine(db_url)

with engine.connect() as conn:
    # Check all new tables
    tables_to_check = [
        'conversation_plans', 'conversation_search', 'conversation_tags',
        'conversation_tag_assignments', 'conversation_analytics',
        'agent_usage_analytics', 'user_activity_summary',
        'workflow_execution_analytics', 'workflow_executions'
    ]
    
    print("Checking new tables:")
    for table in tables_to_check:
        result = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='{table}')"))
        exists = list(result)[0][0]
        print(f"  {table}: {'✓' if exists else '❌'}")

print("\nChecking workflows table columns:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema='public' AND table_name='workflows'
        ORDER BY column_name
    """))
    cols = [row[0] for row in result]
    has_plan_graph = 'plan_graph' in cols
    print(f"  plan_graph: {'✓' if has_plan_graph else '❌'}")
    print(f"  Total columns: {len(cols)}")

print("\nChecking workflow_schedules table columns:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema='public' AND table_name='workflow_schedules'
        ORDER BY column_name
    """))
    cols = [row[0] for row in result]
    has_last_run = 'last_run_at' in cols
    print(f"  last_run_at: {'✓' if has_last_run else '❌'}")
    print(f"  Total columns: {len(cols)}")

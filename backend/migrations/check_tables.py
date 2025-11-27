#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text

db_url = 'postgresql://postgres:Mahesh*456@localhost:5432/agentdb'
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema='public' 
        AND (table_name LIKE 'conversation%' 
             OR table_name LIKE 'agent_usage%' 
             OR table_name LIKE 'workflow_execution%' 
             OR table_name LIKE 'user_activity%')
        ORDER BY table_name
    """))
    tables = [row[0] for row in result]
    
print(f"Found {len(tables)} new tables:")
for t in tables:
    print(f"  ✓ {t}")

# Also check if plan_graph column was added
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema='public' 
        AND table_name='workflows' 
        AND column_name='plan_graph'
    """))
    has_plan_graph = len(list(result)) > 0

print(f"\nplan_graph column in workflows: {'✓' if has_plan_graph else '❌'}")

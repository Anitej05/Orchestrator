#!/usr/bin/env python3
"""
Test script to verify database setup for new users.
Run this to ensure all tables are created correctly.
"""

from database import engine
from db_init import init_database
from sqlalchemy import inspect, text
import sys

def test_database_setup():
    """Test that database initialization works correctly."""
    print("=" * 70)
    print("Database Setup Test")
    print("=" * 70)
    
    # Initialize database
    print("\n1. Initializing database...")
    try:
        init_database(engine)
        print("   ✅ Database initialization completed")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Verify all tables exist
    print("\n2. Verifying tables...")
    inspector = inspect(engine)
    existing_tables = sorted(inspector.get_table_names())
    
    expected_tables = [
        'agents', 'agent_capabilities', 'agent_endpoints', 'endpoint_parameters',
        'agent_credentials', 'user_threads', 'workflows', 'workflow_executions',
        'workflow_schedules', 'workflow_webhooks',
        'conversation_plans', 'conversation_search', 'conversation_tags',
        'conversation_tag_assignments', 'conversation_analytics',
        'agent_usage_analytics', 'user_activity_summary', 'workflow_execution_analytics'
    ]
    
    missing_tables = [table for table in expected_tables if table not in existing_tables]
    extra_tables = [table for table in existing_tables if table not in expected_tables]
    
    if missing_tables:
        print(f"   ❌ Missing tables: {', '.join(missing_tables)}")
        return False
    
    print(f"   ✅ All {len(expected_tables)} required tables exist")
    if extra_tables:
        print(f"   ℹ️  Extra tables found: {', '.join(extra_tables)}")
    
    # Test basic operations
    print("\n3. Testing basic database operations...")
    try:
        with engine.connect() as conn:
            # Test insert
            result = conn.execute(text("SELECT COUNT(*) FROM agents"))
            count = result.scalar()
            print(f"   ✅ Can query agents table (found {count} agents)")
            
            # Test user_threads
            result = conn.execute(text("SELECT COUNT(*) FROM user_threads"))
            count = result.scalar()
            print(f"   ✅ Can query user_threads table (found {count} threads)")
            
            # Test workflows
            result = conn.execute(text("SELECT COUNT(*) FROM workflows"))
            count = result.scalar()
            print(f"   ✅ Can query workflows table (found {count} workflows)")
            
    except Exception as e:
        print(f"   ❌ Database operation failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ All tests passed! Database is ready for use.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_database_setup()
    sys.exit(0 if success else 1)

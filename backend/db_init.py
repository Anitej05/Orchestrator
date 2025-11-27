"""
Database initialization and migration system.

This module ensures all database tables are created and handles schema migrations.
All table definitions come from models.py - this just ensures they exist.
"""

from sqlalchemy import inspect, text
import logging

logger = logging.getLogger(__name__)

def init_database(engine):
    """
    Initialize the database by creating all tables defined in models.py.
    This is safe to run multiple times - it only creates tables that don't exist.
    Also handles column migrations for existing tables.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    from database import Base
    from models import (
        Agent, AgentCapability, AgentEndpoint, EndpointParameter,
        UserThread, Workflow, WorkflowExecution, WorkflowSchedule, 
        WorkflowWebhook, AgentCredential,
        ConversationPlan, ConversationSearch, ConversationTag,
        ConversationTagAssignment, ConversationAnalytics, AgentUsageAnalytics,
        UserActivitySummary, WorkflowExecutionAnalytics
    )
    
    try:
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        
        # Run column migrations for existing installations
        _run_column_migrations(engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        
        expected_tables = [
            'agents', 'agent_capabilities', 'agent_endpoints', 'endpoint_parameters',
            'agent_credentials', 'user_threads', 'workflows', 'workflow_executions',
            'workflow_schedules', 'workflow_webhooks',
            'conversation_plans', 'conversation_search', 'conversation_tags',
            'conversation_tag_assignments', 'conversation_analytics',
            'agent_usage_analytics', 'user_activity_summary', 'workflow_execution_analytics'
        ]
        
        missing_tables = [table for table in expected_tables if table not in existing_tables]
        
        if missing_tables:
            logger.warning(f"⚠️ Some tables were not created: {missing_tables}")
            return False
        else:
            logger.info("✅ Database initialized successfully - all tables ready")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def _run_column_migrations(engine):
    """
    Add missing columns to existing tables.
    This handles schema updates for existing installations.
    """
    inspector = inspect(engine)
    
    # Migration: Add agent_type and connection_config to agents table
    if inspector.has_table('agents'):
        columns = [col['name'] for col in inspector.get_columns('agents')]
        
        with engine.connect() as conn:
            if 'agent_type' not in columns:
                logger.info("  Adding agent_type column to agents table...")
                conn.execute(text("ALTER TABLE agents ADD COLUMN agent_type VARCHAR(50) DEFAULT 'http_rest'"))
                conn.commit()
            
            if 'connection_config' not in columns:
                logger.info("  Adding connection_config column to agents table...")
                conn.execute(text("ALTER TABLE agents ADD COLUMN connection_config JSON"))
                conn.commit()
    
    # Migration: Add plan_graph to workflows table
    if inspector.has_table('workflows'):
        columns = [col['name'] for col in inspector.get_columns('workflows')]
        
        with engine.connect() as conn:
            if 'plan_graph' not in columns:
                logger.info("  Adding plan_graph column to workflows table...")
                conn.execute(text("ALTER TABLE workflows ADD COLUMN plan_graph JSON"))
                conn.commit()

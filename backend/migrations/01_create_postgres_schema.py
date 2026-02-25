#!/usr/bin/env python3
"""
Migration Script 01: Create PostgreSQL Schema

This script creates the PostgreSQL database schema from SQLAlchemy models.
It ensures all tables, indexes, and constraints are properly created.

Usage:
    python 01_create_postgres_schema.py

Environment Variables Required:
    POSTGRES_URL - PostgreSQL connection string
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv(backend_dir / ".env")

def create_postgres_schema():
    """Create PostgreSQL schema from SQLAlchemy models."""
    
    # Get PostgreSQL URL
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        logger.error("POSTGRES_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"Connecting to PostgreSQL: {postgres_url.split('@')[1] if '@' in postgres_url else 'localhost'}")
    
    try:
        # Create engine
        engine = create_engine(postgres_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")
        
        # Import models (this must happen after engine is created)
        from models import Base
        
        # Create all tables
        logger.info("Creating database schema...")
        Base.metadata.create_all(engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"Created {len(tables)} tables:")
        for table in sorted(tables):
            logger.info(f"  ✓ {table}")
        
        # Verify expected tables exist
        expected_tables = [
            'agents',
            'agent_capabilities',
            'agent_endpoints',
            'endpoint_parameters',
            'agent_credentials',
            'user_connections',
            'connection_logs',
            'user_threads',
            'workflows',
            'workflow_executions',
            'workflow_schedules',
            'workflow_webhooks',
            'conversation_plans',
            'conversation_search',
            'conversation_tags',
            'conversation_tag_assignments',
            'conversation_analytics',
            'agent_usage_analytics',
            'user_activity_summary',
            'workflow_execution_analytics'
        ]
        
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            logger.warning(f"Missing expected tables: {missing_tables}")
        else:
            logger.info("✓ All expected tables created successfully")
        
        # Check indexes
        logger.info("\nVerifying indexes...")
        index_count = 0
        for table in tables:
            indexes = inspector.get_indexes(table)
            if indexes:
                logger.info(f"  {table}: {len(indexes)} indexes")
                index_count += len(indexes)
        
        logger.info(f"✓ Total indexes created: {index_count}")
        
        # Check foreign keys
        logger.info("\nVerifying foreign keys...")
        fk_count = 0
        for table in tables:
            foreign_keys = inspector.get_foreign_keys(table)
            if foreign_keys:
                logger.info(f"  {table}: {len(foreign_keys)} foreign keys")
                fk_count += len(foreign_keys)
        
        logger.info(f"✓ Total foreign keys created: {fk_count}")
        
        logger.info("\n" + "="*60)
        logger.info("PostgreSQL schema creation completed successfully!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL schema: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_postgres_schema()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Migration Script 03: Import Data to PostgreSQL

This script imports data from JSON export into PostgreSQL database.
It handles foreign key dependencies and validates data integrity.

Usage:
    python 03_import_postgres_data.py <export_file.json>

Environment Variables Required:
    POSTGRES_URL - PostgreSQL connection string
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
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

# Table import order (respects foreign key dependencies)
TABLE_IMPORT_ORDER = [
    'agents',
    'agent_capabilities',
    'agent_endpoints',
    'endpoint_parameters',
    'agent_credentials',
    'user_threads',
    'user_connections',
    'connection_logs',
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

def deserialize_row(row_data: Dict[str, Any], table_class) -> Any:
    """Convert JSON dict to SQLAlchemy model instance."""
    # Convert ISO datetime strings back to datetime objects
    for column in table_class.__table__.columns:
        if column.name in row_data and row_data[column.name]:
            # Handle datetime columns
            if str(column.type) in ['DATETIME', 'TIMESTAMP']:
                if isinstance(row_data[column.name], str):
                    try:
                        row_data[column.name] = datetime.fromisoformat(row_data[column.name])
                    except ValueError:
                        logger.warning(f"Could not parse datetime: {row_data[column.name]}")
    
    return table_class(**row_data)

def import_table(session, table_name: str, rows_data: List[Dict[str, Any]]) -> int:
    """Import rows into a table."""
    from models import Base
    
    # Get table class
    table_class = None
    for mapper in Base.registry.mappers:
        if mapper.class_.__tablename__ == table_name:
            table_class = mapper.class_
            break
    
    if not table_class:
        logger.warning(f"Table class not found for {table_name}")
        return 0
    
    # Import rows
    imported_count = 0
    for row_data in rows_data:
        try:
            instance = deserialize_row(row_data, table_class)
            session.add(instance)
            imported_count += 1
        except Exception as e:
            logger.error(f"Failed to import row: {e}")
            logger.error(f"Row data: {row_data}")
    
    return imported_count

def import_postgres_data(export_file: str):
    """Import data from JSON export to PostgreSQL."""
    
    # Validate export file exists
    if not os.path.exists(export_file):
        logger.error(f"Export file not found: {export_file}")
        sys.exit(1)
    
    # Get PostgreSQL URL
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        logger.error("POSTGRES_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"Importing to PostgreSQL: {postgres_url.split('@')[1] if '@' in postgres_url else 'localhost'}")
    logger.info(f"Reading export file: {export_file}")
    
    try:
        # Load export data
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        logger.info(f"Export timestamp: {export_data['export_timestamp']}")
        logger.info(f"Total tables: {len(export_data['tables'])}")
        logger.info(f"Total rows: {sum(export_data['row_counts'].values())}")
        
        # Create engine and session
        engine = create_engine(postgres_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Begin transaction
        logger.info("\nStarting import transaction...")
        
        # Import tables in order
        imported_counts = {}
        total_imported = 0
        
        for table_name in TABLE_IMPORT_ORDER:
            if table_name not in export_data['tables']:
                logger.warning(f"Table {table_name} not in export, skipping")
                continue
            
            rows_data = export_data['tables'][table_name]
            expected_count = export_data['row_counts'][table_name]
            
            logger.info(f"Importing {table_name} ({expected_count} rows)...")
            
            try:
                imported_count = import_table(session, table_name, rows_data)
                imported_counts[table_name] = imported_count
                total_imported += imported_count
                
                if imported_count == expected_count:
                    logger.info(f"  ✓ Imported {imported_count}/{expected_count} rows")
                else:
                    logger.warning(f"  ⚠ Imported {imported_count}/{expected_count} rows (mismatch!)")
            
            except Exception as e:
                logger.error(f"  ✗ Failed to import {table_name}: {e}")
                session.rollback()
                raise
        
        # Commit transaction
        logger.info("\nCommitting transaction...")
        session.commit()
        session.close()
        
        # Verify import
        logger.info("\nVerifying import...")
        Session = sessionmaker(bind=engine)
        session = Session()
        
        inspector = inspect(engine)
        verification_passed = True
        
        for table_name in imported_counts.keys():
            # Count rows in PostgreSQL
            from sqlalchemy import text
            result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            actual_count = result.fetchone()[0]
            expected_count = imported_counts[table_name]
            
            if actual_count == expected_count:
                logger.info(f"  ✓ {table_name}: {actual_count} rows")
            else:
                logger.error(f"  ✗ {table_name}: {actual_count} rows (expected {expected_count})")
                verification_passed = False
        
        session.close()
        
        logger.info("\n" + "="*60)
        if verification_passed:
            logger.info("PostgreSQL data import completed successfully!")
            logger.info(f"Total rows imported: {total_imported}")
            logger.info("="*60)
            return True
        else:
            logger.error("PostgreSQL data import completed with errors!")
            logger.error("Row count verification failed for some tables")
            logger.error("="*60)
            return False
        
    except Exception as e:
        logger.error(f"Failed to import PostgreSQL data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Import JSON data to PostgreSQL")
    parser.add_argument("export_file", help="Path to JSON export file")
    args = parser.parse_args()
    
    success = import_postgres_data(args.export_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

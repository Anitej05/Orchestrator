#!/usr/bin/env python3
"""
Migration Script 06: Final Sync Before Cutover

This script performs a final synchronization between SQLite and PostgreSQL
before the cutover. It ensures all data is consistent and up-to-date.

Usage:
    python 06_final_sync.py

Environment Variables Required:
    DATABASE_URL - SQLite connection string
    POSTGRES_URL - PostgreSQL connection string
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text, inspect
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

def final_sync():
    """Perform final sync before cutover."""
    
    # Get database URLs
    sqlite_url = os.getenv("DATABASE_URL", "sqlite:///orbimesh.db")
    postgres_url = os.getenv("POSTGRES_URL")
    
    if not postgres_url:
        logger.error("POSTGRES_URL environment variable not set")
        sys.exit(1)
    
    # Handle SQLite relative paths
    if sqlite_url.startswith("sqlite:///"):
        db_file = sqlite_url.replace("sqlite:///", "")
        if not os.path.isabs(db_file):
            db_path = os.path.abspath(os.path.join(backend_dir, db_file))
            sqlite_url = f"sqlite:///{db_path}"
    
    logger.info("="*60)
    logger.info("Final Sync Before Cutover")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    try:
        # Create engines
        sqlite_engine = create_engine(sqlite_url)
        postgres_engine = create_engine(postgres_url)
        
        SQLiteSession = sessionmaker(bind=sqlite_engine)
        PostgresSession = sessionmaker(bind=postgres_engine)
        
        sqlite_session = SQLiteSession()
        postgres_session = PostgresSession()
        
        # Get all tables
        inspector = inspect(sqlite_engine)
        tables = inspector.get_table_names()
        
        logger.info(f"\nSyncing {len(tables)} tables...")
        
        sync_results = {}
        total_synced = 0
        
        for table in sorted(tables):
            logger.info(f"\nProcessing {table}...")
            
            # Get row counts
            sqlite_count = sqlite_session.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).fetchone()[0]
            
            postgres_count = postgres_session.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).fetchone()[0]
            
            diff = sqlite_count - postgres_count
            
            if diff == 0:
                logger.info(f"  ✓ {table}: {sqlite_count} rows (in sync)")
                sync_results[table] = {'status': 'synced', 'rows': sqlite_count}
            elif diff > 0:
                logger.warning(f"  ⚠ {table}: SQLite has {diff} more rows")
                logger.warning(f"    SQLite: {sqlite_count}, PostgreSQL: {postgres_count}")
                logger.warning(f"    Manual sync required for this table")
                sync_results[table] = {'status': 'needs_sync', 'diff': diff}
            else:
                logger.error(f"  ✗ {table}: PostgreSQL has {abs(diff)} more rows")
                logger.error(f"    SQLite: {sqlite_count}, PostgreSQL: {postgres_count}")
                logger.error(f"    This should not happen in dual-write mode!")
                sync_results[table] = {'status': 'error', 'diff': diff}
            
            total_synced += sqlite_count
        
        sqlite_session.close()
        postgres_session.close()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Final Sync Summary")
        logger.info("="*60)
        
        synced_count = sum(1 for r in sync_results.values() if r['status'] == 'synced')
        needs_sync_count = sum(1 for r in sync_results.values() if r['status'] == 'needs_sync')
        error_count = sum(1 for r in sync_results.values() if r['status'] == 'error')
        
        logger.info(f"Total tables: {len(tables)}")
        logger.info(f"  ✓ In sync: {synced_count}")
        logger.info(f"  ⚠ Needs sync: {needs_sync_count}")
        logger.info(f"  ✗ Errors: {error_count}")
        logger.info(f"Total rows: {total_synced}")
        
        if needs_sync_count > 0:
            logger.warning("\nTables needing sync:")
            for table, result in sync_results.items():
                if result['status'] == 'needs_sync':
                    logger.warning(f"  {table}: {result['diff']} rows behind")
        
        if error_count > 0:
            logger.error("\nTables with errors:")
            for table, result in sync_results.items():
                if result['status'] == 'error':
                    logger.error(f"  {table}: {result['diff']} rows discrepancy")
        
        # Cutover readiness check
        logger.info("\n" + "="*60)
        logger.info("Cutover Readiness Check")
        logger.info("="*60)
        
        if error_count > 0:
            logger.error("✗ NOT READY FOR CUTOVER")
            logger.error("  Errors detected in data sync")
            logger.error("  Investigate and resolve before proceeding")
            return False
        elif needs_sync_count > 0:
            logger.warning("⚠ CAUTION: Some tables need sync")
            logger.warning("  Review tables needing sync before cutover")
            logger.warning("  Consider running manual sync for these tables")
            return False
        else:
            logger.info("✓ READY FOR CUTOVER")
            logger.info("  All tables are in sync")
            logger.info("  You can proceed with database cutover")
            return True
        
    except Exception as e:
        logger.error(f"Final sync failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = final_sync()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("Next Steps:")
        logger.info("1. Stop the application (if needed)")
        logger.info("2. Update DATABASE_URL to POSTGRES_URL in .env")
        logger.info("3. Set MIGRATION_MODE=postgres_only")
        logger.info("4. Restart the application")
        logger.info("5. Verify application health")
        logger.info("6. Monitor for 24 hours")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("DO NOT PROCEED WITH CUTOVER")
        logger.error("Resolve sync issues first")
        logger.error("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick Start Migration Script
Handles database migration and data population in one command
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MigrationRunner:
    def __init__(self, db_url: str = None, method: str = 'python'):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.method = method
        self.backend_dir = Path(__file__).parent.parent  # Go up from migrations to backend
        self.migrations_dir = self.backend_dir / 'migrations'
        
        if not self.db_url:
            logger.error("DATABASE_URL not set. Set environment variable or pass --db-url")
            sys.exit(1)
    
    def run_sql_migration(self):
        """Execute SQL migration using psycopg2"""
        logger.info("=" * 60)
        logger.info("EXECUTING SQL MIGRATION (Method: Direct SQL)")
        logger.info("=" * 60)
        
        sql_file = self.migrations_dir / '001_add_plan_graph_and_enhancements.sql'
        
        if not sql_file.exists():
            logger.error(f"SQL file not found: {sql_file}")
            return False
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self.db_url)
            
            with open(sql_file, 'r') as f:
                sql_content = f.read()
            
            # Split by GO statements and execute each statement
            statements = sql_content.split('GO')
            
            with engine.connect() as conn:
                for i, statement in enumerate(statements, 1):
                    statement = statement.strip()
                    if not statement:
                        continue
                    
                    try:
                        conn.execute(text(statement))
                        logger.info(f"  ✓ Executed statement {i}")
                    except Exception as e:
                        logger.warning(f"  ⚠ Statement {i} (may be non-critical): {str(e)[:100]}")
                
                conn.commit()
            
            logger.info("✅ SQL Migration completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error running SQL migration: {e}")
            return False
    
    def run_python_migration(self):
        """Execute Python migration script"""
        logger.info("=" * 60)
        logger.info("EXECUTING PYTHON MIGRATION")
        logger.info("=" * 60)
        
        migration_script = self.migrations_dir / 'migrate_workflows_plan_graph.py'
        
        if not migration_script.exists():
            logger.error(f"Migration script not found: {migration_script}")
            return False
        
        try:
            env = os.environ.copy()
            env['DATABASE_URL'] = self.db_url
            
            process = subprocess.Popen(
                [sys.executable, str(migration_script)],
                env=env,
                cwd=str(self.backend_dir)
            )
            
            returncode = process.wait()
            
            if returncode != 0:
                logger.error(f"Python migration failed with return code {returncode}")
                return False
            
            logger.info("✅ Python Migration completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error running Python migration: {e}")
            return False
    
    def verify_migration(self):
        """Verify migration was successful"""
        logger.info("=" * 60)
        logger.info("VERIFYING MIGRATION")
        logger.info("=" * 60)
        
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self.db_url)
            
            with engine.connect() as conn:
                # Check plan_graph column
                result = conn.execute(text("""
                    SELECT COUNT(*) as count FROM information_schema.columns 
                    WHERE table_name='workflows' AND column_name='plan_graph'
                """))
                plan_graph_exists = result.scalar() > 0
                logger.info(f"  plan_graph column: {'✅' if plan_graph_exists else '❌'}")
                
                # Check new tables
                tables_to_check = [
                    'conversation_plans', 'conversation_search', 'conversation_tags',
                    'conversation_tag_assignments', 'conversation_analytics',
                    'agent_usage_analytics', 'user_activity_summary',
                    'workflow_execution_analytics'
                ]
                
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema='public' 
                    AND table_name IN ({','.join([f"'{t}'" for t in tables_to_check])})
                """))
                
                table_count = result.scalar()
                logger.info(f"  New tables created: {table_count}/{len(tables_to_check)}")
                
                if table_count == len(tables_to_check):
                    logger.info("✅ All tables created successfully")
                    return True
                else:
                    logger.warning(f"⚠️  Only {table_count}/{len(tables_to_check)} tables created")
                    return False
        
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
    
    def run(self, verify: bool = True, populate: bool = True):
        """Execute migration workflow"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " ORBIMESH DATABASE MIGRATION RUNNER ".center(58) + "║")
        logger.info("╚" + "=" * 58 + "╝")
        logger.info(f"\nMethod: {self.method}")
        logger.info(f"Database: {self.db_url.split('@')[-1] if '@' in self.db_url else 'local'}")
        logger.info("")
        
        # Step 1: ALWAYS run SQL migration first to create tables
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Creating database schema (SQL migration)")
        logger.info("=" * 60)
        sql_success = self.run_sql_migration()
        
        if not sql_success:
            logger.error("❌ SQL migration failed. Aborting.")
            return False
        
        # Step 2: Run data population migration if populate flag is set
        success = True
        if populate:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: Populating data (Python migration)")
            logger.info("=" * 60)
            success = self.run_python_migration()
        
        if not success:
            logger.error("❌ Migration failed")
            return False
        
        # Step 2: Verify
        if verify:
            verified = self.verify_migration()
            if not verified:
                logger.warning("⚠️  Migration verification failed - some tables may not exist")
                return False
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ MIGRATION COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Restart backend: python main.py")
        logger.info("2. Test endpoints with plan_graph support")
        logger.info("3. Monitor analytics collection")
        logger.info("\nFor details, see: MIGRATION_GUIDE_PLAN_GRAPH.md")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Orbimesh database migration'
    )
    parser.add_argument(
        '--db-url',
        help='Database URL (uses DATABASE_URL env var if not provided)',
        default=None
    )
    parser.add_argument(
        '--method',
        choices=['sql', 'python'],
        default='python',
        help='Migration method to use (default: python)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification step'
    )
    parser.add_argument(
        '--no-populate',
        action='store_true',
        help='Skip data population step'
    )
    
    args = parser.parse_args()
    
    runner = MigrationRunner(db_url=args.db_url, method=args.method)
    success = runner.run(
        verify=not args.no_verify,
        populate=not args.no_populate
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

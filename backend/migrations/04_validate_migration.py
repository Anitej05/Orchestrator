#!/usr/bin/env python3
"""
Migration Script 04: Validate Migration

This script validates data consistency between SQLite and PostgreSQL databases.
It performs comprehensive checks including row counts, data integrity, and relationships.

Usage:
    python 04_validate_migration.py [--full]

Environment Variables Required:
    DATABASE_URL - SQLite connection string
    POSTGRES_URL - PostgreSQL connection string
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, inspect, text
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

class MigrationValidator:
    """Validates data consistency between SQLite and PostgreSQL."""
    
    def __init__(self):
        # Get database URLs
        self.sqlite_url = os.getenv("DATABASE_URL", "sqlite:///orbimesh.db")
        self.postgres_url = os.getenv("POSTGRES_URL")
        
        if not self.postgres_url:
            logger.error("POSTGRES_URL environment variable not set")
            sys.exit(1)
        
        # Handle SQLite relative paths
        if self.sqlite_url.startswith("sqlite:///"):
            db_file = self.sqlite_url.replace("sqlite:///", "")
            if not os.path.isabs(db_file):
                db_path = os.path.abspath(os.path.join(backend_dir, db_file))
                self.sqlite_url = f"sqlite:///{db_path}"
        
        # Create engines
        self.sqlite_engine = create_engine(self.sqlite_url)
        self.postgres_engine = create_engine(self.postgres_url)
        
        # Create sessions
        SQLiteSession = sessionmaker(bind=self.sqlite_engine)
        PostgresSession = sessionmaker(bind=self.postgres_engine)
        
        self.sqlite_session = SQLiteSession()
        self.postgres_session = PostgresSession()
        
        self.errors = []
        self.warnings = []
    
    def validate_row_counts(self) -> bool:
        """Validate row counts match between databases."""
        logger.info("Validating row counts...")
        
        # Get tables from both databases
        sqlite_inspector = inspect(self.sqlite_engine)
        postgres_inspector = inspect(self.postgres_engine)
        
        sqlite_tables = set(sqlite_inspector.get_table_names())
        postgres_tables = set(postgres_inspector.get_table_names())
        
        # Check for missing tables
        missing_in_postgres = sqlite_tables - postgres_tables
        if missing_in_postgres:
            self.errors.append(f"Tables missing in PostgreSQL: {missing_in_postgres}")
        
        # Compare row counts
        all_match = True
        for table in sorted(sqlite_tables & postgres_tables):
            sqlite_count = self.sqlite_session.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).fetchone()[0]
            
            postgres_count = self.postgres_session.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).fetchone()[0]
            
            if sqlite_count == postgres_count:
                logger.info(f"  ✓ {table}: {sqlite_count} rows")
            else:
                logger.error(f"  ✗ {table}: SQLite={sqlite_count}, PostgreSQL={postgres_count}")
                self.errors.append(f"Row count mismatch in {table}")
                all_match = False
        
        return all_match
    
    def validate_foreign_keys(self) -> bool:
        """Validate foreign key relationships are intact."""
        logger.info("\nValidating foreign key relationships...")
        
        postgres_inspector = inspect(self.postgres_engine)
        tables = postgres_inspector.get_table_names()
        
        all_valid = True
        for table in tables:
            foreign_keys = postgres_inspector.get_foreign_keys(table)
            
            for fk in foreign_keys:
                # Check for orphaned records
                query = text(f"""
                    SELECT COUNT(*) FROM {table} t
                    LEFT JOIN {fk['referred_table']} r 
                    ON t.{fk['constrained_columns'][0]} = r.{fk['referred_columns'][0]}
                    WHERE t.{fk['constrained_columns'][0]} IS NOT NULL 
                    AND r.{fk['referred_columns'][0]} IS NULL
                """)
                
                try:
                    result = self.postgres_session.execute(query)
                    orphaned_count = result.fetchone()[0]
                    
                    if orphaned_count > 0:
                        logger.error(f"  ✗ {table}.{fk['constrained_columns'][0]} has {orphaned_count} orphaned records")
                        self.errors.append(f"Orphaned records in {table}")
                        all_valid = False
                    else:
                        logger.info(f"  ✓ {table}.{fk['constrained_columns'][0]} → {fk['referred_table']}.{fk['referred_columns'][0]}")
                
                except Exception as e:
                    logger.warning(f"  ⚠ Could not validate FK {table}.{fk['constrained_columns'][0]}: {e}")
                    self.warnings.append(f"FK validation skipped for {table}")
        
        return all_valid
    
    def validate_encrypted_data(self) -> bool:
        """Validate encrypted connection IDs can be decrypted."""
        logger.info("\nValidating encrypted data...")
        
        try:
            from utils.encryption import decrypt_connection_id
            
            # Check user_connections table
            result = self.postgres_session.execute(
                text("SELECT id, connection_id FROM user_connections LIMIT 10")
            )
            
            all_valid = True
            for row in result:
                try:
                    decrypted = decrypt_connection_id(row[1])
                    if decrypted:
                        logger.info(f"  ✓ Connection {row[0]}: decryption successful")
                    else:
                        logger.error(f"  ✗ Connection {row[0]}: decryption returned None")
                        self.errors.append(f"Decryption failed for connection {row[0]}")
                        all_valid = False
                except Exception as e:
                    logger.error(f"  ✗ Connection {row[0]}: {e}")
                    self.errors.append(f"Decryption error for connection {row[0]}")
                    all_valid = False
            
            return all_valid
        
        except ImportError:
            logger.warning("  ⚠ Encryption module not available, skipping validation")
            self.warnings.append("Encryption validation skipped")
            return True
        except Exception as e:
            logger.error(f"  ✗ Encryption validation failed: {e}")
            self.errors.append("Encryption validation failed")
            return False
    
    def validate_json_fields(self) -> bool:
        """Validate JSON fields parse correctly."""
        logger.info("\nValidating JSON fields...")
        
        import json
        
        # Tables with JSON columns
        json_tables = {
            'agents': ['capabilities', 'connection_config', 'credential_fields'],
            'workflows': ['blueprint', 'plan_graph'],
            'workflow_executions': ['inputs', 'outputs'],
            'user_connections': ['app_metadata']
        }
        
        all_valid = True
        for table, columns in json_tables.items():
            for column in columns:
                try:
                    query = text(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 5")
                    result = self.postgres_session.execute(query)
                    
                    for row in result:
                        if row[1]:
                            try:
                                # PostgreSQL returns dict/list directly for JSONB
                                if isinstance(row[1], (dict, list)):
                                    logger.info(f"  ✓ {table}.{column} (id={row[0]}): valid JSONB")
                                else:
                                    # Try parsing as string
                                    json.loads(row[1])
                                    logger.info(f"  ✓ {table}.{column} (id={row[0]}): valid JSON")
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.error(f"  ✗ {table}.{column} (id={row[0]}): invalid JSON")
                                self.errors.append(f"Invalid JSON in {table}.{column}")
                                all_valid = False
                
                except Exception as e:
                    logger.warning(f"  ⚠ Could not validate {table}.{column}: {e}")
                    self.warnings.append(f"JSON validation skipped for {table}.{column}")
        
        return all_valid
    
    def validate_timestamps(self) -> bool:
        """Validate timestamp conversion."""
        logger.info("\nValidating timestamps...")
        
        from datetime import datetime
        
        # Tables with timestamp columns
        timestamp_tables = {
            'user_connections': ['auth_timestamp', 'created_at', 'updated_at'],
            'agents': ['created_at'],
            'workflows': ['created_at', 'updated_at']
        }
        
        all_valid = True
        for table, columns in timestamp_tables.items():
            for column in columns:
                try:
                    query = text(f"SELECT id, {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 3")
                    result = self.postgres_session.execute(query)
                    
                    for row in result:
                        if row[1]:
                            if isinstance(row[1], datetime):
                                logger.info(f"  ✓ {table}.{column} (id={row[0]}): valid timestamp")
                            else:
                                logger.error(f"  ✗ {table}.{column} (id={row[0]}): not a datetime object")
                                self.errors.append(f"Invalid timestamp in {table}.{column}")
                                all_valid = False
                
                except Exception as e:
                    logger.warning(f"  ⚠ Could not validate {table}.{column}: {e}")
                    self.warnings.append(f"Timestamp validation skipped for {table}.{column}")
        
        return all_valid
    
    def run_validation(self, full: bool = False) -> bool:
        """Run all validation checks."""
        logger.info("="*60)
        logger.info("Starting Migration Validation")
        logger.info("="*60)
        
        results = []
        
        # Always run row count validation
        results.append(("Row Counts", self.validate_row_counts()))
        
        if full:
            # Run comprehensive validation
            results.append(("Foreign Keys", self.validate_foreign_keys()))
            results.append(("Encrypted Data", self.validate_encrypted_data()))
            results.append(("JSON Fields", self.validate_json_fields()))
            results.append(("Timestamps", self.validate_timestamps()))
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Validation Summary")
        logger.info("="*60)
        
        for check_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{check_name}: {status}")
        
        if self.warnings:
            logger.info(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if self.errors:
            logger.info(f"\nErrors: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        all_passed = all(result[1] for result in results)
        
        logger.info("\n" + "="*60)
        if all_passed:
            logger.info("✓ All validation checks passed!")
        else:
            logger.error("✗ Some validation checks failed!")
        logger.info("="*60)
        
        # Cleanup
        self.sqlite_session.close()
        self.postgres_session.close()
        
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Validate database migration")
    parser.add_argument("--full", action="store_true", help="Run full validation (slower)")
    args = parser.parse_args()
    
    validator = MigrationValidator()
    success = validator.run_validation(full=args.full)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Migration Script 05: Monitor Dual-Write Period

This script continuously monitors data consistency during the dual-write period.
It runs validation checks at regular intervals and alerts on discrepancies.

Usage:
    python 05_monitor_dual_write.py [--interval <seconds>]

Environment Variables Required:
    DATABASE_URL - SQLite connection string
    POSTGRES_URL - PostgreSQL connection string
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(backend_dir / 'logs' / 'dual_write_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv(backend_dir / ".env")

class DualWriteMonitor:
    """Monitors data consistency during dual-write period."""
    
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
        
        self.check_count = 0
        self.error_count = 0
    
    def check_row_counts(self) -> dict:
        """Check row counts in both databases."""
        SQLiteSession = sessionmaker(bind=self.sqlite_engine)
        PostgresSession = sessionmaker(bind=self.postgres_engine)
        
        sqlite_session = SQLiteSession()
        postgres_session = PostgresSession()
        
        # Key tables to monitor
        tables = [
            'user_connections',
            'connection_logs',
            'agents',
            'workflows',
            'workflow_executions',
            'user_threads'
        ]
        
        results = {}
        discrepancies = []
        
        for table in tables:
            try:
                sqlite_count = sqlite_session.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).fetchone()[0]
                
                postgres_count = postgres_session.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).fetchone()[0]
                
                results[table] = {
                    'sqlite': sqlite_count,
                    'postgres': postgres_count,
                    'match': sqlite_count == postgres_count
                }
                
                if sqlite_count != postgres_count:
                    discrepancies.append({
                        'table': table,
                        'sqlite': sqlite_count,
                        'postgres': postgres_count,
                        'diff': sqlite_count - postgres_count
                    })
            
            except Exception as e:
                logger.error(f"Error checking {table}: {e}")
                results[table] = {'error': str(e)}
        
        sqlite_session.close()
        postgres_session.close()
        
        return results, discrepancies
    
    def check_write_latency(self) -> dict:
        """Measure write latency to both databases."""
        import time
        
        SQLiteSession = sessionmaker(bind=self.sqlite_engine)
        PostgresSession = sessionmaker(bind=self.postgres_engine)
        
        sqlite_session = SQLiteSession()
        postgres_session = PostgresSession()
        
        # Test write to connection_logs
        test_data = {
            'user_id': 'monitor_test',
            'app_slug': 'test',
            'event_type': 'monitor_check',
            'status': 'success'
        }
        
        # SQLite write
        start = time.time()
        try:
            sqlite_session.execute(
                text("""
                    INSERT INTO connection_logs (user_id, app_slug, event_type, status)
                    VALUES (:user_id, :app_slug, :event_type, :status)
                """),
                test_data
            )
            sqlite_session.commit()
            sqlite_latency = (time.time() - start) * 1000  # ms
        except Exception as e:
            logger.error(f"SQLite write error: {e}")
            sqlite_latency = -1
            sqlite_session.rollback()
        
        # PostgreSQL write
        start = time.time()
        try:
            postgres_session.execute(
                text("""
                    INSERT INTO connection_logs (user_id, app_slug, event_type, status)
                    VALUES (:user_id, :app_slug, :event_type, :status)
                """),
                test_data
            )
            postgres_session.commit()
            postgres_latency = (time.time() - start) * 1000  # ms
        except Exception as e:
            logger.error(f"PostgreSQL write error: {e}")
            postgres_latency = -1
            postgres_session.rollback()
        
        # Cleanup test data
        try:
            sqlite_session.execute(
                text("DELETE FROM connection_logs WHERE user_id = 'monitor_test'")
            )
            sqlite_session.commit()
            
            postgres_session.execute(
                text("DELETE FROM connection_logs WHERE user_id = 'monitor_test'")
            )
            postgres_session.commit()
        except:
            pass
        
        sqlite_session.close()
        postgres_session.close()
        
        return {
            'sqlite_ms': sqlite_latency,
            'postgres_ms': postgres_latency,
            'overhead_ms': postgres_latency - sqlite_latency if sqlite_latency > 0 and postgres_latency > 0 else 0
        }
    
    def run_check(self):
        """Run a single monitoring check."""
        self.check_count += 1
        
        logger.info("="*60)
        logger.info(f"Dual-Write Monitor Check #{self.check_count}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("="*60)
        
        # Check row counts
        logger.info("\nChecking row counts...")
        results, discrepancies = self.check_row_counts()
        
        all_match = True
        for table, data in results.items():
            if 'error' in data:
                logger.error(f"  ✗ {table}: {data['error']}")
                all_match = False
            elif data['match']:
                logger.info(f"  ✓ {table}: {data['sqlite']} rows (match)")
            else:
                logger.error(f"  ✗ {table}: SQLite={data['sqlite']}, PostgreSQL={data['postgres']}")
                all_match = False
        
        if discrepancies:
            logger.error(f"\n⚠ Found {len(discrepancies)} discrepancies:")
            for disc in discrepancies:
                logger.error(f"  {disc['table']}: diff={disc['diff']} (SQLite={disc['sqlite']}, PostgreSQL={disc['postgres']})")
            self.error_count += 1
        
        # Check write latency
        logger.info("\nChecking write latency...")
        latency = self.check_write_latency()
        
        if latency['sqlite_ms'] > 0 and latency['postgres_ms'] > 0:
            logger.info(f"  SQLite: {latency['sqlite_ms']:.2f}ms")
            logger.info(f"  PostgreSQL: {latency['postgres_ms']:.2f}ms")
            logger.info(f"  Overhead: {latency['overhead_ms']:.2f}ms")
            
            if latency['overhead_ms'] > 100:
                logger.warning(f"  ⚠ High overhead detected: {latency['overhead_ms']:.2f}ms")
        else:
            logger.error("  ✗ Write latency check failed")
            self.error_count += 1
        
        # Summary
        logger.info("\n" + "="*60)
        if all_match and latency['sqlite_ms'] > 0:
            logger.info("✓ Check passed - databases in sync")
        else:
            logger.error("✗ Check failed - discrepancies detected")
        logger.info(f"Total checks: {self.check_count}, Errors: {self.error_count}")
        logger.info("="*60 + "\n")
        
        return all_match
    
    def monitor(self, interval: int = 3600):
        """Continuously monitor dual-write period."""
        logger.info("Starting dual-write monitoring...")
        logger.info(f"Check interval: {interval} seconds ({interval/60:.1f} minutes)")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self.run_check()
                
                # Wait for next check
                logger.info(f"Next check in {interval} seconds...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("\n\nMonitoring stopped by user")
            logger.info(f"Total checks: {self.check_count}")
            logger.info(f"Total errors: {self.error_count}")
            logger.info(f"Success rate: {((self.check_count - self.error_count) / self.check_count * 100):.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Monitor dual-write period")
    parser.add_argument("--interval", "-i", type=int, default=3600, 
                       help="Check interval in seconds (default: 3600 = 1 hour)")
    args = parser.parse_args()
    
    monitor = DualWriteMonitor()
    monitor.monitor(interval=args.interval)

if __name__ == "__main__":
    main()

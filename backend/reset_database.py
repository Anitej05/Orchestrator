"""
Database Reset Script - Complete database recreation and migration

This script will:
1. Drop the existing database (if exists)
2. Create a fresh database
3. Run all Alembic migrations
4. Sync agent definitions to database

Use this when you need to start fresh or fix migration issues.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "root")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agentdb")

def print_step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def print_success(msg):
    print(f"✓ {msg}")

def print_error(msg):
    print(f"✗ {msg}")

def drop_database():
    """Drop the existing database"""
    print_step("Step 1: Dropping existing database")
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Terminate existing connections to the database
        cursor.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{DB_NAME}'
            AND pid <> pg_backend_pid();
        """)
        
        # Drop the database
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        print_success(f"Database '{DB_NAME}' dropped successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Failed to drop database: {e}")
        return False

def create_database():
    """Create a fresh database"""
    print_step("Step 2: Creating fresh database")
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT,
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create the database
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print_success(f"Database '{DB_NAME}' created successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Failed to create database: {e}")
        return False

def enable_pgvector():
    """Enable pgvector extension"""
    print_step("Step 3: Enabling pgvector extension")
    
    try:
        conn = psycopg2.connect(
            user=PG_USER,
            password=PG_PASSWORD,
            host=PG_HOST,
            port=PG_PORT,
            database=DB_NAME
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print_success("pgvector extension enabled")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Failed to enable pgvector: {e}")
        print("Note: pgvector may not be installed. The system will work without it.")
        print("To install pgvector, see: https://github.com/pgvector/pgvector")
        return True  # Non-critical error

def run_migrations():
    """Run Alembic migrations"""
    print_step("Step 4: Running database migrations")
    
    try:
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Migrations completed successfully")
            print(result.stdout)
            return True
        else:
            print_error("Migration failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print_error(f"Failed to run migrations: {e}")
        return False

def sync_agents():
    """Sync agent definitions to database"""
    print_step("Step 5: Syncing agent definitions")
    
    try:
        # Import and run agent sync
        from manage import sync_agents_to_db
        
        sync_agents_to_db()
        print_success("Agent definitions synced to database")
        return True
        
    except Exception as e:
        print_error(f"Failed to sync agents: {e}")
        print("You can manually sync later using: python manage.py sync_agents")
        return False

def main():
    """Main reset workflow"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           DATABASE RESET & MIGRATION TOOL                  ║
    ║                                                            ║
    ║  This will DESTROY all data in the database!              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Confirm action
    confirm = input(f"\nAre you sure you want to reset database '{DB_NAME}'? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print(f"\nTarget Database: {DB_NAME}")
    print(f"PostgreSQL Host: {PG_HOST}:{PG_PORT}")
    print(f"PostgreSQL User: {PG_USER}")
    
    # Step-by-step execution
    steps = [
        ("Drop Database", drop_database),
        ("Create Database", create_database),
        ("Enable pgvector", enable_pgvector),
        ("Run Migrations", run_migrations),
        ("Sync Agents", sync_agents)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n{'!'*60}")
            print(f"  FAILED at step: {step_name}")
            print(f"{'!'*60}")
            sys.exit(1)
    
    print_step("✓ Database reset completed successfully!")
    print("""
    Next steps:
    1. Start the backend: python main.py
    2. Start agent services: .\\start_agents.bat
    3. Test your application
    
    The database is now ready to use!
    """)

if __name__ == "__main__":
    main()

"""
Orbimesh Backend Setup Script
Automates database creation, pgvector extension, and migrations
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load environment variables
load_dotenv()

def print_step(message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_error(message):
    """Print error message"""
    print(f"✗ {message}")

def check_postgres_running():
    """Check if PostgreSQL is running"""
    print_step("Checking PostgreSQL connection...")
    
    try:
        # Try to connect to default postgres database
        conn = psycopg2.connect(
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', 'postgres'),
            host=os.getenv('PG_HOST', 'localhost'),
            port=os.getenv('PG_PORT', '5432'),
            database='postgres'
        )
        conn.close()
        print_success("PostgreSQL is running")
        return True
    except Exception as e:
        print_error(f"Cannot connect to PostgreSQL: {e}")
        print("\nPlease ensure PostgreSQL is running and credentials in .env are correct")
        return False

def create_database():
    """Create the database if it doesn't exist"""
    print_step("Creating database...")
    
    db_name = os.getenv('DB_NAME', 'agentdb')
    
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', 'postgres'),
            host=os.getenv('PG_HOST', 'localhost'),
            port=os.getenv('PG_PORT', '5432'),
            database='postgres'
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(f'CREATE DATABASE {db_name}')
            print_success(f"Database '{db_name}' created")
        else:
            print_success(f"Database '{db_name}' already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Failed to create database: {e}")
        return False

def enable_pgvector():
    """Enable pgvector extension"""
    print_step("Enabling pgvector extension...")
    
    db_name = os.getenv('DB_NAME', 'agentdb')
    
    try:
        conn = psycopg2.connect(
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', 'postgres'),
            host=os.getenv('PG_HOST', 'localhost'),
            port=os.getenv('PG_PORT', '5432'),
            database=db_name
        )
        cursor = conn.cursor()
        
        # Enable pgvector extension
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.commit()
        
        print_success("pgvector extension enabled")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"Failed to enable pgvector: {e}")
        print("\nNote: You may need to install pgvector on your system:")
        print("  - macOS: brew install pgvector")
        print("  - Linux: apt-get install postgresql-<version>-pgvector")
        print("  - Windows: Download from https://github.com/pgvector/pgvector/releases")
        return False

def run_migrations():
    """Run Alembic migrations"""
    print_step("Running database migrations...")
    
    try:
        # Check if alembic is initialized
        if not Path('alembic').exists():
            print("Initializing Alembic...")
            subprocess.run(['alembic', 'init', 'alembic'], check=True)
        
        # Run migrations
        result = subprocess.run(['alembic', 'upgrade', 'head'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            print_success("Migrations applied successfully")
            return True
        else:
            # If no migrations exist, try initializing from models
            print("No migrations found, initializing from models...")
            return initialize_from_models()
            
    except Exception as e:
        print_error(f"Migration failed: {e}")
        return False

def initialize_from_models():
    """Initialize database from models.py"""
    print_step("Initializing database from models...")
    
    try:
        # Try running db_init.py if it exists
        if Path('db_init.py').exists():
            subprocess.run([sys.executable, 'db_init.py'], check=True)
            print_success("Database initialized from models")
            return True
        else:
            print_error("db_init.py not found")
            return False
            
    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        return False

def verify_setup():
    """Verify the setup is complete"""
    print_step("Verifying setup...")
    
    try:
        # Import database to test connection
        from database import SessionLocal
        
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        print_success("Database connection verified")
        
        # Check for tables
        db = SessionLocal()
        result = db.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"
        )
        table_count = result.scalar()
        db.close()
        
        print_success(f"Found {table_count} tables in database")
        
        return True
        
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("  🚀 Orbimesh Backend Setup")
    print("="*60)
    
    # Check .env file exists
    if not Path('.env').exists():
        print_error(".env file not found")
        print("\nPlease create .env file from .env.example:")
        print("  Copy-Item .env.example .env  # Windows")
        print("  cp .env.example .env         # macOS/Linux")
        sys.exit(1)
    
    # Step 1: Check PostgreSQL
    if not check_postgres_running():
        sys.exit(1)
    
    # Step 2: Create database
    if not create_database():
        sys.exit(1)
    
    # Step 3: Enable pgvector
    if not enable_pgvector():
        sys.exit(1)
    
    # Step 4: Run migrations
    if not run_migrations():
        print("\nWarning: Migrations failed, but setup may still work")
    
    # Step 5: Verify setup
    if not verify_setup():
        sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("  ✓ Setup Complete!")
    print("="*60)
    print("\nYou can now start the backend server:")
    print("  python -m uvicorn main:app --reload")
    print("\nAPI docs will be available at:")
    print("  http://127.0.0.1:8000/docs")
    print("\n")

if __name__ == "__main__":
    main()

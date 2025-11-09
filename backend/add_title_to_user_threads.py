"""
Migration script to add title and updated_at columns to user_threads table
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# Database configuration
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agentdb")

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{DB_NAME}"

def migrate():
    """Add title and updated_at columns to user_threads table"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Check if title column exists
        check_title = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='user_threads' AND column_name='title'
        """)
        result = conn.execute(check_title)
        title_exists = result.fetchone() is not None
        
        if not title_exists:
            print("Adding 'title' column to user_threads table...")
            add_title = text("""
                ALTER TABLE user_threads 
                ADD COLUMN title VARCHAR;
            """)
            conn.execute(add_title)
            print("✓ 'title' column added successfully")
        else:
            print("'title' column already exists")
        
        # Check if updated_at column exists
        check_updated = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='user_threads' AND column_name='updated_at'
        """)
        result = conn.execute(check_updated)
        updated_exists = result.fetchone() is not None
        
        if not updated_exists:
            print("Adding 'updated_at' column to user_threads table...")
            add_updated = text("""
                ALTER TABLE user_threads 
                ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            """)
            conn.execute(add_updated)
            
            # Set updated_at to created_at for existing records
            update_existing = text("""
                UPDATE user_threads 
                SET updated_at = created_at 
                WHERE updated_at IS NULL;
            """)
            conn.execute(update_existing)
            print("✓ 'updated_at' column added successfully")
        else:
            print("'updated_at' column already exists")
        
        conn.commit()
        print("\n✓ Migration completed successfully!")

if __name__ == "__main__":
    migrate()

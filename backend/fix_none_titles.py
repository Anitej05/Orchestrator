"""
Fix "None" string titles in database - set them to NULL so API fallback works
"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

# Database configuration
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agentdb")

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{DB_NAME}"

def fix_none_titles():
    """Set all 'None' string titles to NULL"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Update all titles that are the string "None" to NULL
        update_query = text("""
            UPDATE user_threads 
            SET title = NULL 
            WHERE title = 'None' OR title = '' OR title = 'Untitled Conversation'
        """)
        
        result = conn.execute(update_query)
        conn.commit()
        
        print(f"✅ Updated {result.rowcount} titles from 'None' to NULL")

if __name__ == "__main__":
    print("Fixing 'None' string titles...")
    fix_none_titles()
    print("Done!")

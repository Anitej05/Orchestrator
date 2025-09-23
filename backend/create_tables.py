import psycopg2
from sqlalchemy import text
from database import engine, Base, DB_NAME, PG_USER, PG_PASSWORD, PG_HOST, PG_PORT

# IMPORTANT: You must import all your models here so that SQLAlchemy's
# Base class knows about them and can create the corresponding tables.
import models

def setup_database():
    """
    A standalone script to initialize the database. It:
    1. Creates the main database if it doesn't exist.
    2. Connects to the database and enables the pgvector extension.
    3. Creates all tables defined in the models.
    """
    print("--- Starting Database Setup ---")
    
    # Step 1: Connect to the default 'postgres' database to create our application DB.
    try:
        conn = psycopg2.connect(
            dbname="postgres", 
            user=PG_USER, 
            password=PG_PASSWORD, 
            host=PG_HOST, 
            port=PG_PORT
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error while creating database: {e}")
        # Exit if we can't create the database, as subsequent steps will fail.
        return

    # Step 2: Connect to our application DB to enable extensions and create tables.
    try:
        with engine.connect() as connection:
            # Enable the pgvector extension required for vector operations.
            print("Enabling pgvector extension...")
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            connection.commit()
            print("pgvector extension enabled.")

            # Create all tables that inherit from Base in your models.py.
            print("Creating all tables...")
            Base.metadata.create_all(bind=engine)
            print("✅ All tables created successfully!")
    except Exception as e:
        print(f"❌ Error during table creation or extension enabling: {e}")

if __name__ == "__main__":
    setup_database()
    print("--- Database Setup Complete ---")
# backend/migrations/add_mcp_support.py
"""
Migration script to add MCP support to the database.
Adds agent_type, connection_config columns to agents table
and creates agent_credentials table.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text
from database import engine, SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_migration():
    """Run the MCP support migration"""
    
    db = SessionLocal()
    
    try:
        logger.info("Starting MCP support migration...")
        
        # 1. Create enum types if they don't exist
        logger.info("Creating enum types...")
        db.execute(text("""
            DO $$ BEGIN
                CREATE TYPE agenttype AS ENUM ('http_rest', 'mcp_http');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """))
        
        db.execute(text("""
            DO $$ BEGIN
                CREATE TYPE authtype AS ENUM ('none', 'api_key', 'oauth2');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """))
        
        db.commit()
        logger.info("Enum types created successfully")
        
        # 2. Add new columns to agents table
        logger.info("Adding new columns to agents table...")
        
        # Check if columns already exist
        result = db.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='agents' AND column_name IN ('agent_type', 'connection_config')
        """))
        existing_columns = [row[0] for row in result]
        
        if 'agent_type' not in existing_columns:
            db.execute(text("""
                ALTER TABLE agents 
                ADD COLUMN agent_type agenttype DEFAULT 'http_rest'
            """))
            logger.info("Added agent_type column")
        else:
            logger.info("agent_type column already exists")
        
        if 'connection_config' not in existing_columns:
            db.execute(text("""
                ALTER TABLE agents 
                ADD COLUMN connection_config JSON
            """))
            logger.info("Added connection_config column")
        else:
            logger.info("connection_config column already exists")
        
        # Make public_key_pem nullable for MCP agents
        db.execute(text("""
            ALTER TABLE agents 
            ALTER COLUMN public_key_pem DROP NOT NULL
        """))
        logger.info("Made public_key_pem nullable")
        
        # Make price_per_call_usd have a default
        db.execute(text("""
            ALTER TABLE agents 
            ALTER COLUMN price_per_call_usd SET DEFAULT 0.0
        """))
        logger.info("Set default for price_per_call_usd")
        
        db.commit()
        logger.info("Agent table columns updated successfully")
        
        # 3. Create agent_credentials table
        logger.info("Creating agent_credentials table...")
        
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_credentials (
                id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                agent_id VARCHAR NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
                auth_type authtype DEFAULT 'none',
                encrypted_access_token TEXT,
                encrypted_refresh_token TEXT,
                auth_header_name VARCHAR DEFAULT 'Authorization',
                token_expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create indexes
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_credentials_user_id 
            ON agent_credentials(user_id)
        """))
        
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_credentials_agent_id 
            ON agent_credentials(agent_id)
        """))
        
        db.commit()
        logger.info("agent_credentials table created successfully")
        
        logger.info("✅ MCP support migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_migration()

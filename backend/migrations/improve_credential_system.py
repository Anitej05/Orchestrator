"""
Migration: Improve Agent Credential System
- Add support for multiple credential fields per agent
- Add credential field definitions in agent config
- Support both REST and MCP agents uniformly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database import engine

def upgrade():
    """Add new columns and tables for improved credential management"""
    
    with engine.connect() as conn:
        # Add credential_fields to agents table (JSON field defining what credentials are needed)
        try:
            conn.execute(text("""
                ALTER TABLE agents 
                ADD COLUMN IF NOT EXISTS credential_fields JSON DEFAULT '[]'::json;
            """))
            conn.commit()
            print("✅ Added credential_fields column to agents table")
        except Exception as e:
            print(f"⚠️  credential_fields column might already exist: {e}")
            conn.rollback()
        
        # Add encrypted_credentials JSON field to agent_credentials table
        # This allows storing multiple key-value pairs instead of just access_token
        try:
            conn.execute(text("""
                ALTER TABLE agent_credentials 
                ADD COLUMN IF NOT EXISTS encrypted_credentials JSON DEFAULT '{}'::json;
            """))
            conn.commit()
            print("✅ Added encrypted_credentials column to agent_credentials table")
        except Exception as e:
            print(f"⚠️  encrypted_credentials column might already exist: {e}")
            conn.rollback()
        
        # Add is_active flag to agent_credentials table
        try:
            conn.execute(text("""
                ALTER TABLE agent_credentials 
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;
            """))
            conn.commit()
            print("✅ Added is_active column to agent_credentials table")
        except Exception as e:
            print(f"⚠️  is_active column might already exist: {e}")
            conn.rollback()
        
        # Add is_configured flag to agents table
        try:
            conn.execute(text("""
                ALTER TABLE agents 
                ADD COLUMN IF NOT EXISTS requires_credentials BOOLEAN DEFAULT false;
            """))
            conn.commit()
            print("✅ Added requires_credentials column to agents table")
        except Exception as e:
            print(f"⚠️  requires_credentials column might already exist: {e}")
            conn.rollback()
        
        print("\n✅ Migration completed successfully!")
        print("\nNext steps:")
        print("1. Update agent JSON files to include credential_fields")
        print("2. Use the new credential management UI")
        print("3. Migrate existing credentials to new format")

def downgrade():
    """Remove the new columns (if needed)"""
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE agents DROP COLUMN IF EXISTS credential_fields;"))
        conn.execute(text("ALTER TABLE agents DROP COLUMN IF EXISTS requires_credentials;"))
        conn.execute(text("ALTER TABLE agent_credentials DROP COLUMN IF EXISTS encrypted_credentials;"))
        conn.commit()
        print("✅ Migration rolled back")

if __name__ == "__main__":
    print("Running credential system improvement migration...")
    upgrade()

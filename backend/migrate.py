#!/usr/bin/env python
"""
Database Migration Helper Script

Usage:
    python migrate.py generate "description"  - Generate a new migration
    python migrate.py upgrade                 - Apply all pending migrations
    python migrate.py downgrade               - Rollback one migration
    python migrate.py current                 - Show current migration
    python migrate.py history                 - Show migration history
"""

import sys
import subprocess

def run_alembic(args):
    """Run alembic command"""
    cmd = ["alembic"] + args
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide a migration description")
            print("Usage: python migrate.py generate 'add user table'")
            sys.exit(1)
        
        description = sys.argv[2]
        print(f"🔍 Generating migration: {description}")
        return run_alembic(["revision", "--autogenerate", "-m", description])
    
    elif command == "upgrade":
        print("⬆️  Applying migrations...")
        return run_alembic(["upgrade", "head"])
    
    elif command == "downgrade":
        print("⬇️  Rolling back last migration...")
        return run_alembic(["downgrade", "-1"])
    
    elif command == "current":
        print("📍 Current migration:")
        return run_alembic(["current"])
    
    elif command == "history":
        print("📜 Migration history:")
        return run_alembic(["history"])
    
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())

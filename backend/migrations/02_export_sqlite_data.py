#!/usr/bin/env python3
"""
Migration Script 02: Export SQLite Data

This script exports all data from the SQLite database to JSON format.
The export includes metadata for validation and can be used for import to PostgreSQL.

Usage:
    python 02_export_sqlite_data.py [--output <file>]

Environment Variables Required:
    DATABASE_URL - SQLite connection string (default: sqlite:///orbimesh.db)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import create_engine, inspect
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

def serialize_row(row: Any) -> Dict[str, Any]:
    """Convert SQLAlchemy row to JSON-serializable dict."""
    result = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        
        # Handle datetime objects
        if isinstance(value, datetime):
            result[column.name] = value.isoformat()
        # Handle None
        elif value is None:
            result[column.name] = None
        # Handle other types
        else:
            result[column.name] = value
    
    return result

def export_table(session, table_name: str) -> List[Dict[str, Any]]:
    """Export all rows from a table."""
    from models import Base
    
    # Get table class
    table_class = None
    for mapper in Base.registry.mappers:
        if mapper.class_.__tablename__ == table_name:
            table_class = mapper.class_
            break
    
    if not table_class:
        logger.warning(f"Table class not found for {table_name}")
        return []
    
    # Query all rows
    rows = session.query(table_class).all()
    
    # Serialize rows
    return [serialize_row(row) for row in rows]

def export_sqlite_data(output_file: str = None):
    """Export all data from SQLite to JSON."""
    
    # Get SQLite URL
    database_url = os.getenv("DATABASE_URL", "sqlite:///orbimesh.db")
    
    # Ensure it's SQLite
    if not database_url.startswith("sqlite"):
        logger.error(f"DATABASE_URL must be SQLite, got: {database_url}")
        sys.exit(1)
    
    # Handle relative paths
    if database_url.startswith("sqlite:///"):
        db_file = database_url.replace("sqlite:///", "")
        if not os.path.isabs(db_file):
            db_path = os.path.abspath(os.path.join(backend_dir, db_file))
            database_url = f"sqlite:///{db_path}"
    
    logger.info(f"Exporting from SQLite: {database_url}")
    
    try:
        # Create engine and session
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get all table names
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"Found {len(tables)} tables to export")
        
        # Export data
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "database_version": "1.0",
            "source_database": "sqlite",
            "tables": {},
            "row_counts": {}
        }
        
        total_rows = 0
        for table_name in sorted(tables):
            logger.info(f"Exporting table: {table_name}")
            
            try:
                rows = export_table(session, table_name)
                export_data["tables"][table_name] = rows
                export_data["row_counts"][table_name] = len(rows)
                total_rows += len(rows)
                
                logger.info(f"  ✓ Exported {len(rows)} rows from {table_name}")
            
            except Exception as e:
                logger.error(f"  ✗ Failed to export {table_name}: {e}")
                export_data["tables"][table_name] = []
                export_data["row_counts"][table_name] = 0
        
        session.close()
        
        # Determine output file
        if not output_file:
            data_dir = Path(__file__).parent / "data"
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = data_dir / f"export_{timestamp}.json"
        
        # Write to file
        logger.info(f"\nWriting export to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Write metadata
        metadata_file = Path(output_file).parent / "export_metadata.json"
        metadata = {
            "export_file": str(output_file),
            "export_timestamp": export_data["export_timestamp"],
            "total_tables": len(tables),
            "total_rows": total_rows,
            "row_counts": export_data["row_counts"]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("\n" + "="*60)
        logger.info("SQLite data export completed successfully!")
        logger.info(f"Total tables: {len(tables)}")
        logger.info(f"Total rows: {total_rows}")
        logger.info(f"Export file: {output_file}")
        logger.info(f"Metadata file: {metadata_file}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export SQLite data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Export SQLite data to JSON")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()
    
    success = export_sqlite_data(args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

"""
Display utilities for spreadsheet data.

Handles conversion of DataFrames to canvas display format.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def dataframe_to_canvas(
    df: pd.DataFrame,
    title: str,
    filename: str,
    display_mode: str = 'full',
    max_rows: int = 100,
    file_id: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convert dataframe to canvas display format.
    
    Args:
        df: DataFrame to display
        title: Display title
        filename: Source filename
        display_mode: Display mode ('full', 'preview', etc.)
        max_rows: Maximum rows to display
        file_id: Optional file ID
        metadata: Optional additional metadata
    
    Returns:
        Canvas display dict with data and metadata
    """
    try:
        # Import canvas utilities
        try:
            from agents.utils.canvas_utils import create_spreadsheet_display
        except ImportError:
            try:
                from utils.canvas_utils import create_spreadsheet_display
            except ImportError:
                logger.error("Failed to import canvas_utils")
                # Fallback: return basic structure
                return {
                    'type': 'spreadsheet',
                    'title': title,
                    'filename': filename,
                    'data': df.head(max_rows).to_dict('records'),
                    'error': 'Canvas utils not available'
                }
        
        # Limit rows for display
        display_df = df.head(max_rows) if len(df) > max_rows else df
        
        # Convert to 2D array: [headers, row1, row2, ...]
        data = [display_df.columns.tolist()] + display_df.values.tolist()
        
        # Add metadata
        full_metadata = {
            'rows_total': len(df),
            'rows_shown': len(display_df),
            'columns': len(df.columns),
            'truncated': len(df) > max_rows
        }
        if metadata:
            full_metadata.update(metadata)
        
        return create_spreadsheet_display(
            data=data,
            title=title,
            filename=filename,
            display_mode=display_mode,
            metadata=full_metadata,
            file_id=file_id
        )
    
    except Exception as e:
        logger.error(f"Failed to create canvas display: {e}")
        # Return basic structure as fallback
        return {
            'type': 'spreadsheet',
            'title': title,
            'filename': filename,
            'data': df.head(max_rows).to_dict('records') if len(df) > 0 else [],
            'error': str(e)
        }


def format_dataframe_preview(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    """
    Format dataframe as a preview dict.
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum rows to include
    
    Returns:
        Dict with preview data
    """
    preview_df = df.head(max_rows)
    
    return {
        'columns': df.columns.tolist(),
        'data': preview_df.to_dict('records'),
        'total_rows': len(df),
        'preview_rows': len(preview_df),
        'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()}
    }

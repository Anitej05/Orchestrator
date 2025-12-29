"""
Data utilities: Validation and conversion functions
"""
import io
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from .core_utils import FileLoadError, SpreadsheetError

logger = logging.getLogger(__name__)


# ============== FILE VALIDATION ==============

def validate_file(file, max_size_mb: int = 50) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file
    
    Args:
        file: UploadFile object
        max_size_mb: Maximum file size in MB
    
    Returns:
        (is_valid, error_message)
    """
    # Check file extension
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ['.csv', '.xlsx', '.xls']):
        return False, "Invalid file type. Only CSV and Excel files are supported."
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to start
    
    if size > max_size_mb * 1024 * 1024:
        return False, f"File too large. Maximum size is {max_size_mb}MB."
    
    return True, None


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate dataframe integrity
    
    Args:
        df: DataFrame to validate
    
    Returns:
        (is_valid, error_message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df.columns) == 0:
        return False, "DataFrame has no columns"
    
    return True, None


def validate_column_names(df: pd.DataFrame, columns: list) -> Tuple[bool, Optional[str]]:
    """
    Check if columns exist in dataframe
    
    Args:
        df: DataFrame to check
        columns: List of column names
    
    Returns:
        (are_valid, error_message)
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        return False, f"Columns not found: {', '.join(missing)}"
    return True, None


# ============== FILE CONVERSION ==============

def csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load CSV file to DataFrame
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Pandas DataFrame
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise FileLoadError(f"Failed to load CSV: {str(e)}")


def excel_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load Excel file to DataFrame
    
    Args:
        file_path: Path to Excel file
    
    Returns:
        Pandas DataFrame
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        raise FileLoadError(f"Failed to load Excel: {str(e)}")


def dataframe_to_csv(df: pd.DataFrame, file_path: str):
    """
    Export DataFrame to CSV
    
    Args:
        df: DataFrame to export
        file_path: Output file path
    """
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise SpreadsheetError(f"Failed to export CSV: {str(e)}")


def dataframe_to_excel(df: pd.DataFrame, file_path: str):
    """
    Export DataFrame to Excel
    
    Args:
        df: DataFrame to export
        file_path: Output file path
    """
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
    except Exception as e:
        raise SpreadsheetError(f"Failed to export Excel: {str(e)}")


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    Export DataFrame to JSON string
    
    Args:
        df: DataFrame to export
    
    Returns:
        JSON string
    """
    try:
        return df.to_json(orient='records')
    except Exception as e:
        raise SpreadsheetError(f"Failed to export JSON: {str(e)}")


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load dataframe from file (auto-detects format)
    
    Args:
        file_path: Path to file
    
    Returns:
        Pandas DataFrame
    """
    file_path_lower = file_path.lower()
    
    if file_path_lower.endswith('.csv'):
        return csv_to_dataframe(file_path)
    elif file_path_lower.endswith(('.xlsx', '.xls')):
        return excel_to_dataframe(file_path)
    else:
        raise FileLoadError(f"Unsupported file format: {file_path}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names (lowercase, underscores)
    
    Args:
        df: DataFrame to normalize
    
    Returns:
        DataFrame with normalized column names
    """
    df.columns = [
        col.lower().strip().replace(' ', '_').replace('-', '_')
        for col in df.columns
    ]
    return df


def is_valid_csv(content: str) -> bool:
    """
    Check if content is valid CSV format
    
    Args:
        content: String content to check
    
    Returns:
        True if valid CSV
    """
    try:
        # Basic checks
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check if first line looks like headers
        first_line = lines[0]
        if ',' not in first_line and '\t' not in first_line:
            return False
        
        # Try to parse as CSV
        df = pd.read_csv(io.StringIO(content))
        return len(df) > 0 and len(df.columns) > 0
    except:
        return False

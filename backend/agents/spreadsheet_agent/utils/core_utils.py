"""
Core utilities: Error handling and numpy serialization
"""
import json
import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ============== CUSTOM EXCEPTIONS ==============

class SpreadsheetError(Exception):
    """Base exception for spreadsheet operations"""
    pass


class FileLoadError(SpreadsheetError):
    """Error loading file"""
    pass


class QueryExecutionError(SpreadsheetError):
    """Error executing query"""
    pass


class CodeGenerationError(SpreadsheetError):
    """Error generating code"""
    pass


# ============== NUMPY JSON ENCODER ==============

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types
    
    Args:
        obj: Object to convert
    
    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def serialize_dataframe(df) -> Dict[str, Any]:
    """
    Serialize dataframe to JSON-compatible dict
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        Dictionary with native Python types
    """
    return {
        'data': convert_numpy_types(df.to_dict('records')),
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()}
    }


# ============== ERROR HANDLING ==============

def handle_execution_error(error: Exception, operation: str = "operation") -> str:
    """
    Handle execution errors gracefully and return user-friendly message
    
    Args:
        error: The exception that occurred
        operation: Description of the operation
    
    Returns:
        User-friendly error message
    """
    error_msg = str(error)
    
    # Common pandas errors
    if "KeyError" in error_msg:
        return f"Column not found. Please check column names and try again."
    elif "SyntaxError" in error_msg:
        return f"Invalid pandas syntax. Please check your query."
    elif "NameError" in error_msg:
        return f"Variable not defined. Make sure you're using 'df' for the dataframe."
    elif "TypeError" in error_msg:
        return f"Type mismatch in operation. Check data types and operations."
    elif "ValueError" in error_msg:
        return f"Invalid value in operation: {error_msg}"
    else:
        return f"Error during {operation}: {error_msg}"


def log_operation_error(operation: str, error: Exception, context: Dict[str, Any] = None):
    """
    Log operation error with context
    
    Args:
        operation: Operation that failed
        error: Exception that occurred
        context: Additional context (file_id, code, etc.)
    """
    logger.error(f"[{operation}] Error: {str(error)}")
    if context:
        logger.error(f"[{operation}] Context: {context}")
    logger.exception(error)


def format_error_message(error: Exception, user_friendly: bool = True) -> str:
    """
    Format error message for display
    
    Args:
        error: Exception to format
        user_friendly: If True, return user-friendly message
    
    Returns:
        Formatted error message
    """
    if user_friendly:
        return handle_execution_error(error)
    return f"{type(error).__name__}: {str(error)}"

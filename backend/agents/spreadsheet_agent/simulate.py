"""
Simulation module for safe preview of spreadsheet operations.

Executes pandas code on a copy of the DataFrame and returns detailed
preview information before applying changes to the actual data.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from difflib import get_close_matches

from .models import ObservationData

logger = logging.getLogger(__name__)


class SimulationResult:
    """Result of a simulated operation"""
    def __init__(
        self,
        success: bool,
        preview_df: Optional[pd.DataFrame] = None,
        observation: Optional[ObservationData] = None,
        error: Optional[str] = None,
        warnings: Optional[List[str]] = None
    ):
        self.success = success
        self.preview_df = preview_df
        self.observation = observation
        self.error = error
        self.warnings = warnings or []


def validate_code_against_schema(df: pd.DataFrame, code: str) -> Optional[str]:
    """
    Validate pandas code against DataFrame schema before execution.
    Returns error message if validation fails, None if valid.
    """
    import re
    
    actual_columns = set(df.columns)
    
    # Extract column references from code
    patterns = [
        r"df\['([^']+)'\]",  # df['column']
        r'df\["([^"]+)"\]',  # df["column"]
        r"df\.([a-zA-Z_][a-zA-Z0-9_]*)",  # df.column
    ]
    
    referenced_columns = set()
    for pattern in patterns:
        matches = re.findall(pattern, code)
        referenced_columns.update(matches)
    
    # Filter out pandas methods (not actual columns)
    pandas_methods = {
        'head', 'tail', 'describe', 'info', 'shape', 'columns', 'dtypes', 
        'groupby', 'sort_values', 'query', 'drop', 'rename', 'insert',
        'select_dtypes', 'sum', 'mean', 'count', 'max', 'min', 'fillna',
        'dropna', 'isnull', 'notnull', 'merge', 'join', 'to_dict', 'copy',
        'apply', 'map', 'filter', 'pivot', 'melt', 'stack', 'unstack'
    }
    referenced_columns = {col for col in referenced_columns if col not in pandas_methods}
    
    # Check for non-existent columns
    missing_columns = referenced_columns - actual_columns
    
    if missing_columns:
        # Find similar column names
        suggestions = {}
        for missing_col in missing_columns:
            matches = get_close_matches(missing_col, actual_columns, n=3, cutoff=0.6)
            if matches:
                suggestions[missing_col] = matches
        
        error_msg = f"Column(s) not found: {', '.join(missing_columns)}. "
        error_msg += f"Available columns: {', '.join(sorted(actual_columns))}. "
        if suggestions:
            error_msg += "Did you mean: "
            error_msg += ", ".join([f"'{missing}' -> {matches}" for missing, matches in suggestions.items()])
        
        return error_msg
    
    return None


def detect_potential_issues(
    original_df: pd.DataFrame, 
    modified_df: pd.DataFrame
) -> List[str]:
    """
    Detect potential issues with the transformation.
    Returns list of warning messages.
    """
    warnings = []
    
    # Check for significant data loss
    original_rows = len(original_df)
    modified_rows = len(modified_df)
    
    if modified_rows < original_rows * 0.5:
        loss_pct = ((original_rows - modified_rows) / original_rows) * 100
        warnings.append(f"⚠️ Significant data loss: {loss_pct:.1f}% of rows removed ({original_rows} → {modified_rows})")
    
    # Check for null value introduction
    original_nulls = original_df.isnull().sum().sum()
    modified_nulls = modified_df.isnull().sum().sum()
    
    if modified_nulls > original_nulls * 1.2:  # More than 20% increase in nulls
        warnings.append(f"⚠️ Null values increased: {original_nulls} → {modified_nulls}")
    
    # Check for dtype changes that might indicate issues
    common_cols = set(original_df.columns) & set(modified_df.columns)
    for col in common_cols:
        orig_dtype = original_df[col].dtype
        mod_dtype = modified_df[col].dtype
        
        if orig_dtype != mod_dtype:
            # Warn about potentially lossy conversions
            if pd.api.types.is_numeric_dtype(orig_dtype) and not pd.api.types.is_numeric_dtype(mod_dtype):
                warnings.append(f"⚠️ Column '{col}' converted from numeric ({orig_dtype}) to non-numeric ({mod_dtype})")
            elif pd.api.types.is_datetime64_any_dtype(orig_dtype) and not pd.api.types.is_datetime64_any_dtype(mod_dtype):
                warnings.append(f"⚠️ Column '{col}' converted from datetime to {mod_dtype}")
    
    # Check for complete data removal
    if len(modified_df) == 0 and len(original_df) > 0:
        warnings.append("⚠️ CRITICAL: All data removed! The resulting DataFrame is empty.")
    
    # Check for duplicate columns
    if len(modified_df.columns) != len(set(modified_df.columns)):
        warnings.append("⚠️ Duplicate column names detected in result")
    
    return warnings


def simulate_operation(
    df: pd.DataFrame, 
    code: str,
    validate_first: bool = True
) -> SimulationResult:
    """
    Simulate a pandas operation on a copy of the DataFrame.
    
    Args:
        df: Original DataFrame (will not be modified)
        code: Pandas code to execute
        validate_first: Whether to validate schema before execution
    
    Returns:
        SimulationResult with preview and analysis
    """
    logger.info(f"🧪 Simulating operation on {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Pre-execution validation
    if validate_first:
        validation_error = validate_code_against_schema(df, code)
        if validation_error:
            logger.warning(f"❌ Simulation validation failed: {validation_error}")
            return SimulationResult(
                success=False,
                error=validation_error
            )
    
    # Work on a copy
    df_copy = df.copy()
    
    # Capture before state
    before_shape = df_copy.shape
    before_columns = df_copy.columns.tolist()
    before_sample = df_copy.head(3).to_dict(orient="records") if len(df_copy) > 0 else []
    
    try:
        # Execute in isolated environment
        local_vars = {"df": df_copy, "pd": pd}
        exec(code, {"__builtins__": {}}, local_vars)
        
        # Get modified DataFrame
        modified_df = local_vars.get("df", df_copy)
        
        # Ensure it's a DataFrame
        if not isinstance(modified_df, pd.DataFrame):
            # If result is a Series or other type, try to convert
            if isinstance(modified_df, pd.Series):
                modified_df = modified_df.to_frame()
            else:
                return SimulationResult(
                    success=False,
                    error=f"Operation returned {type(modified_df).__name__} instead of DataFrame"
                )
        
        # Capture after state
        after_shape = modified_df.shape
        after_columns = modified_df.columns.tolist()
        after_sample = modified_df.head(3).to_dict(orient="records") if len(modified_df) > 0 else []
        
        # Calculate changes
        cols_added = list(set(after_columns) - set(before_columns))
        cols_removed = list(set(before_columns) - set(after_columns))
        rows_change = after_shape[0] - before_shape[0]
        
        # Build changes summary
        changes_parts = []
        if cols_added:
            changes_parts.append(f"Added {len(cols_added)} column(s): {', '.join(cols_added)}")
        if cols_removed:
            changes_parts.append(f"Removed {len(cols_removed)} column(s): {', '.join(cols_removed)}")
        if rows_change > 0:
            changes_parts.append(f"Added {rows_change} rows")
        elif rows_change < 0:
            changes_parts.append(f"Removed {abs(rows_change)} rows")
        
        # Check if data was modified without structural changes
        data_modified = False
        if before_shape == after_shape and before_columns == after_columns:
            # Quick check: compare first few rows
            if not df.head(5).equals(modified_df.head(5)):
                data_modified = True
                changes_parts.append("Data values modified")
        
        changes_summary = "; ".join(changes_parts) if changes_parts else "No changes detected"
        
        # Create observation data
        observation = ObservationData(
            before_shape=before_shape,
            after_shape=after_shape,
            before_columns=before_columns,
            after_columns=after_columns,
            before_sample=before_sample,
            after_sample=after_sample,
            changes_summary=changes_summary,
            columns_added=cols_added,
            columns_removed=cols_removed,
            rows_added=max(0, rows_change),
            rows_removed=max(0, -rows_change),
            data_modified=data_modified
        )
        
        # Detect potential issues
        warnings = detect_potential_issues(df, modified_df)
        
        logger.info(f"✅ Simulation successful: {changes_summary}")
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        return SimulationResult(
            success=True,
            preview_df=modified_df,
            observation=observation,
            warnings=warnings
        )
        
    except KeyError as e:
        # Enhanced KeyError handling
        column_name = str(e).strip("'\"")
        similar = get_close_matches(column_name, df.columns, n=3, cutoff=0.6)
        error_msg = f"Column '{column_name}' not found. Available: {', '.join(df.columns)}."
        if similar:
            error_msg += f" Did you mean: {', '.join(similar)}?"
        
        logger.error(f"❌ Simulation failed: {error_msg}")
        return SimulationResult(
            success=False,
            error=error_msg
        )
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {str(e)}", exc_info=True)
        return SimulationResult(
            success=False,
            error=str(e)
        )


def preview_operation(
    df: pd.DataFrame,
    code: str,
    max_preview_rows: int = 10
) -> Dict[str, Any]:
    """
    Execute operation in simulation mode and return formatted preview.
    
    Returns:
        Dict with success, preview data, changes summary, and warnings
    """
    result = simulate_operation(df, code, validate_first=True)
    
    response = {
        "success": result.success,
        "error": result.error,
        "warnings": result.warnings
    }
    
    if result.success and result.preview_df is not None:
        response["preview"] = {
            "shape": result.preview_df.shape,
            "columns": result.preview_df.columns.tolist(),
            "rows": result.preview_df.head(max_preview_rows).to_dict(orient="records"),
            "dtypes": {k: str(v) for k, v in result.preview_df.dtypes.to_dict().items()}
        }
        
        if result.observation:
            response["changes"] = {
                "summary": result.observation.changes_summary,
                "columns_added": result.observation.columns_added,
                "columns_removed": result.observation.columns_removed,
                "rows_added": result.observation.rows_added,
                "rows_removed": result.observation.rows_removed,
                "before_shape": result.observation.before_shape,
                "after_shape": result.observation.after_shape
            }
    
    return response

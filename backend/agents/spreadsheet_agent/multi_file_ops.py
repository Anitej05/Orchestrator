"""
Multi-file operations for spreadsheet agent.

Handles comparing, merging, and creating derived artifacts from multiple spreadsheets.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from difflib import get_close_matches

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA COMPARISON
# ============================================================================

def compare_schemas(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compare schemas of multiple dataframes.
    
    Args:
        dataframes: Dict mapping file_id -> DataFrame
    
    Returns:
        Schema comparison result with columns, dtypes, shapes
    """
    file_ids = list(dataframes.keys())
    schemas = {}
    
    for file_id, df in dataframes.items():
        schemas[file_id] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    # Find common and unique columns
    all_columns = set()
    for schema in schemas.values():
        all_columns.update(schema["columns"])
    
    common_columns = set(schemas[file_ids[0]]["columns"])
    for file_id in file_ids[1:]:
        common_columns &= set(schemas[file_id]["columns"])
    
    unique_columns = {}
    for file_id in file_ids:
        file_cols = set(schemas[file_id]["columns"])
        unique_columns[file_id] = list(file_cols - common_columns)
    
    # Check for dtype mismatches in common columns
    dtype_mismatches = {}
    for col in common_columns:
        dtypes_for_col = {file_id: schemas[file_id]["dtypes"].get(col) 
                         for file_id in file_ids}
        if len(set(dtypes_for_col.values())) > 1:
            dtype_mismatches[col] = dtypes_for_col
    
    return {
        "file_ids": file_ids,
        "schemas": schemas,
        "common_columns": list(common_columns),
        "unique_columns": unique_columns,
        "dtype_mismatches": dtype_mismatches,
        "summary": _generate_schema_summary(schemas, common_columns, unique_columns, dtype_mismatches)
    }


def _generate_schema_summary(schemas, common_columns, unique_columns, dtype_mismatches) -> str:
    """Generate human-readable schema comparison summary."""
    lines = []
    lines.append(f"Comparing {len(schemas)} files:")
    
    for file_id, schema in schemas.items():
        lines.append(f"  - {file_id}: {schema['shape'][0]} rows × {schema['shape'][1]} columns")
    
    lines.append(f"\nCommon columns: {len(common_columns)}")
    
    has_unique = any(cols for cols in unique_columns.values())
    if has_unique:
        lines.append("\nUnique columns per file:")
        for file_id, cols in unique_columns.items():
            if cols:
                lines.append(f"  - {file_id}: {', '.join(cols)}")
    
    if dtype_mismatches:
        lines.append(f"\nData type mismatches: {len(dtype_mismatches)} columns")
        for col, dtypes in list(dtype_mismatches.items())[:5]:  # Show first 5
            lines.append(f"  - {col}: {dtypes}")
    
    return "\n".join(lines)


# ============================================================================
# KEY-BASED ROW COMPARISON
# ============================================================================

def compare_by_keys(
    dataframes: Dict[str, pd.DataFrame],
    key_columns: List[str],
    comparison_mode: str = "schema_and_key"
) -> Dict[str, Any]:
    """
    Compare dataframes by key columns to identify added/removed/changed rows.
    
    Args:
        dataframes: Dict mapping file_id -> DataFrame
        key_columns: Columns to use as keys
        comparison_mode: 'schema_and_key' or 'full_diff'
    
    Returns:
        Row-level comparison results
    """
    file_ids = list(dataframes.keys())
    
    if len(file_ids) != 2:
        # For >2 files, do pairwise comparisons
        return _compare_multiple_by_keys(dataframes, key_columns, comparison_mode)
    
    # Two-file comparison
    df1_id, df2_id = file_ids
    df1 = dataframes[df1_id].copy()
    df2 = dataframes[df2_id].copy()
    
    # Validate key columns exist
    for col in key_columns:
        if col not in df1.columns:
            raise ValueError(f"Key column '{col}' not found in {df1_id}")
        if col not in df2.columns:
            raise ValueError(f"Key column '{col}' not found in {df2_id}")
    
    # Create composite key
    df1['_key'] = df1[key_columns].astype(str).agg('||'.join, axis=1)
    df2['_key'] = df2[key_columns].astype(str).agg('||'.join, axis=1)
    
    keys1 = set(df1['_key'])
    keys2 = set(df2['_key'])
    
    added_keys = keys2 - keys1
    removed_keys = keys1 - keys2
    common_keys = keys1 & keys2
    
    # For common keys, check if values changed
    changed_rows = []
    if comparison_mode == "full_diff" and common_keys:
        df1_indexed = df1.set_index('_key')
        df2_indexed = df2.set_index('_key')
        
        for key in list(common_keys)[:100]:  # Limit to first 100 for performance
            row1 = df1_indexed.loc[key]
            row2 = df2_indexed.loc[key]
            
            if not row1.equals(row2):
                changed_fields = []
                for col in df1.columns:
                    if col == '_key':
                        continue
                    if col in df2.columns:
                        val1 = row1.get(col)
                        val2 = row2.get(col)
                        if pd.notna(val1) and pd.notna(val2) and val1 != val2:
                            changed_fields.append({
                                "column": col,
                                "old_value": val1,
                                "new_value": val2
                            })
                
                if changed_fields:
                    changed_rows.append({
                        "key": key,
                        "key_values": {k: row1[k] for k in key_columns},
                        "changed_fields": changed_fields
                    })
    
    return {
        "file_ids": file_ids,
        "key_columns": key_columns,
        "added_rows": len(added_keys),
        "removed_rows": len(removed_keys),
        "common_rows": len(common_keys),
        "changed_rows": len(changed_rows),
        "changed_rows_detail": changed_rows[:20],  # Return first 20 for preview
        "summary": _generate_row_diff_summary(
            df1_id, df2_id, len(added_keys), len(removed_keys), len(common_keys), len(changed_rows)
        )
    }


def _compare_multiple_by_keys(dataframes, key_columns, comparison_mode) -> Dict[str, Any]:
    """Compare >2 files by doing pairwise comparisons."""
    file_ids = list(dataframes.keys())
    pairwise_results = []
    
    for i in range(len(file_ids) - 1):
        pair_dict = {
            file_ids[i]: dataframes[file_ids[i]],
            file_ids[i+1]: dataframes[file_ids[i+1]]
        }
        result = compare_by_keys(pair_dict, key_columns, comparison_mode)
        pairwise_results.append(result)
    
    return {
        "file_ids": file_ids,
        "key_columns": key_columns,
        "pairwise_comparisons": pairwise_results,
        "summary": f"Compared {len(file_ids)} files in {len(pairwise_results)} pairwise comparisons"
    }


def _generate_row_diff_summary(file1_id, file2_id, added, removed, common, changed) -> str:
    """Generate human-readable row diff summary."""
    lines = []
    lines.append(f"Row comparison: {file1_id} vs {file2_id}")
    lines.append(f"  Rows in both: {common}")
    if added > 0:
        lines.append(f"  Rows added in {file2_id}: {added}")
    if removed > 0:
        lines.append(f"  Rows removed from {file1_id}: {removed}")
    if changed > 0:
        lines.append(f"  Rows with changes: {changed}")
    
    return "\n".join(lines)


# ============================================================================
# SMART KEY DETECTION
# ============================================================================

def detect_key_columns(df: pd.DataFrame, max_keys: int = 3) -> List[str]:
    """
    Heuristically detect likely key columns.
    
    Looks for columns with high uniqueness, common key names, or index-like properties.
    """
    candidates = []
    
    # Common key column names
    key_patterns = ['id', 'key', 'code', 'number', 'serial', 'invoice', 'order', 'customer', 'product']
    
    for col in df.columns:
        col_lower = col.lower()
        uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0
        
        # Score based on uniqueness and name
        score = uniqueness
        if any(pattern in col_lower for pattern in key_patterns):
            score += 0.5
        
        candidates.append((col, score, uniqueness))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return top candidates with >80% uniqueness or name match
    keys = []
    for col, score, uniqueness in candidates[:max_keys]:
        if uniqueness > 0.8 or score > 1.0:
            keys.append(col)
    
    return keys


# ============================================================================
# MERGE OPERATIONS
# ============================================================================

def merge_dataframes(
    dataframes: Dict[str, pd.DataFrame],
    merge_type: str,
    join_type: str = "inner",
    key_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Merge multiple dataframes.
    
    Args:
        dataframes: Dict mapping file_id -> DataFrame
        merge_type: 'join', 'union', or 'concat'
        join_type: For join: 'inner', 'left', 'right', 'outer'
        key_columns: Columns to join on (required for 'join')
    
    Returns:
        (merged_df, summary_message)
    """
    file_ids = list(dataframes.keys())
    dfs = list(dataframes.values())
    
    if merge_type == "concat":
        # Simple vertical concatenation
        result = pd.concat(dfs, ignore_index=True)
        summary = f"Concatenated {len(dfs)} files vertically: {result.shape[0]} rows × {result.shape[1]} columns"
    
    elif merge_type == "union":
        # Union: concat + drop duplicates
        result = pd.concat(dfs, ignore_index=True).drop_duplicates()
        summary = f"Union of {len(dfs)} files (removed duplicates): {result.shape[0]} rows × {result.shape[1]} columns"
    
    elif merge_type == "join":
        if not key_columns:
            raise ValueError("key_columns required for join operation")
        
        # Sequential join
        result = dfs[0].copy()
        for i, df in enumerate(dfs[1:], 1):
            result = result.merge(df, on=key_columns, how=join_type, suffixes=('', f'_{file_ids[i]}'))
        
        summary = f"Joined {len(dfs)} files on {key_columns} ({join_type} join): {result.shape[0]} rows × {result.shape[1]} columns"
    
    else:
        raise ValueError(f"Unknown merge_type: {merge_type}")
    
    return result, summary


# ============================================================================
# DIFF REPORT GENERATION
# ============================================================================

def generate_diff_report(
    schema_diff: Dict[str, Any],
    row_diff: Optional[Dict[str, Any]] = None,
    format: str = "json"
) -> str:
    """
    Generate a formatted diff report.
    
    Args:
        schema_diff: Result from compare_schemas
        row_diff: Optional result from compare_by_keys
        format: 'json', 'csv', or 'markdown'
    
    Returns:
        Formatted report string
    """
    if format == "json":
        import json
        report = {
            "schema_diff": schema_diff,
            "row_diff": row_diff
        }
        return json.dumps(report, indent=2)
    
    elif format == "markdown":
        lines = ["# Spreadsheet Comparison Report", ""]
        lines.append("## Schema Comparison")
        lines.append(schema_diff.get("summary", ""))
        lines.append("")
        
        if row_diff:
            lines.append("## Row Comparison")
            lines.append(row_diff.get("summary", ""))
            lines.append("")
            
            if row_diff.get("changed_rows_detail"):
                lines.append("### Sample Changes")
                for change in row_diff["changed_rows_detail"][:10]:
                    lines.append(f"- Key: {change['key']}")
                    for field in change["changed_fields"][:3]:
                        lines.append(f"  - {field['column']}: {field['old_value']} → {field['new_value']}")
        
        return "\n".join(lines)
    
    elif format == "csv":
        # Simple CSV summary
        import io
        output = io.StringIO()
        output.write("metric,value\n")
        output.write(f"files_compared,{len(schema_diff['file_ids'])}\n")
        output.write(f"common_columns,{len(schema_diff['common_columns'])}\n")
        
        if row_diff:
            output.write(f"added_rows,{row_diff['added_rows']}\n")
            output.write(f"removed_rows,{row_diff['removed_rows']}\n")
            output.write(f"changed_rows,{row_diff['changed_rows']}\n")
        
        return output.getvalue()
    
    else:
        raise ValueError(f"Unknown format: {format}")

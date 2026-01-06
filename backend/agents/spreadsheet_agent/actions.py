"""
Structured action API for spreadsheet operations.

Replaces raw pandas code with type-safe, validated action objects.
Provides better safety, auditability, and error handling.
"""

import logging
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, validator
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION MODELS
# ============================================================================

class SpreadsheetAction(BaseModel):
    """Base class for all spreadsheet actions"""
    action_type: str
    description: Optional[str] = None
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        """Validate action against DataFrame schema. Returns error message or None."""
        return None
    
    def to_pandas_code(self) -> str:
        """Convert action to executable pandas code"""
        raise NotImplementedError("Subclasses must implement to_pandas_code()")
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute action on DataFrame (uses to_pandas_code internally)"""
        code = self.to_pandas_code()
        safe_builtins = {
            "range": range,
            "len": len,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "int": int,
            "float": float,
            "str": str,
        }
        local_vars = {"df": df, "pd": pd}
        exec(code, {"__builtins__": safe_builtins}, local_vars)
        return local_vars["df"]


class FilterAction(SpreadsheetAction):
    """Filter rows based on condition"""
    action_type: Literal["filter"] = "filter"
    column: str
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "startswith", "endswith"]
    value: Union[str, int, float, bool]
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        if self.column not in df.columns:
            from difflib import get_close_matches
            similar = get_close_matches(self.column, df.columns, n=3, cutoff=0.6)
            msg = f"Column '{self.column}' not found. Available: {', '.join(df.columns)}."
            if similar:
                msg += f" Did you mean: {', '.join(similar)}?"
            return msg
        return None
    
    def to_pandas_code(self) -> str:
        if self.operator in ["==", "!=", ">", "<", ">=", "<="]:
            if isinstance(self.value, str):
                return f"df = df[df['{self.column}'] {self.operator} '{self.value}']"
            else:
                return f"df = df[df['{self.column}'] {self.operator} {self.value}]"
        elif self.operator == "contains":
            return f"df = df[df['{self.column}'].astype(str).str.contains('{self.value}', case=False, na=False)]"
        elif self.operator == "startswith":
            return f"df = df[df['{self.column}'].astype(str).str.startswith('{self.value}', na=False)]"
        elif self.operator == "endswith":
            return f"df = df[df['{self.column}'].astype(str).str.endswith('{self.value}', na=False)]"
        return "df"


class SortAction(SpreadsheetAction):
    """Sort rows by column(s)"""
    action_type: Literal["sort"] = "sort"
    columns: List[str]
    ascending: Union[bool, List[bool]] = True
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        return None
    
    def to_pandas_code(self) -> str:
        cols_str = str(self.columns)
        asc_str = str(self.ascending) if isinstance(self.ascending, bool) else str(self.ascending)
        return f"df = df.sort_values({cols_str}, ascending={asc_str})"


class AddColumnAction(SpreadsheetAction):
    """Add a new calculated column"""
    action_type: Literal["add_column"] = "add_column"
    new_column: str
    formula: str  # e.g., "Quantity * Price" or "Age + 1"
    position: Optional[int] = None  # None = append, 0 = first, etc.
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        # Check if column already exists
        if self.new_column in df.columns:
            return f"Column '{self.new_column}' already exists. Use RenameColumnAction first if needed."
        
        # Extract referenced columns from formula
        import re
        potential_cols = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', self.formula)
        # Filter out Python keywords and pandas methods
        python_keywords = {'and', 'or', 'not', 'if', 'else', 'True', 'False', 'None', 'sum', 'mean', 'max', 'min'}
        referenced_cols = [col for col in potential_cols if col not in python_keywords and col in df.columns]
        
        missing = [col for col in referenced_cols if col not in df.columns]
        if missing:
            return f"Formula references non-existent column(s): {', '.join(missing)}"
        
        return None
    
    def to_pandas_code(self) -> str:
        # Replace optional {col} with col to avoid set literals
        import re
        formula = re.sub(r"{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}", r"\1", self.formula)
        # Find all potential column references
        for col in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', formula):
            if col not in {'pd', 'df', 'sum', 'mean', 'max', 'min', 'abs', 'round', 'int', 'float', 'str', 'len'}:
                formula = re.sub(rf'\b{col}\b', f"df['{col}']", formula)

        if self.position is None:
            return f"df['{self.new_column}'] = {formula}\ndf"
        else:
            return f"df.insert({self.position}, '{self.new_column}', {formula})\ndf"


class RenameColumnAction(SpreadsheetAction):
    """Rename column(s)"""
    action_type: Literal["rename_column"] = "rename_column"
    mapping: Dict[str, str]  # old_name -> new_name
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        missing = [old for old in self.mapping.keys() if old not in df.columns]
        if missing:
            return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        return None
    
    def to_pandas_code(self) -> str:
        return f"df = df.rename(columns={self.mapping})\ndf"


class DropColumnAction(SpreadsheetAction):
    """Remove column(s)"""
    action_type: Literal["drop_column"] = "drop_column"
    columns: List[str]
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        return None
    
    def to_pandas_code(self) -> str:
        return f"df = df.drop(columns={self.columns})\ndf"


class GroupByAction(SpreadsheetAction):
    """Group by column(s) and aggregate"""
    action_type: Literal["group_by"] = "group_by"
    group_columns: List[str]
    agg_column: str
    agg_function: Literal["sum", "mean", "count", "min", "max", "std", "median"]
    result_column_name: Optional[str] = None
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        all_cols = self.group_columns + [self.agg_column]
        missing = [col for col in all_cols if col not in df.columns]
        if missing:
            return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        return None
    
    def to_pandas_code(self) -> str:
        group_str = str(self.group_columns) if len(self.group_columns) > 1 else f"'{self.group_columns[0]}'"
        code = f"df = df.groupby({group_str})['{self.agg_column}'].{self.agg_function}()"
        
        if self.result_column_name:
            code += f".reset_index(name='{self.result_column_name}')"
        else:
            code += ".reset_index()"
        
        return code + "\ndf"


class FillNaAction(SpreadsheetAction):
    """Fill missing values"""
    action_type: Literal["fill_na"] = "fill_na"
    columns: Optional[List[str]] = None  # None = all columns
    method: Literal["value", "forward", "backward", "mean", "median"]
    value: Optional[Union[str, int, float]] = None  # Required if method="value"
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        if self.columns:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        
        if self.method == "value" and self.value is None:
            return "method='value' requires a value parameter"
        
        return None
    
    def to_pandas_code(self) -> str:
        if self.columns:
            cols_str = str(self.columns)
            if self.method == "value":
                return f"df[{cols_str}] = df[{cols_str}].fillna({repr(self.value)})\ndf"
            elif self.method == "forward":
                return f"df[{cols_str}] = df[{cols_str}].fillna(method='ffill')\ndf"
            elif self.method == "backward":
                return f"df[{cols_str}] = df[{cols_str}].fillna(method='bfill')\ndf"
            elif self.method == "mean":
                return f"df[{cols_str}] = df[{cols_str}].fillna(df[{cols_str}].mean())\ndf"
            elif self.method == "median":
                return f"df[{cols_str}] = df[{cols_str}].fillna(df[{cols_str}].median())\ndf"
        else:
            if self.method == "value":
                return f"df = df.fillna({repr(self.value)})\ndf"
            elif self.method == "forward":
                return f"df = df.fillna(method='ffill')\ndf"
            elif self.method == "backward":
                return f"df = df.fillna(method='bfill')\ndf"
            elif self.method == "mean":
                return f"df = df.fillna(df.mean())\ndf"
            elif self.method == "median":
                return f"df = df.fillna(df.median())\ndf"
        return "df"


class DropDuplicatesAction(SpreadsheetAction):
    """Remove duplicate rows"""
    action_type: Literal["drop_duplicates"] = "drop_duplicates"
    subset: Optional[List[str]] = None  # None = consider all columns
    keep: Literal["first", "last", "none"] = "first"
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        if self.subset:
            missing = [col for col in self.subset if col not in df.columns]
            if missing:
                return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        return None
    
    def to_pandas_code(self) -> str:
        subset_str = str(self.subset) if self.subset else "None"
        keep_val = "False" if self.keep == "none" else f"'{self.keep}'"
        return f"df = df.drop_duplicates(subset={subset_str}, keep={keep_val})\ndf"


class AddSerialNumberAction(SpreadsheetAction):
    """Add a serial number column"""
    action_type: Literal["add_serial"] = "add_serial"
    column_name: str = "Sl.No."
    start: int = 1
    position: Union[int, str] = 0  # supports "first"/"last"
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        if self.column_name in df.columns:
            return f"Column '{self.column_name}' already exists."
        return None
    
    def to_pandas_code(self) -> str:
        pos_expr = "0" if self.position == "first" else ("len(df.columns)" if self.position == "last" else str(self.position))
        return f"df.insert({pos_expr}, '{self.column_name}', range({self.start}, {self.start} + len(df)))\ndf"


class AppendSummaryRowAction(SpreadsheetAction):
    """Append a summary row with aggregations for specific columns."""

    action_type: Literal["append_summary_row"] = "append_summary_row"
    aggregations: Dict[str, Literal["sum", "mean", "count", "min", "max"]]
    label_column: Optional[str] = None
    label_value: Optional[Union[str, int, float]] = None

    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        missing = [col for col in self.aggregations.keys() if col not in df.columns]
        if missing:
            return f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}."
        if self.label_column and self.label_column not in df.columns:
            return f"Label column '{self.label_column}' not found. Available: {', '.join(df.columns)}."
        return None

    def to_pandas_code(self) -> str:
        # Build a mostly-empty row
        code_lines = [
            "new_row = {col: None for col in df.columns}",
        ]

        if self.label_column is not None and self.label_value is not None:
            code_lines.append(f"new_row['{self.label_column}'] = {repr(self.label_value)}")

        for col, func in self.aggregations.items():
            series_expr = f"pd.to_numeric(df['{col}'], errors='coerce')"
            if func == "sum":
                code_lines.append(f"new_row['{col}'] = {series_expr}.sum()")
            elif func == "mean":
                code_lines.append(f"new_row['{col}'] = {series_expr}.mean()")
            elif func == "count":
                code_lines.append(f"new_row['{col}'] = {series_expr}.count()")
            elif func == "min":
                code_lines.append(f"new_row['{col}'] = {series_expr}.min()")
            elif func == "max":
                code_lines.append(f"new_row['{col}'] = {series_expr}.max()")

        code_lines.append("df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)")
        code_lines.append("df")
        return "\n".join(code_lines)


class CompareFilesAction(SpreadsheetAction):
    """Compare multiple spreadsheet files (multi-file operation)"""
    
    action_type: Literal["compare_files"] = "compare_files"
    file_ids: List[str]
    comparison_mode: Literal["schema_only", "schema_and_key", "full_diff"] = "schema_and_key"
    key_columns: Optional[List[str]] = None
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        # This is a multi-file action; validation happens at execution with all DFs
        return None
    
    def to_pandas_code(self) -> str:
        # Multi-file actions don't use simple pandas code; handled by executor
        return "# Multi-file compare operation (handled by executor)"


class MergeFilesAction(SpreadsheetAction):
    """Merge multiple spreadsheet files (multi-file operation)"""
    
    action_type: Literal["merge_files"] = "merge_files"
    file_ids: List[str]
    merge_type: Literal["join", "union", "concat"] = "join"
    join_type: Literal["inner", "left", "right", "outer"] = "inner"
    key_columns: Optional[List[str]] = None
    
    def validate_against_df(self, df: pd.DataFrame) -> Optional[str]:
        # This is a multi-file action; validation happens at execution with all DFs
        return None
    
    def to_pandas_code(self) -> str:
        # Multi-file actions don't use simple pandas code; handled by executor
        return "# Multi-file merge operation (handled by executor)"


# ============================================================================
# ACTION PARSER
# ============================================================================

class ActionParser:
    """Parse action dictionaries into action objects"""
    
    ACTION_MAP = {
        "filter": FilterAction,
        "sort": SortAction,
        "add_column": AddColumnAction,
        "rename_column": RenameColumnAction,
        "drop_column": DropColumnAction,
        "group_by": GroupByAction,
        "fill_na": FillNaAction,
        "drop_duplicates": DropDuplicatesAction,
        "add_serial": AddSerialNumberAction,
        "append_summary_row": AppendSummaryRowAction,
        "compare_files": CompareFilesAction,
        "merge_files": MergeFilesAction
    }
    
    @classmethod
    def parse(cls, action_dict: Dict[str, Any]) -> SpreadsheetAction:
        """Parse action dictionary into action object"""
        action_type = action_dict.get("action_type")
        
        if not action_type:
            raise ValueError("action_dict must have 'action_type' field")
        
        # Normalize aliases
        if action_type == "add_serial_number":
            action_type = "add_serial"
            action_dict["action_type"] = "add_serial"  # Update in dict too
        if action_type == "group_by" and "group_by_columns" in action_dict:
            action_dict.setdefault("group_columns", action_dict.pop("group_by_columns"))
        if action_type == "group_by" and "aggregate" in action_dict:
            agg = action_dict.pop("aggregate")
            if isinstance(agg, dict) and agg:
                col, func = next(iter(agg.items()))
                action_dict.setdefault("agg_column", col)
                action_dict.setdefault("agg_function", func)
        action_class = cls.ACTION_MAP.get(action_type)
        if not action_class:
            raise ValueError(f"Unknown action_type: {action_type}. Valid types: {list(cls.ACTION_MAP.keys())}")
        
        return action_class(**action_dict)
    
    @classmethod
    def parse_multiple(cls, action_dicts: List[Dict[str, Any]]) -> List[SpreadsheetAction]:
        """Parse multiple actions"""
        return [cls.parse(ad) for ad in action_dicts]


# ============================================================================
# ACTION EXECUTOR
# ============================================================================

class ActionExecutor:
    """Execute actions with validation and error handling"""
    
    @staticmethod
    def validate_action(action: SpreadsheetAction, df: pd.DataFrame) -> Optional[str]:
        """Validate action against DataFrame. Returns error or None."""
        return action.validate_against_df(df)
    
    @staticmethod
    def execute_action(action: SpreadsheetAction, df: pd.DataFrame, validate: bool = True) -> tuple[pd.DataFrame, Optional[str]]:
        """
        Execute action on DataFrame.
        
        Returns:
            (modified_df, error_message)
        """
        try:
            # Validate first
            if validate:
                error = ActionExecutor.validate_action(action, df)
                if error:
                    logger.warning(f"❌ Action validation failed: {error}")
                    return df, error
            
            # Execute
            code = action.to_pandas_code()
            logger.info(f"🔧 Executing {action.action_type}: {action.description or code[:60]}")
            
            safe_builtins = {
                "range": range,
                "len": len,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "int": int,
                "float": float,
                "str": str,
            }
            local_vars = {"df": df.copy(), "pd": pd}
            exec(code, {"__builtins__": safe_builtins}, local_vars)
            result_df = local_vars["df"]
            
            logger.info(f"✅ Action executed: {df.shape} → {result_df.shape}")
            return result_df, None
            
        except Exception as e:
            error_msg = f"Action execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return df, error_msg
    
    @staticmethod
    def execute_actions(
        actions: List[SpreadsheetAction],
        df: pd.DataFrame,
        stop_on_error: bool = False
    ) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Execute multiple actions sequentially.
        
        Returns:
            (final_df, execution_log)
        """
        current_df = df
        execution_log = []
        
        for i, action in enumerate(actions):
            log_entry = {
                "step": i + 1,
                "action_type": action.action_type,
                "description": action.description,
                "success": False,
                "error": None,
                "before_shape": current_df.shape,
                "after_shape": None
            }
            
            result_df, error = ActionExecutor.execute_action(action, current_df)
            
            if error:
                log_entry["error"] = error
                execution_log.append(log_entry)
                
                if stop_on_error:
                    logger.warning(f"⚠️ Stopping execution at step {i+1} due to error")
                    break
                else:
                    logger.warning(f"⚠️ Continuing despite error at step {i+1}")
                    continue
            
            current_df = result_df
            log_entry["success"] = True
            log_entry["after_shape"] = current_df.shape
            execution_log.append(log_entry)
        
        return current_df, execution_log


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def action_to_dict(action: SpreadsheetAction) -> Dict[str, Any]:
    """Convert action to dictionary"""
    return action.model_dump()


def actions_to_dicts(actions: List[SpreadsheetAction]) -> List[Dict[str, Any]]:
    """Convert actions to dictionaries"""
    return [action_to_dict(a) for a in actions]

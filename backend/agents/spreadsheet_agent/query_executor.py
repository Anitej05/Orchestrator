"""
Query Executor for Spreadsheet Agent

Executes pandas operations (filter, aggregate, sort, search) and handles
exact row retrieval and aggregations on full data.

Requirements: 3.5, 7.1, 7.2, 7.3, 7.4
"""

import logging
from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np
from difflib import get_close_matches

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class QueryPlan:
    """A single query operation to execute"""
    operation: str  # 'filter', 'aggregate', 'sort', 'search', 'retrieve'
    columns: Optional[List[str]] = None
    conditions: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure conditions and parameters are dicts"""
        if self.conditions is None:
            self.conditions = {}
        if self.parameters is None:
            self.parameters = {}


@dataclass
class QueryResult:
    """Result of a query execution"""
    success: bool
    data: Any  # DataFrame, scalar, or dict
    explanation: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "data": self._serialize_data(self.data),
            "explanation": self.explanation,
            "metadata": self.metadata,
            "error": self.error
        }
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON response"""
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, (np.integer, np.int64, np.int32)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        return data


# ============================================================================
# QUERY EXECUTOR
# ============================================================================

class QueryExecutor:
    """
    Executes pandas operations based on query plans.
    
    Supports:
    - Filter: Filter rows based on conditions
    - Aggregate: Compute aggregations (sum, mean, count, min, max) on full data
    - Sort: Sort rows by columns
    - Search: Search for text in columns
    - Retrieve: Retrieve specific rows by index or condition
    """
    
    def __init__(self):
        """Initialize query executor"""
        self.logger = logging.getLogger(f"{__name__}.QueryExecutor")
    
    def execute_query(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute a single query operation on a DataFrame.
        
        Args:
            df: Source DataFrame
            query: Query plan to execute
            
        Returns:
            QueryResult with success status, data, and explanation
        """
        try:
            self.logger.info(f"Executing query: {query.operation}")
            
            # Route to appropriate handler
            if query.operation == 'filter':
                return self._execute_filter(df, query)
            elif query.operation == 'aggregate':
                return self._execute_aggregate(df, query)
            elif query.operation == 'sort':
                return self._execute_sort(df, query)
            elif query.operation == 'search':
                return self._execute_search(df, query)
            elif query.operation == 'retrieve':
                return self._execute_retrieve(df, query)
            else:
                return QueryResult(
                    success=False,
                    data=None,
                    explanation=f"Unknown operation: {query.operation}",
                    metadata={},
                    error=f"Unsupported operation type: {query.operation}"
                )
                
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}", exc_info=True)
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Query execution failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def _execute_filter(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute filter operation.
        
        Conditions format:
        {
            "column": "Age",
            "operator": ">",  # ==, !=, >, <, >=, <=, contains, startswith, endswith
            "value": 25
        }
        """
        conditions = query.conditions
        column = conditions.get('column')
        operator = conditions.get('operator')
        value = conditions.get('value')
        
        if not column or not operator:
            return QueryResult(
                success=False,
                data=None,
                explanation="Filter requires 'column' and 'operator' in conditions",
                metadata={},
                error="Missing required filter parameters"
            )
        
        # Validate column exists
        if column not in df.columns:
            similar = get_close_matches(column, df.columns, n=3, cutoff=0.6)
            error_msg = f"Column '{column}' not found. Available: {', '.join(df.columns)}."
            if similar:
                error_msg += f" Did you mean: {', '.join(similar)}?"
            return QueryResult(
                success=False,
                data=None,
                explanation=error_msg,
                metadata={},
                error=error_msg
            )
        
        # Apply filter
        try:
            if operator in ['==', '!=', '>', '<', '>=', '<=']:
                mask = eval(f"df['{column}'] {operator} {repr(value)}")
            elif operator == 'contains':
                mask = df[column].astype(str).str.contains(str(value), case=False, na=False)
            elif operator == 'startswith':
                mask = df[column].astype(str).str.startswith(str(value), na=False)
            elif operator == 'endswith':
                mask = df[column].astype(str).str.endswith(str(value), na=False)
            else:
                return QueryResult(
                    success=False,
                    data=None,
                    explanation=f"Unsupported operator: {operator}",
                    metadata={},
                    error=f"Invalid operator: {operator}"
                )
            
            filtered_df = df[mask].copy()
            
            return QueryResult(
                success=True,
                data=filtered_df,
                explanation=f"Filtered {len(filtered_df)} rows where {column} {operator} {value}",
                metadata={
                    "original_rows": len(df),
                    "filtered_rows": len(filtered_df),
                    "filter_column": column,
                    "filter_operator": operator,
                    "filter_value": value
                }
            )
            
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Filter operation failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def _execute_aggregate(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute aggregation operation on FULL data (not sampled).
        
        Supports: sum, mean, count, min, max, std, median
        
        Conditions format:
        {
            "function": "sum",  # sum, mean, count, min, max, std, median
            "column": "Revenue"  # optional, if None applies to all numeric columns
        }
        """
        conditions = query.conditions
        function = conditions.get('function', 'sum')
        column = conditions.get('column')
        
        try:
            # If column specified, aggregate that column
            if column:
                if column not in df.columns:
                    similar = get_close_matches(column, df.columns, n=3, cutoff=0.6)
                    error_msg = f"Column '{column}' not found. Available: {', '.join(df.columns)}."
                    if similar:
                        error_msg += f" Did you mean: {', '.join(similar)}?"
                    return QueryResult(
                        success=False,
                        data=None,
                        explanation=error_msg,
                        metadata={},
                        error=error_msg
                    )
                
                # Convert to numeric if needed
                series = pd.to_numeric(df[column], errors='coerce')
                
                # Apply aggregation function
                if function == 'sum':
                    result = series.sum()
                elif function == 'mean':
                    result = series.mean()
                elif function == 'count':
                    result = series.count()
                elif function == 'min':
                    result = series.min()
                elif function == 'max':
                    result = series.max()
                elif function == 'std':
                    result = series.std()
                elif function == 'median':
                    result = series.median()
                else:
                    return QueryResult(
                        success=False,
                        data=None,
                        explanation=f"Unsupported aggregation function: {function}",
                        metadata={},
                        error=f"Invalid function: {function}"
                    )
                
                return QueryResult(
                    success=True,
                    data=result,
                    explanation=f"Computed {function} of {column}: {result}",
                    metadata={
                        "function": function,
                        "column": column,
                        "total_rows": len(df),
                        "non_null_values": series.count()
                    }
                )
            
            # If no column specified, aggregate all numeric columns
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_cols:
                    return QueryResult(
                        success=False,
                        data=None,
                        explanation="No numeric columns found for aggregation",
                        metadata={},
                        error="No numeric columns available"
                    )
                
                # Apply aggregation to all numeric columns
                if function == 'sum':
                    result = df[numeric_cols].sum().to_dict()
                elif function == 'mean':
                    result = df[numeric_cols].mean().to_dict()
                elif function == 'count':
                    result = df[numeric_cols].count().to_dict()
                elif function == 'min':
                    result = df[numeric_cols].min().to_dict()
                elif function == 'max':
                    result = df[numeric_cols].max().to_dict()
                elif function == 'std':
                    result = df[numeric_cols].std().to_dict()
                elif function == 'median':
                    result = df[numeric_cols].median().to_dict()
                else:
                    return QueryResult(
                        success=False,
                        data=None,
                        explanation=f"Unsupported aggregation function: {function}",
                        metadata={},
                        error=f"Invalid function: {function}"
                    )
                
                return QueryResult(
                    success=True,
                    data=result,
                    explanation=f"Computed {function} for all numeric columns",
                    metadata={
                        "function": function,
                        "columns": numeric_cols,
                        "total_rows": len(df)
                    }
                )
                
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Aggregation failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def _execute_sort(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute sort operation.
        
        Conditions format:
        {
            "columns": ["Age", "Name"],
            "ascending": [False, True]  # or single bool
        }
        """
        conditions = query.conditions
        columns = conditions.get('columns', [])
        ascending = conditions.get('ascending', True)
        
        if not columns:
            return QueryResult(
                success=False,
                data=None,
                explanation="Sort requires 'columns' in conditions",
                metadata={},
                error="Missing columns parameter"
            )
        
        # Validate columns exist
        missing = [col for col in columns if col not in df.columns]
        if missing:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Column(s) not found: {', '.join(missing)}. Available: {', '.join(df.columns)}.",
                metadata={},
                error=f"Invalid columns: {missing}"
            )
        
        try:
            sorted_df = df.sort_values(by=columns, ascending=ascending).copy()
            
            return QueryResult(
                success=True,
                data=sorted_df,
                explanation=f"Sorted by {', '.join(columns)}",
                metadata={
                    "sort_columns": columns,
                    "ascending": ascending,
                    "total_rows": len(sorted_df)
                }
            )
            
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Sort operation failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def _execute_search(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute search operation across columns.
        
        Conditions format:
        {
            "text": "search term",
            "columns": ["Name", "Description"],  # optional, if None searches all text columns
            "case_sensitive": False  # optional
        }
        """
        conditions = query.conditions
        search_text = conditions.get('text', '')
        columns = conditions.get('columns')
        case_sensitive = conditions.get('case_sensitive', False)
        
        if not search_text:
            return QueryResult(
                success=False,
                data=None,
                explanation="Search requires 'text' in conditions",
                metadata={},
                error="Missing search text"
            )
        
        try:
            # If columns not specified, search all object/string columns
            if not columns:
                columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
            if not columns:
                return QueryResult(
                    success=False,
                    data=None,
                    explanation="No text columns available for search",
                    metadata={},
                    error="No searchable columns"
                )
            
            # Validate columns exist
            missing = [col for col in columns if col not in df.columns]
            if missing:
                return QueryResult(
                    success=False,
                    data=None,
                    explanation=f"Column(s) not found: {', '.join(missing)}",
                    metadata={},
                    error=f"Invalid columns: {missing}"
                )
            
            # Search across columns
            mask = pd.Series([False] * len(df))
            for col in columns:
                mask |= df[col].astype(str).str.contains(
                    search_text,
                    case=case_sensitive,
                    na=False
                )
            
            result_df = df[mask].copy()
            
            return QueryResult(
                success=True,
                data=result_df,
                explanation=f"Found {len(result_df)} rows containing '{search_text}'",
                metadata={
                    "search_text": search_text,
                    "searched_columns": columns,
                    "case_sensitive": case_sensitive,
                    "matches_found": len(result_df)
                }
            )
            
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Search operation failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def _execute_retrieve(self, df: pd.DataFrame, query: QueryPlan) -> QueryResult:
        """
        Execute retrieve operation for exact row retrieval.
        
        Conditions format:
        {
            "indices": [0, 5, 10],  # specific row indices
            # OR
            "condition": {"column": "ID", "operator": "==", "value": 123}
        }
        """
        conditions = query.conditions
        
        try:
            # Retrieve by indices
            if 'indices' in conditions:
                indices = conditions['indices']
                if not isinstance(indices, list):
                    indices = [indices]
                
                # Validate indices
                valid_indices = [i for i in indices if 0 <= i < len(df)]
                if len(valid_indices) != len(indices):
                    invalid = [i for i in indices if i not in valid_indices]
                    return QueryResult(
                        success=False,
                        data=None,
                        explanation=f"Invalid indices: {invalid}. DataFrame has {len(df)} rows.",
                        metadata={},
                        error=f"Index out of range: {invalid}"
                    )
                
                result_df = df.iloc[valid_indices].copy()
                
                return QueryResult(
                    success=True,
                    data=result_df,
                    explanation=f"Retrieved {len(result_df)} rows by index",
                    metadata={
                        "indices": valid_indices,
                        "rows_retrieved": len(result_df)
                    }
                )
            
            # Retrieve by condition (delegate to filter)
            elif 'condition' in conditions:
                filter_query = QueryPlan(
                    operation='filter',
                    conditions=conditions['condition']
                )
                return self._execute_filter(df, filter_query)
            
            else:
                return QueryResult(
                    success=False,
                    data=None,
                    explanation="Retrieve requires 'indices' or 'condition' in conditions",
                    metadata={},
                    error="Missing retrieval parameters"
                )
                
        except Exception as e:
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Retrieve operation failed: {str(e)}",
                metadata={},
                error=str(e)
            )
    
    def execute_pandas_query(self, df: pd.DataFrame, query_string: str) -> QueryResult:
        """
        Execute a pandas query string directly.
        
        Args:
            df: Source DataFrame
            query_string: Pandas query string (e.g., "Age > 25 and Name.str.contains('John')")
            
        Returns:
            QueryResult with success status, data, and explanation
        """
        try:
            self.logger.info(f"Executing pandas query: {query_string[:100]}")
            
            # Execute the pandas query
            result_df = df.query(query_string).copy()
            
            return QueryResult(
                success=True,
                data=result_df,
                explanation=f"Pandas query returned {len(result_df)} rows",
                metadata={
                    "query_string": query_string,
                    "original_rows": len(df),
                    "result_rows": len(result_df)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Pandas query execution failed: {str(e)}", exc_info=True)
            return QueryResult(
                success=False,
                data=None,
                explanation=f"Pandas query failed: {str(e)}",
                metadata={"query_string": query_string},
                error=str(e)
            )

"""
Schema Extractor

Extracts column headers and infers data types from table regions.
Handles headerless tables, mixed types, merged cells, and formula values.

Implements Requirements 2.1, 2.3, 2.4, 2.5, 6.3 from the intelligent 
spreadsheet parsing spec.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import re
from datetime import datetime

from ..parsing_models import (
    TableSchema,
    TableRegion
)

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """
    Extracts schema information from table regions.
    
    Capabilities:
    - Extract column headers from header rows
    - Generate meaningful names for headerless tables
    - Infer data types with mixed type handling
    - Handle merged cells in headers
    - Extract formula values (not formula text)
    """
    
    def __init__(self):
        """Initialize the schema extractor."""
        self.logger = logging.getLogger(__name__)
    
    def extract_schema(
        self,
        df: pd.DataFrame,
        table_region: TableRegion,
        header_row_idx: Optional[int] = None
    ) -> TableSchema:
        """
        Extract complete schema from a table region.
        
        Args:
            df: Full pandas DataFrame
            table_region: Detected table region
            header_row_idx: Optional explicit header row index
        
        Returns:
            TableSchema object with headers and type information
        
        Validates: Requirements 2.1, 2.3, 2.4, 2.5
        """
        # Extract table data
        table_df = df.iloc[
            table_region.start_row:table_region.end_row + 1,
            table_region.start_col:table_region.end_col + 1
        ]
        
        # Determine if we have explicit headers
        has_explicit_headers = header_row_idx is not None
        
        if has_explicit_headers:
            # Extract headers from specified row
            headers = self.extract_headers(df, header_row_idx, table_region)
            # Data starts after header row
            data_start_idx = header_row_idx - table_region.start_row + 1
            if data_start_idx < len(table_df):
                data_df = table_df.iloc[data_start_idx:]
            else:
                data_df = table_df
        else:
            # Generate column names for headerless table
            headers = self.generate_column_names(table_region.col_count)
            data_df = table_df
            has_explicit_headers = False
        
        # Infer data types from the data
        dtypes = self.infer_dtypes(data_df, headers)
        
        # Calculate null counts
        null_counts = data_df.isnull().sum().to_dict()
        
        # Detect mixed type columns
        mixed_type_columns = self.detect_mixed_types(data_df, headers, dtypes)
        
        schema = TableSchema(
            headers=headers,
            dtypes=dtypes,
            row_count=len(data_df),
            col_count=len(headers),
            null_counts=null_counts
        )
        
        self.logger.info(
            f"Extracted schema: {len(headers)} columns, {len(data_df)} data rows, "
            f"explicit_headers={has_explicit_headers}"
        )
        
        return schema
    
    def extract_headers(
        self,
        df: pd.DataFrame,
        header_row_idx: int,
        table_region: Optional[TableRegion] = None
    ) -> List[str]:
        """
        Extract column names from header row.
        
        Handles merged cells by forward-filling with suffixes.
        
        Args:
            df: Full pandas DataFrame
            header_row_idx: Index of the header row
            table_region: Optional table region to limit extraction
        
        Returns:
            List of column header names
        
        Validates: Requirements 2.1, 2.2
        """
        if header_row_idx >= len(df):
            self.logger.warning(f"Header row index {header_row_idx} out of bounds")
            return []
        
        # Extract header row
        header_row = df.iloc[header_row_idx]
        
        # If table region specified, limit to those columns
        if table_region:
            header_row = header_row.iloc[table_region.start_col:table_region.end_col + 1]
        
        headers = []
        last_header = None
        merge_count = 0
        
        for idx, val in enumerate(header_row):
            if pd.notna(val) and str(val).strip():
                # New header value
                header_str = str(val).strip()
                # Clean up header name
                header_str = self._clean_header_name(header_str)
                headers.append(header_str)
                last_header = header_str
                merge_count = 0
            else:
                # Null or empty value - might be merged cell
                if last_header is not None:
                    # This is likely a merged cell continuation
                    merge_count += 1
                    # Append suffix to distinguish columns
                    headers.append(f"{last_header}_{merge_count}")
                else:
                    # No previous header, generate generic name
                    col_idx = table_region.start_col + idx if table_region else idx
                    headers.append(f"Column_{col_idx}")
        
        # Ensure unique headers
        headers = self._ensure_unique_headers(headers)
        
        self.logger.debug(f"Extracted {len(headers)} headers: {headers[:5]}...")
        return headers
    
    def generate_column_names(self, col_count: int) -> List[str]:
        """
        Generate meaningful column identifiers for headerless tables.
        
        Uses Excel-style column naming: A, B, C, ..., Z, AA, AB, ...
        
        Args:
            col_count: Number of columns to generate names for
        
        Returns:
            List of generated column names
        
        Validates: Requirement 2.3
        """
        headers = []
        for i in range(col_count):
            # Convert to Excel-style column name
            col_name = self._number_to_excel_column(i)
            headers.append(f"Column_{col_name}")
        
        self.logger.debug(f"Generated {col_count} column names for headerless table")
        return headers
    
    def infer_dtypes(
        self,
        df: pd.DataFrame,
        headers: List[str]
    ) -> Dict[str, str]:
        """
        Infer data types for each column based on cell content.
        
        Type categories:
        - numeric: int or float
        - date: datetime values
        - boolean: True/False/Yes/No
        - text: string values
        
        Args:
            df: DataFrame with data (no header row)
            headers: List of column names
        
        Returns:
            Dictionary mapping column name to inferred type
        
        Validates: Requirement 2.4
        """
        dtypes = {}
        
        for idx, header in enumerate(headers):
            if idx >= len(df.columns):
                # Column index out of bounds
                dtypes[header] = "text"
                continue
            
            col_data = df.iloc[:, idx].dropna()
            
            if len(col_data) == 0:
                # Empty column
                dtypes[header] = "text"
                continue
            
            # Analyze column values
            inferred_type = self._infer_column_type(col_data)
            dtypes[header] = inferred_type
        
        self.logger.debug(f"Inferred dtypes for {len(dtypes)} columns")
        return dtypes
    
    def detect_mixed_types(
        self,
        df: pd.DataFrame,
        headers: List[str],
        dtypes: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns with mixed data types and analyze the distribution.
        
        Args:
            df: DataFrame with data
            headers: List of column names
            dtypes: Inferred primary types
        
        Returns:
            Dictionary mapping column name to mixed type info
        
        Validates: Requirement 2.5
        """
        mixed_type_columns = {}
        
        for idx, header in enumerate(headers):
            if idx >= len(df.columns):
                continue
            
            col_data = df.iloc[:, idx].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Count types in column
            type_counts = Counter()
            for val in col_data:
                val_type = self._get_value_type(val)
                type_counts[val_type] += 1
            
            # If more than one type present, it's mixed
            if len(type_counts) > 1:
                total = sum(type_counts.values())
                predominant_type = type_counts.most_common(1)[0][0]
                predominant_pct = (type_counts[predominant_type] / total) * 100
                
                # Calculate exception percentage
                exception_pct = 100 - predominant_pct
                
                mixed_type_columns[header] = {
                    "predominant_type": predominant_type,
                    "predominant_percentage": round(predominant_pct, 2),
                    "exception_percentage": round(exception_pct, 2),
                    "type_distribution": dict(type_counts),
                    "total_values": total
                }
                
                self.logger.debug(
                    f"Column '{header}' has mixed types: "
                    f"{predominant_type} ({predominant_pct:.1f}%), "
                    f"exceptions ({exception_pct:.1f}%)"
                )
        
        return mixed_type_columns
    
    def handle_merged_cells(
        self,
        df: pd.DataFrame,
        table_region: TableRegion
    ) -> pd.DataFrame:
        """
        Handle merged cells in data (not just headers).
        
        When cells are merged in Excel:
        - First cell has the value
        - Subsequent merged cells are NaN
        
        This method forward-fills merged values within the table region.
        
        Args:
            df: Full pandas DataFrame
            table_region: Table region to process
        
        Returns:
            DataFrame with merged cells handled
        
        Validates: Requirement 6.6
        """
        # Extract table region
        table_df = df.iloc[
            table_region.start_row:table_region.end_row + 1,
            table_region.start_col:table_region.end_col + 1
        ].copy()
        
        # Forward fill NaN values that are likely from merged cells
        # We do this column by column to preserve intentional NaNs
        for col_idx in range(len(table_df.columns)):
            col = table_df.iloc[:, col_idx]
            
            # Only forward fill if there's a pattern suggesting merged cells
            # (consecutive NaNs followed by a value)
            if col.isna().any():
                # Forward fill within reasonable limits (max 3 consecutive)
                table_df.iloc[:, col_idx] = col.fillna(method='ffill', limit=3)
        
        return table_df
    
    def extract_formula_values(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract calculated values from formula cells.
        
        Note: pandas automatically reads calculated values from Excel files,
        not the formula text. This method is primarily for validation and
        documentation purposes.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            DataFrame with formula values (already extracted by pandas)
        
        Validates: Requirement 6.3
        """
        # Pandas openpyxl engine automatically reads calculated values
        # from formulas, not the formula text itself.
        # 
        # If we needed to access formula text, we would need to use
        # openpyxl directly:
        # 
        # from openpyxl import load_workbook
        # wb = load_workbook(filename, data_only=False)
        # cell.value would give formula text
        # 
        # But for our use case, we want the calculated values,
        # which pandas already provides.
        
        self.logger.debug("Formula values already extracted by pandas")
        return df
    
    # ============== PRIVATE HELPER METHODS ==============
    
    def _clean_header_name(self, header: str) -> str:
        """
        Clean and normalize header name.
        
        Args:
            header: Raw header string
        
        Returns:
            Cleaned header string
        """
        # Remove extra whitespace
        header = " ".join(header.split())
        
        # Remove special characters that might cause issues
        # but keep underscores, hyphens, and spaces
        header = re.sub(r'[^\w\s\-]', '', header)
        
        # Limit length
        if len(header) > 50:
            header = header[:50]
        
        return header
    
    def _ensure_unique_headers(self, headers: List[str]) -> List[str]:
        """
        Ensure all headers are unique by adding suffixes to duplicates.
        
        Args:
            headers: List of header names (may contain duplicates)
        
        Returns:
            List of unique header names
        """
        seen = {}
        unique_headers = []
        
        for header in headers:
            if header not in seen:
                seen[header] = 0
                unique_headers.append(header)
            else:
                seen[header] += 1
                unique_headers.append(f"{header}_{seen[header]}")
        
        return unique_headers
    
    def _number_to_excel_column(self, n: int) -> str:
        """
        Convert column number to Excel-style column name.
        
        0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, 27 -> AB, ...
        
        Args:
            n: Column index (0-based)
        
        Returns:
            Excel-style column name
        """
        result = ""
        while n >= 0:
            result = chr(65 + (n % 26)) + result
            n = n // 26 - 1
            if n < 0:
                break
        return result
    
    def _infer_column_type(self, col_data: pd.Series) -> str:
        """
        Infer the primary data type of a column.
        
        Args:
            col_data: Series of non-null values
        
        Returns:
            Type string: "numeric", "date", "boolean", or "text"
        """
        if len(col_data) == 0:
            return "text"
        
        # Count different type categories
        numeric_count = 0
        date_count = 0
        boolean_count = 0
        text_count = 0
        
        for val in col_data:
            val_type = self._get_value_type(val)
            if val_type == "numeric":
                numeric_count += 1
            elif val_type == "date":
                date_count += 1
            elif val_type == "boolean":
                boolean_count += 1
            else:
                text_count += 1
        
        total = len(col_data)
        
        # Determine predominant type (>80% threshold)
        if numeric_count / total > 0.8:
            return "numeric"
        elif date_count / total > 0.8:
            return "date"
        elif boolean_count / total > 0.8:
            return "boolean"
        else:
            return "text"
    
    def _get_value_type(self, val: Any) -> str:
        """
        Get the type category of a single value.
        
        Args:
            val: Value to categorize
        
        Returns:
            Type string: "numeric", "date", "boolean", or "text"
        """
        # Check for numeric
        if isinstance(val, (int, float, np.number)):
            return "numeric"
        
        # Check for datetime
        if isinstance(val, (datetime, pd.Timestamp)):
            return "date"
        
        # Check for boolean
        if isinstance(val, bool):
            return "boolean"
        
        # Check string representations
        if isinstance(val, str):
            val_lower = val.lower().strip()
            
            # Boolean patterns
            if val_lower in ['true', 'false', 'yes', 'no', 't', 'f', 'y', 'n']:
                return "boolean"
            
            # Try to parse as number
            try:
                float(val.replace(',', ''))
                return "numeric"
            except (ValueError, AttributeError):
                pass
            
            # Try to parse as date
            if self._is_date_string(val):
                return "date"
        
        return "text"
    
    def _is_date_string(self, s: str) -> bool:
        """
        Check if a string represents a date.
        
        Args:
            s: String to check
        
        Returns:
            True if string appears to be a date
        """
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',  # 01-15-2024
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # 1/15/24 or 1/15/2024
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, s.strip()):
                return True
        
        # Try pandas date parser
        try:
            pd.to_datetime(s)
            return True
        except (ValueError, TypeError):
            return False

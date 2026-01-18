"""
Table Detector

Identifies rectangular data regions within spreadsheets using heuristics.
Handles multiple tables, wide tables (beyond column Z), and distinguishes
data tables from pivot tables and summaries.

Implements Requirements 1.1, 1.2, 1.4, 1.5, 15.1, 15.5 from the intelligent 
spreadsheet parsing spec.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter

from ..parsing_models import (
    TableRegion,
    DocumentSection,
    SectionType
)

logger = logging.getLogger(__name__)


class TableDetector:
    """
    Detects and identifies table regions within spreadsheets.
    
    Uses heuristics to:
    - Identify table boundaries with section awareness
    - Handle multiple tables at different positions
    - Detect wide tables extending beyond column Z
    - Distinguish data tables from pivot tables and summaries
    - Detect header rows in non-standard positions
    """
    
    def __init__(self, min_table_rows: int = 3, min_table_cols: int = 2):
        """
        Initialize the table detector.
        
        Args:
            min_table_rows: Minimum rows to consider as a table
            min_table_cols: Minimum columns to consider as a table
        """
        self.logger = logging.getLogger(__name__)
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
    
    def detect_primary_table(
        self,
        df: pd.DataFrame,
        sections: Optional[List[DocumentSection]] = None
    ) -> Optional[TableRegion]:
        """
        Find the main data table in the spreadsheet.
        
        Args:
            df: Pandas DataFrame representing the spreadsheet
            sections: Optional list of document sections for context
        
        Returns:
            TableRegion for the primary table, or None if no table found
        
        Validates: Requirements 1.1, 1.2
        """
        all_tables = self.detect_all_tables(df, sections)
        
        if not all_tables:
            self.logger.warning("No tables detected in spreadsheet")
            return None
        
        # Return the largest table by area
        primary = max(all_tables, key=lambda t: t.area)
        self.logger.info(
            f"Primary table detected: rows {primary.start_row}-{primary.end_row}, "
            f"cols {primary.start_col}-{primary.end_col}, "
            f"confidence {primary.confidence:.2f}"
        )
        return primary
    
    def detect_all_tables(
        self,
        df: pd.DataFrame,
        sections: Optional[List[DocumentSection]] = None
    ) -> List[TableRegion]:
        """
        Find all potential tables in the spreadsheet.
        
        Args:
            df: Pandas DataFrame
            sections: Optional list of document sections for context
        
        Returns:
            List of TableRegion objects
        
        Validates: Requirements 15.1
        """
        if df.empty:
            return []
        
        tables = []
        
        # If sections are provided, focus on LINE_ITEMS sections
        if sections:
            for section in sections:
                if section.section_type == SectionType.LINE_ITEMS:
                    section_df = df.iloc[section.start_row:section.end_row + 1]
                    table = self._detect_table_in_region(
                        section_df,
                        row_offset=section.start_row
                    )
                    if table:
                        tables.append(table)
        
        # Also scan the entire dataframe for tables
        # This catches tables that might not be in identified sections
        full_scan_tables = self._scan_for_tables(df)
        
        # Merge results, avoiding duplicates
        for table in full_scan_tables:
            if not self._is_duplicate_table(table, tables):
                tables.append(table)
        
        # Filter out tables that are likely summaries or pivot tables
        tables = [t for t in tables if not self._is_summary_or_pivot(df, t)]
        
        self.logger.info(f"Detected {len(tables)} tables in spreadsheet")
        return tables
    
    def detect_header_row(
        self,
        df: pd.DataFrame,
        table_region: Optional[TableRegion] = None
    ) -> Optional[int]:
        """
        Identify which row contains column headers.
        
        Uses heuristics to detect headers in non-standard positions.
        
        Args:
            df: Pandas DataFrame
            table_region: Optional table region to search within
        
        Returns:
            Row index of header row, or None if no clear header
        
        Validates: Requirements 2.1, 15.4
        """
        if df.empty:
            return None
        
        # Determine search range
        if table_region:
            start_row = table_region.start_row
            end_row = min(table_region.start_row + 10, table_region.end_row)
            search_df = df.iloc[start_row:end_row + 1]
            row_offset = start_row
        else:
            # Search first 15 rows
            search_df = df.iloc[:min(15, len(df))]
            row_offset = 0
        
        # Score each row as potential header
        header_scores = []
        for idx in range(len(search_df)):
            row = search_df.iloc[idx]
            score = self._score_header_row(row, search_df, idx)
            header_scores.append((idx + row_offset, score))
        
        if not header_scores:
            return None
        
        # Return row with highest score if above threshold
        best_row, best_score = max(header_scores, key=lambda x: x[1])
        
        if best_score > 0.5:
            self.logger.info(f"Header row detected at index {best_row} (score: {best_score:.2f})")
            return best_row
        
        self.logger.warning("No clear header row detected")
        return None
    
    def handle_merged_cells(
        self,
        df: pd.DataFrame,
        header_row_idx: int
    ) -> List[str]:
        """
        Handle merged cells in header row.
        
        When cells are merged in Excel, pandas typically reads them as:
        - First cell has the value
        - Subsequent merged cells are NaN
        
        This method detects and handles this pattern by forward-filling
        merged header values.
        
        Args:
            df: Pandas DataFrame
            header_row_idx: Index of the header row
        
        Returns:
            List of header names with merged cells handled
        
        Validates: Requirement 2.2
        """
        if df.empty or header_row_idx >= len(df):
            return []
        
        header_row = df.iloc[header_row_idx]
        headers = []
        
        # Track last non-null header for forward filling
        last_header = None
        merge_count = 0
        
        for idx, val in enumerate(header_row):
            if pd.notna(val):
                # New header value
                header_str = str(val).strip()
                headers.append(header_str)
                last_header = header_str
                merge_count = 0
            else:
                # Null value - might be merged cell
                if last_header is not None:
                    # This is likely a merged cell continuation
                    merge_count += 1
                    # Append suffix to distinguish columns
                    headers.append(f"{last_header}_{merge_count}")
                else:
                    # No previous header, generate generic name
                    headers.append(f"Column_{idx}")
        
        return headers
    
    def detect_header_with_merged_cells(
        self,
        df: pd.DataFrame,
        table_region: Optional[TableRegion] = None
    ) -> Tuple[Optional[int], List[str]]:
        """
        Detect header row and handle merged cells in one operation.
        
        Args:
            df: Pandas DataFrame
            table_region: Optional table region to search within
        
        Returns:
            Tuple of (header_row_index, list_of_header_names)
        
        Validates: Requirements 2.1, 2.2, 15.4
        """
        header_row_idx = self.detect_header_row(df, table_region)
        
        if header_row_idx is None:
            return (None, [])
        
        headers = self.handle_merged_cells(df, header_row_idx)
        
        return (header_row_idx, headers)
    
    def _detect_table_in_region(
        self,
        df: pd.DataFrame,
        row_offset: int = 0
    ) -> Optional[TableRegion]:
        """
        Detect a table within a specific region.
        
        Args:
            df: DataFrame slice to search
            row_offset: Offset to add to row indices
        
        Returns:
            TableRegion or None
        """
        if df.empty or len(df) < self.min_table_rows:
            return None
        
        # Find the densest rectangular region
        best_region = None
        best_score = 0
        
        # Try different starting rows
        for start_row in range(len(df) - self.min_table_rows + 1):
            # Try different ending rows
            for end_row in range(start_row + self.min_table_rows - 1, len(df)):
                region_df = df.iloc[start_row:end_row + 1]
                
                # Find column boundaries
                col_start, col_end = self._find_column_boundaries(region_df)
                
                if col_end - col_start + 1 < self.min_table_cols:
                    continue
                
                # Score this region
                score = self._score_table_region(region_df, col_start, col_end)
                
                if score > best_score:
                    best_score = score
                    best_region = TableRegion(
                        start_row=start_row + row_offset,
                        end_row=end_row + row_offset,
                        start_col=col_start,
                        end_col=col_end,
                        confidence=score
                    )
        
        return best_region if best_score > 0.3 else None
    
    def _scan_for_tables(self, df: pd.DataFrame) -> List[TableRegion]:
        """
        Scan entire dataframe for table regions.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            List of detected TableRegion objects
        
        Validates: Requirements 1.1, 1.5 (wide tables)
        """
        tables = []
        
        # Analyze row characteristics
        row_analysis = self._analyze_rows(df)
        
        # Find contiguous data regions
        current_region_start = None
        current_region_rows = []
        
        for i, analysis in enumerate(row_analysis):
            is_data_row = (
                analysis['fill_ratio'] > 0.4 and
                not analysis['is_empty'] and
                not analysis['is_calculation']
            )
            
            if is_data_row:
                if current_region_start is None:
                    current_region_start = i
                current_region_rows.append(i)
            else:
                # End of region
                if current_region_start is not None and len(current_region_rows) >= self.min_table_rows:
                    # Create table region
                    region_df = df.iloc[current_region_rows]
                    col_start, col_end = self._find_column_boundaries(region_df)
                    
                    if col_end - col_start + 1 >= self.min_table_cols:
                        score = self._score_table_region(region_df, col_start, col_end)
                        
                        if score > 0.3:
                            table = TableRegion(
                                start_row=current_region_start,
                                end_row=current_region_rows[-1],
                                start_col=col_start,
                                end_col=col_end,
                                confidence=score
                            )
                            tables.append(table)
                
                # Reset
                current_region_start = None
                current_region_rows = []
        
        # Handle final region
        if current_region_start is not None and len(current_region_rows) >= self.min_table_rows:
            region_df = df.iloc[current_region_rows]
            col_start, col_end = self._find_column_boundaries(region_df)
            
            if col_end - col_start + 1 >= self.min_table_cols:
                score = self._score_table_region(region_df, col_start, col_end)
                
                if score > 0.3:
                    table = TableRegion(
                        start_row=current_region_start,
                        end_row=current_region_rows[-1],
                        start_col=col_start,
                        end_col=col_end,
                        confidence=score
                    )
                    tables.append(table)
        
        return tables
    
    def _find_column_boundaries(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Find the start and end columns of a table region.
        
        Handles wide tables that extend beyond column Z.
        
        Args:
            df: DataFrame slice
        
        Returns:
            Tuple of (start_col_idx, end_col_idx)
        
        Validates: Requirement 1.5 (wide tables)
        """
        if df.empty:
            return (0, 0)
        
        # Count non-null values per column
        col_fill_counts = df.notna().sum()
        
        # Find first and last columns with significant data
        threshold = len(df) * 0.3  # At least 30% filled
        
        start_col = None
        end_col = None
        
        for idx, count in enumerate(col_fill_counts):
            if count >= threshold:
                if start_col is None:
                    start_col = idx
                end_col = idx
        
        if start_col is None:
            # Fallback: use first and last non-empty columns
            non_empty_cols = [i for i, count in enumerate(col_fill_counts) if count > 0]
            if non_empty_cols:
                start_col = non_empty_cols[0]
                end_col = non_empty_cols[-1]
            else:
                start_col = 0
                end_col = len(df.columns) - 1
        
        return (start_col, end_col)
    
    def _score_table_region(
        self,
        df: pd.DataFrame,
        col_start: int,
        col_end: int
    ) -> float:
        """
        Score a potential table region.
        
        Higher scores indicate more table-like characteristics.
        
        Args:
            df: DataFrame slice
            col_start: Starting column index
            col_end: Ending column index
        
        Returns:
            Score between 0 and 1
        """
        if df.empty:
            return 0.0
        
        score = 0.0
        
        # Extract region
        region = df.iloc[:, col_start:col_end + 1]
        
        # Factor 1: Fill ratio (0-0.3 points)
        fill_ratio = region.notna().sum().sum() / (len(region) * len(region.columns))
        score += min(fill_ratio, 1.0) * 0.3
        
        # Factor 2: Column consistency (0-0.25 points)
        # Rows should have similar number of filled cells
        row_fill_counts = region.notna().sum(axis=1)
        if len(row_fill_counts) > 1:
            consistency = 1.0 - (row_fill_counts.std() / (row_fill_counts.mean() + 1))
            score += max(0, consistency) * 0.25
        
        # Factor 3: Data type consistency per column (0-0.25 points)
        type_consistency = 0.0
        for col in region.columns:
            col_data = region[col].dropna()
            if len(col_data) > 0:
                # Check if column has consistent types
                types = [type(val).__name__ for val in col_data]
                most_common_type_count = Counter(types).most_common(1)[0][1]
                type_consistency += most_common_type_count / len(col_data)
        if len(region.columns) > 0:
            score += (type_consistency / len(region.columns)) * 0.25
        
        # Factor 4: Size bonus (0-0.2 points)
        # Larger tables are more likely to be the primary table
        area = len(region) * len(region.columns)
        size_score = min(area / 100, 1.0) * 0.2
        score += size_score
        
        return min(score, 1.0)
    
    def _score_header_row(
        self,
        row: pd.Series,
        context_df: pd.DataFrame,
        row_idx: int
    ) -> float:
        """
        Score a row as a potential header row.
        
        Args:
            row: The row to score
            context_df: Surrounding rows for context
            row_idx: Index of row within context_df
        
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Factor 1: Text-heavy (0-0.3 points)
        non_null = row.dropna()
        if len(non_null) > 0:
            text_count = sum(1 for val in non_null if isinstance(val, str))
            text_ratio = text_count / len(non_null)
            score += text_ratio * 0.3
        
        # Factor 2: No numeric values or few numeric (0-0.2 points)
        numeric_count = sum(1 for val in non_null if isinstance(val, (int, float, np.number)))
        if len(non_null) > 0:
            non_numeric_ratio = 1.0 - (numeric_count / len(non_null))
            score += non_numeric_ratio * 0.2
        
        # Factor 3: Followed by data rows (0-0.3 points)
        if row_idx < len(context_df) - 1:
            next_rows = context_df.iloc[row_idx + 1:min(row_idx + 4, len(context_df))]
            if len(next_rows) > 0:
                # Check if next rows have more numeric data
                next_numeric_ratio = 0.0
                for _, next_row in next_rows.iterrows():
                    next_non_null = next_row.dropna()
                    if len(next_non_null) > 0:
                        next_numeric = sum(1 for val in next_non_null if isinstance(val, (int, float, np.number)))
                        next_numeric_ratio += next_numeric / len(next_non_null)
                next_numeric_ratio /= len(next_rows)
                score += next_numeric_ratio * 0.3
        
        # Factor 4: Unique values (0-0.2 points)
        # Headers typically have unique column names
        if len(non_null) > 0:
            unique_ratio = len(set(str(v) for v in non_null)) / len(non_null)
            score += unique_ratio * 0.2
        
        return min(score, 1.0)
    
    def _analyze_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze characteristics of each row.
        
        Returns:
            List of dicts with row characteristics
        """
        analysis = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Count non-null cells
            non_null_count = row.notna().sum()
            fill_ratio = non_null_count / len(row) if len(row) > 0 else 0
            
            # Check for calculation keywords
            is_calculation = False
            if non_null_count > 0:
                text_values = [str(val).lower() for val in row if pd.notna(val) and isinstance(val, str)]
                calc_keywords = ['total', 'subtotal', 'sum', 'average', 'grand total']
                is_calculation = any(kw in text for text in text_values for kw in calc_keywords)
            
            analysis.append({
                'row_idx': idx,
                'non_null_count': non_null_count,
                'fill_ratio': fill_ratio,
                'is_empty': non_null_count == 0,
                'is_calculation': is_calculation
            })
        
        return analysis
    
    def _is_duplicate_table(self, table: TableRegion, existing_tables: List[TableRegion]) -> bool:
        """
        Check if a table overlaps significantly with existing tables.
        
        Args:
            table: Table to check
            existing_tables: List of already detected tables
        
        Returns:
            True if table is a duplicate
        """
        for existing in existing_tables:
            # Check for overlap
            row_overlap = (
                table.start_row <= existing.end_row and
                table.end_row >= existing.start_row
            )
            col_overlap = (
                table.start_col <= existing.end_col and
                table.end_col >= existing.start_col
            )
            
            if row_overlap and col_overlap:
                # Calculate overlap area
                overlap_rows = min(table.end_row, existing.end_row) - max(table.start_row, existing.start_row) + 1
                overlap_cols = min(table.end_col, existing.end_col) - max(table.start_col, existing.start_col) + 1
                overlap_area = overlap_rows * overlap_cols
                
                # If >50% overlap, consider duplicate
                table_area = table.area
                if overlap_area / table_area > 0.5:
                    return True
        
        return False
    
    def _is_summary_or_pivot(self, df: pd.DataFrame, table: TableRegion) -> bool:
        """
        Determine if a table is likely a summary or pivot table.
        
        Summary/pivot characteristics:
        - Contains calculation keywords
        - Sparse structure
        - Small size
        - Located at bottom of document
        
        Args:
            df: Full dataframe
            table: Table region to check
        
        Returns:
            True if likely a summary or pivot table
        
        Validates: Requirement 15.5
        """
        # Extract table data
        table_df = df.iloc[table.start_row:table.end_row + 1, table.start_col:table.end_col + 1]
        
        # Check for calculation keywords
        calc_keyword_count = 0
        for _, row in table_df.iterrows():
            for val in row:
                if pd.notna(val) and isinstance(val, str):
                    val_lower = val.lower()
                    calc_keywords = ['total', 'subtotal', 'sum', 'average', 'grand total', 'summary']
                    if any(kw in val_lower for kw in calc_keywords):
                        calc_keyword_count += 1
        
        # High density of calculation keywords suggests summary
        if calc_keyword_count > len(table_df) * 0.3:
            return True
        
        # Check if table is at bottom of document (last 20%)
        if table.start_row > len(df) * 0.8:
            # Small tables at bottom are likely summaries
            if table.row_count < 10:
                return True
        
        # Check for pivot table structure (hierarchical row headers)
        # Pivot tables often have merged cells or indented text in first column
        first_col = table_df.iloc[:, 0]
        text_values = [str(val) for val in first_col if pd.notna(val) and isinstance(val, str)]
        if text_values:
            # Check for indentation patterns (leading spaces)
            indented_count = sum(1 for val in text_values if val.startswith(' ') or val.startswith('\t'))
            if indented_count / len(text_values) > 0.3:
                return True
        
        return False

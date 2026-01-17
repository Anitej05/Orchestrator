"""
Intentional Gap Detector

Distinguishes between intentional structural separators and missing data by analyzing
empty rows in context of surrounding content and patterns.

Implements Requirements 17.1, 17.2, 17.3, 17.4, 17.5 from the intelligent spreadsheet parsing spec.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter

from ..parsing_models import EmptyRowClassification

logger = logging.getLogger(__name__)


class IntentionalGapDetector:
    """
    Detects and classifies empty rows as intentional structural separators or missing data.
    
    Uses heuristics based on:
    - Surrounding content type (metadata vs data vs summary)
    - Spacing patterns (consistent gaps vs random)
    - Position in document (between sections vs within data)
    - Row characteristics before and after the gap
    """
    
    def __init__(self):
        """Initialize the intentional gap detector."""
        self.logger = logging.getLogger(__name__)
    
    def classify_empty_rows(self, df: pd.DataFrame) -> Dict[int, EmptyRowClassification]:
        """
        Classify all empty rows in a dataframe.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            Dict mapping row index to EmptyRowClassification
        
        Validates: Requirement 17.1
        """
        if df.empty:
            return {}
        
        # Find all empty rows
        empty_rows = self._find_empty_rows(df)
        
        if not empty_rows:
            return {}
        
        # Analyze row characteristics
        row_analysis = self._analyze_all_rows(df)
        
        # Classify each empty row
        classifications = {}
        for row_idx in empty_rows:
            classification = self._classify_single_empty_row(
                row_idx, df, row_analysis, empty_rows
            )
            classifications[row_idx] = classification
        
        self.logger.info(
            f"Classified {len(empty_rows)} empty rows: "
            f"{sum(1 for c in classifications.values() if c == EmptyRowClassification.STRUCTURAL_SEPARATOR)} structural, "
            f"{sum(1 for c in classifications.values() if c == EmptyRowClassification.SECTION_BOUNDARY)} boundaries, "
            f"{sum(1 for c in classifications.values() if c == EmptyRowClassification.MISSING_DATA)} missing data"
        )
        
        return classifications
    
    def identify_intentional_gaps(self, df: pd.DataFrame) -> List[int]:
        """
        Identify row indices that are intentional structural gaps.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            List of row indices that are intentional gaps (not missing data)
        
        Validates: Requirements 17.2, 17.3, 17.5
        """
        classifications = self.classify_empty_rows(df)
        
        # Return rows classified as structural separators or section boundaries
        intentional_gaps = [
            row_idx for row_idx, classification in classifications.items()
            if classification in [
                EmptyRowClassification.STRUCTURAL_SEPARATOR,
                EmptyRowClassification.SECTION_BOUNDARY
            ]
        ]
        
        return intentional_gaps
    
    def _find_empty_rows(self, df: pd.DataFrame) -> List[int]:
        """
        Find all empty rows in the dataframe.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            List of row indices that are completely empty
        """
        empty_rows = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            if row.isna().all():
                empty_rows.append(idx)
        
        return empty_rows
    
    def _analyze_all_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze characteristics of all rows.
        
        Returns:
            List of dicts with row characteristics
        """
        analysis = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Count non-null cells
            non_null_count = row.notna().sum()
            fill_ratio = non_null_count / len(row) if len(row) > 0 else 0
            
            # Count numeric vs text cells
            numeric_count = sum(1 for val in row if pd.notna(val) and isinstance(val, (int, float, np.number)))
            text_count = sum(1 for val in row if pd.notna(val) and isinstance(val, str))
            
            # Determine content type
            content_type = self._determine_row_content_type(row)
            
            analysis.append({
                'row_idx': idx,
                'is_empty': non_null_count == 0,
                'fill_ratio': fill_ratio,
                'numeric_count': numeric_count,
                'text_count': text_count,
                'content_type': content_type
            })
        
        return analysis
    
    def _classify_single_empty_row(
        self,
        row_idx: int,
        df: pd.DataFrame,
        row_analysis: List[Dict[str, Any]],
        all_empty_rows: List[int]
    ) -> EmptyRowClassification:
        """
        Classify a single empty row.
        
        Args:
            row_idx: Index of the empty row
            df: Full dataframe
            row_analysis: Pre-computed analysis of all rows
            all_empty_rows: List of all empty row indices
        
        Returns:
            EmptyRowClassification
        
        Validates: Requirements 17.1, 17.2, 17.3, 17.4
        """
        # Analyze content before and after
        before_content = self._analyze_content_before(row_idx, row_analysis)
        after_content = self._analyze_content_after(row_idx, row_analysis)
        
        # Check for content type transition
        if before_content['type'] != after_content['type']:
            # Different content types suggest section boundary
            return EmptyRowClassification.SECTION_BOUNDARY
        
        # Check for consistent spacing pattern
        if self._is_consistent_spacing(row_idx, all_empty_rows):
            return EmptyRowClassification.STRUCTURAL_SEPARATOR
        
        # Check if it's between header and data
        if self._is_header_data_separator(row_idx, row_analysis):
            return EmptyRowClassification.STRUCTURAL_SEPARATOR
        
        # Check if it's between data and summary
        if self._is_data_summary_separator(row_idx, row_analysis):
            return EmptyRowClassification.SECTION_BOUNDARY
        
        # Check if surrounded by similar data rows (random gap)
        if before_content['type'] == 'data' and after_content['type'] == 'data':
            # Check consistency of surrounding data
            if before_content['fill_ratio'] > 0.5 and after_content['fill_ratio'] > 0.5:
                return EmptyRowClassification.MISSING_DATA
        
        # Default to structural separator if at document boundaries
        if row_idx < 5 or row_idx > len(df) - 5:
            return EmptyRowClassification.STRUCTURAL_SEPARATOR
        
        # Default to missing data for ambiguous cases
        return EmptyRowClassification.MISSING_DATA
    
    def _analyze_content_before(
        self,
        row_idx: int,
        row_analysis: List[Dict[str, Any]],
        window: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze content in rows before the empty row.
        
        Args:
            row_idx: Index of the empty row
            row_analysis: Pre-computed row analysis
            window: Number of rows to look back
        
        Returns:
            Dict with content characteristics
        """
        start_idx = max(0, row_idx - window)
        rows_before = [r for r in row_analysis[start_idx:row_idx] if not r['is_empty']]
        
        if not rows_before:
            return {'type': 'none', 'fill_ratio': 0}
        
        # Aggregate characteristics
        avg_fill_ratio = np.mean([r['fill_ratio'] for r in rows_before])
        content_types = [r['content_type'] for r in rows_before]
        most_common_type = Counter(content_types).most_common(1)[0][0] if content_types else 'unknown'
        
        return {
            'type': most_common_type,
            'fill_ratio': avg_fill_ratio,
            'row_count': len(rows_before)
        }
    
    def _analyze_content_after(
        self,
        row_idx: int,
        row_analysis: List[Dict[str, Any]],
        window: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze content in rows after the empty row.
        
        Args:
            row_idx: Index of the empty row
            row_analysis: Pre-computed row analysis
            window: Number of rows to look ahead
        
        Returns:
            Dict with content characteristics
        """
        end_idx = min(len(row_analysis), row_idx + window + 1)
        rows_after = [r for r in row_analysis[row_idx + 1:end_idx] if not r['is_empty']]
        
        if not rows_after:
            return {'type': 'none', 'fill_ratio': 0}
        
        # Aggregate characteristics
        avg_fill_ratio = np.mean([r['fill_ratio'] for r in rows_after])
        content_types = [r['content_type'] for r in rows_after]
        most_common_type = Counter(content_types).most_common(1)[0][0] if content_types else 'unknown'
        
        return {
            'type': most_common_type,
            'fill_ratio': avg_fill_ratio,
            'row_count': len(rows_after)
        }
    
    def _determine_row_content_type(self, row: pd.Series) -> str:
        """
        Determine the content type of a row.
        
        Args:
            row: Pandas Series representing a row
        
        Returns:
            Content type string: 'metadata', 'data', 'summary', 'empty'
        """
        if row.isna().all():
            return 'empty'
        
        non_null_count = row.notna().sum()
        fill_ratio = non_null_count / len(row)
        
        # Low fill ratio suggests metadata
        if fill_ratio < 0.3:
            return 'metadata'
        
        # Check for calculation keywords
        text_values = [str(val).lower() for val in row if pd.notna(val) and isinstance(val, str)]
        calc_keywords = ['total', 'subtotal', 'sum', 'average', 'grand total']
        has_calc = any(kw in text for text in text_values for kw in calc_keywords)
        
        if has_calc:
            return 'summary'
        
        # High fill ratio suggests data
        if fill_ratio > 0.5:
            return 'data'
        
        return 'metadata'
    
    def _is_consistent_spacing(self, row_idx: int, all_empty_rows: List[int]) -> bool:
        """
        Check if this empty row is part of a consistent spacing pattern.
        
        Args:
            row_idx: Index of the empty row
            all_empty_rows: List of all empty row indices
        
        Returns:
            True if part of consistent pattern
        
        Validates: Requirement 17.5
        """
        if len(all_empty_rows) < 2:
            return False
        
        # Find position in empty rows list
        try:
            pos = all_empty_rows.index(row_idx)
        except ValueError:
            return False
        
        # Check spacing before and after
        if pos > 0 and pos < len(all_empty_rows) - 1:
            spacing_before = row_idx - all_empty_rows[pos - 1]
            spacing_after = all_empty_rows[pos + 1] - row_idx
            
            # Consistent spacing (within 2 rows)
            if abs(spacing_before - spacing_after) <= 2:
                return True
        
        return False
    
    def _is_header_data_separator(self, row_idx: int, row_analysis: List[Dict[str, Any]]) -> bool:
        """
        Check if this empty row separates header from data.
        
        Args:
            row_idx: Index of the empty row
            row_analysis: Pre-computed row analysis
        
        Returns:
            True if this is a header-data separator
        
        Validates: Requirement 17.2
        """
        # Must be in first 10 rows
        if row_idx > 10:
            return False
        
        # Check content before (should be metadata/sparse)
        before = self._analyze_content_before(row_idx, row_analysis, window=3)
        if before['type'] not in ['metadata', 'none']:
            return False
        
        # Check content after (should be data/dense)
        after = self._analyze_content_after(row_idx, row_analysis, window=3)
        if after['type'] != 'data':
            return False
        
        # Fill ratio should increase significantly after the gap
        if after['fill_ratio'] > before['fill_ratio'] + 0.3:
            return True
        
        return False
    
    def _is_data_summary_separator(self, row_idx: int, row_analysis: List[Dict[str, Any]]) -> bool:
        """
        Check if this empty row separates data from summary.
        
        Args:
            row_idx: Index of the empty row
            row_analysis: Pre-computed row analysis
        
        Returns:
            True if this is a data-summary separator
        
        Validates: Requirement 17.3
        """
        # Check content before (should be data)
        before = self._analyze_content_before(row_idx, row_analysis, window=5)
        if before['type'] != 'data':
            return False
        
        # Check content after (should be summary)
        after = self._analyze_content_after(row_idx, row_analysis, window=3)
        if after['type'] != 'summary':
            return False
        
        return True

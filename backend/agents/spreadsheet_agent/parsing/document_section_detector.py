"""
Document Section Detector

Identifies distinct sections within spreadsheet documents (header, data, summary, footer)
and classifies document types (invoice, report, form).

Implements Requirements 16.1, 16.5 from the intelligent spreadsheet parsing spec.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from collections import Counter

from ..parsing_models import (
    DocumentSection,
    SectionType,
    ContentType,
    DocumentType
)

logger = logging.getLogger(__name__)


class DocumentSectionDetector:
    """
    Detects and classifies sections within spreadsheet documents.
    
    Uses heuristics to identify:
    - Header sections (metadata, titles, document info)
    - Line item sections (tabular data)
    - Summary sections (totals, calculations)
    - Footer sections (notes, signatures)
    
    Also classifies overall document type (invoice, report, form, etc.)
    """
    
    def __init__(self):
        """Initialize the document section detector."""
        self.logger = logging.getLogger(__name__)
    
    def detect_sections(self, df: pd.DataFrame) -> List[DocumentSection]:
        """
        Detect all sections in a spreadsheet.
        
        Args:
            df: Pandas DataFrame representing the spreadsheet
        
        Returns:
            List of DocumentSection objects
        
        Validates: Requirement 16.1
        """
        if df.empty:
            return []
        
        sections = []
        current_row = 0
        total_rows = len(df)
        
        # Analyze row characteristics
        row_analysis = self._analyze_rows(df)
        
        # Detect header section (typically at the top)
        header_end = self._detect_header_section(df, row_analysis)
        if header_end > 0:
            sections.append(DocumentSection(
                section_type=SectionType.HEADER,
                start_row=0,
                end_row=header_end - 1,
                content_type=self._determine_content_type(df.iloc[0:header_end]),
                metadata=self._extract_section_metadata(df.iloc[0:header_end])
            ))
            current_row = header_end
        
        # Detect main data section (line items)
        if current_row < total_rows:
            data_end = self._detect_data_section(df, row_analysis, current_row)
            if data_end > current_row:
                sections.append(DocumentSection(
                    section_type=SectionType.LINE_ITEMS,
                    start_row=current_row,
                    end_row=data_end - 1,
                    content_type=ContentType.TABLE,
                    metadata=self._extract_section_metadata(df.iloc[current_row:data_end])
                ))
                current_row = data_end
        
        # Detect summary section (totals, calculations)
        if current_row < total_rows:
            summary_end = self._detect_summary_section(df, row_analysis, current_row)
            if summary_end > current_row:
                sections.append(DocumentSection(
                    section_type=SectionType.SUMMARY,
                    start_row=current_row,
                    end_row=summary_end - 1,
                    content_type=ContentType.CALCULATIONS,
                    metadata=self._extract_section_metadata(df.iloc[current_row:summary_end])
                ))
                current_row = summary_end
        
        # Remaining rows are footer
        if current_row < total_rows:
            sections.append(DocumentSection(
                section_type=SectionType.FOOTER,
                start_row=current_row,
                end_row=total_rows - 1,
                content_type=ContentType.TEXT,
                metadata=self._extract_section_metadata(df.iloc[current_row:])
            ))
        
        self.logger.info(f"Detected {len(sections)} sections in document")
        return sections
    
    def classify_document_type(self, df: pd.DataFrame, sections: List[DocumentSection]) -> DocumentType:
        """
        Classify the overall document type.
        
        Args:
            df: Pandas DataFrame
            sections: List of detected sections
        
        Returns:
            DocumentType enum value
        
        Validates: Requirement 16.5
        """
        # Extract text content from the document
        text_content = self._extract_text_content(df)
        text_lower = text_content.lower()
        
        # Invoice detection
        invoice_keywords = ['invoice', 'bill', 'payment', 'due date', 'subtotal', 'vat', 'tax', 'total amount']
        invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
        
        # Report detection
        report_keywords = ['report', 'summary', 'analysis', 'period', 'quarter', 'year', 'performance']
        report_score = sum(1 for kw in report_keywords if kw in text_lower)
        
        # Form detection
        form_keywords = ['form', 'application', 'name:', 'address:', 'date:', 'signature']
        form_score = sum(1 for kw in form_keywords if kw in text_lower)
        
        # Check section structure
        has_line_items = any(s.section_type == SectionType.LINE_ITEMS for s in sections)
        has_summary = any(s.section_type == SectionType.SUMMARY for s in sections)
        
        # Decision logic
        if invoice_score >= 3 and has_line_items and has_summary:
            return DocumentType.INVOICE
        elif report_score >= 2:
            return DocumentType.REPORT
        elif form_score >= 3:
            return DocumentType.FORM
        elif has_line_items:
            return DocumentType.DATA_TABLE
        else:
            return DocumentType.UNKNOWN
    
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
            
            # Count numeric vs text cells
            numeric_count = sum(1 for val in row if pd.notna(val) and isinstance(val, (int, float, np.number)))
            text_count = sum(1 for val in row if pd.notna(val) and isinstance(val, str))
            
            # Check for key-value patterns (e.g., "Label: Value")
            has_key_value = False
            if text_count > 0:
                text_cells = [str(val) for val in row if pd.notna(val) and isinstance(val, str)]
                has_key_value = any(':' in cell for cell in text_cells)
            
            # Check for calculation indicators
            has_calculation = False
            if text_count > 0:
                text_cells_lower = [str(val).lower() for val in row if pd.notna(val) and isinstance(val, str)]
                calc_keywords = ['total', 'subtotal', 'sum', 'average', 'count', 'grand total']
                has_calculation = any(kw in cell for cell in text_cells_lower for kw in calc_keywords)
            
            analysis.append({
                'row_idx': idx,
                'non_null_count': non_null_count,
                'fill_ratio': fill_ratio,
                'numeric_count': numeric_count,
                'text_count': text_count,
                'has_key_value': has_key_value,
                'has_calculation': has_calculation,
                'is_empty': non_null_count == 0
            })
        
        return analysis
    
    def _detect_header_section(self, df: pd.DataFrame, row_analysis: List[Dict]) -> int:
        """
        Detect the end of the header section.
        
        Header characteristics:
        - Low fill ratio (<50%)
        - Key-value patterns
        - Text-heavy
        - At the top of the document
        
        Returns:
            Row index where header ends (exclusive)
        """
        header_end = 0
        
        for i, analysis in enumerate(row_analysis):
            # Stop at first row with high fill ratio and consistent structure
            if i > 0 and analysis['fill_ratio'] > 0.5:
                # Check if this looks like a data row
                if analysis['numeric_count'] > 0 or analysis['text_count'] > 2:
                    # Check if previous rows were sparse (header-like)
                    prev_sparse = all(
                        row_analysis[j]['fill_ratio'] < 0.5 
                        for j in range(max(0, i-3), i)
                    )
                    if prev_sparse:
                        header_end = i
                        break
            
            # Stop if we see consistent data rows
            if i >= 5:
                # Check last 3 rows for consistency
                recent_rows = row_analysis[i-2:i+1]
                if all(r['fill_ratio'] > 0.5 for r in recent_rows):
                    header_end = i - 2
                    break
        
        # If no clear boundary, assume first 3 rows are header if they're sparse
        if header_end == 0 and len(row_analysis) > 3:
            if all(row_analysis[i]['fill_ratio'] < 0.5 for i in range(min(3, len(row_analysis)))):
                header_end = 3
        
        return header_end
    
    def _detect_data_section(self, df: pd.DataFrame, row_analysis: List[Dict], start_row: int) -> int:
        """
        Detect the end of the data section (line items).
        
        Data section characteristics:
        - High fill ratio (>50%)
        - Consistent column count
        - Mix of text and numeric data
        - No calculation keywords
        
        Returns:
            Row index where data section ends (exclusive)
        """
        data_end = start_row
        consistent_rows = 0
        
        for i in range(start_row, len(row_analysis)):
            analysis = row_analysis[i]
            
            # Skip empty rows (might be intentional gaps)
            if analysis['is_empty']:
                continue
            
            # Data rows have good fill ratio and no calculation keywords
            if analysis['fill_ratio'] > 0.5 and not analysis['has_calculation']:
                consistent_rows += 1
                data_end = i + 1
            else:
                # If we've seen consistent data rows, this might be end of data
                if consistent_rows >= 3:
                    break
                # Otherwise, might still be in header
                if i == start_row:
                    data_end = i + 1
        
        return data_end
    
    def _detect_summary_section(self, df: pd.DataFrame, row_analysis: List[Dict], start_row: int) -> int:
        """
        Detect the end of the summary section.
        
        Summary characteristics:
        - Calculation keywords (total, subtotal, etc.)
        - Lower fill ratio than data section
        - Numeric values
        
        Returns:
            Row index where summary section ends (exclusive)
        """
        summary_end = start_row
        
        for i in range(start_row, len(row_analysis)):
            analysis = row_analysis[i]
            
            # Skip empty rows
            if analysis['is_empty']:
                continue
            
            # Summary rows have calculations or key-value patterns
            if analysis['has_calculation'] or (analysis['has_key_value'] and analysis['numeric_count'] > 0):
                summary_end = i + 1
            else:
                # If we've found summary rows, stop at first non-summary row
                if summary_end > start_row:
                    break
        
        return summary_end
    
    def _determine_content_type(self, section_df: pd.DataFrame) -> ContentType:
        """
        Determine the content type of a section.
        
        Args:
            section_df: DataFrame slice for the section
        
        Returns:
            ContentType enum value
        """
        if section_df.empty:
            return ContentType.TEXT
        
        # Check for table structure (consistent columns, multiple rows)
        if len(section_df) >= 3:
            # Check fill ratio consistency
            fill_ratios = [row.notna().sum() / len(row) for _, row in section_df.iterrows()]
            if np.std(fill_ratios) < 0.2 and np.mean(fill_ratios) > 0.5:
                return ContentType.TABLE
        
        # Check for key-value pairs
        text_cells = []
        for _, row in section_df.iterrows():
            for val in row:
                if pd.notna(val) and isinstance(val, str):
                    text_cells.append(val)
        
        if text_cells:
            key_value_count = sum(1 for cell in text_cells if ':' in cell)
            if key_value_count / len(text_cells) > 0.3:
                return ContentType.KEY_VALUE
        
        # Check for calculations
        calc_keywords = ['total', 'subtotal', 'sum', 'average', 'count']
        has_calc = any(
            any(kw in str(val).lower() for kw in calc_keywords)
            for _, row in section_df.iterrows()
            for val in row if pd.notna(val)
        )
        if has_calc:
            return ContentType.CALCULATIONS
        
        return ContentType.TEXT
    
    def _extract_section_metadata(self, section_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract metadata from a section.
        
        Args:
            section_df: DataFrame slice for the section
        
        Returns:
            Dict with section metadata
        """
        metadata = {
            'row_count': len(section_df),
            'col_count': len(section_df.columns),
            'fill_ratio': section_df.notna().sum().sum() / (len(section_df) * len(section_df.columns)) if not section_df.empty else 0
        }
        
        # Extract key-value pairs if present
        key_values = {}
        for _, row in section_df.iterrows():
            for val in row:
                if pd.notna(val) and isinstance(val, str) and ':' in val:
                    parts = val.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        key_values[key] = value
        
        if key_values:
            metadata['key_values'] = key_values
        
        return metadata
    
    def _extract_text_content(self, df: pd.DataFrame) -> str:
        """
        Extract all text content from dataframe for keyword analysis.
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            Concatenated text content
        """
        text_parts = []
        
        for _, row in df.iterrows():
            for val in row:
                if pd.notna(val) and isinstance(val, str):
                    text_parts.append(val)
        
        return ' '.join(text_parts)

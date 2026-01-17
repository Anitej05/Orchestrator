"""
Metadata Extractor

Extracts and structures key-value metadata from spreadsheets.
Identifies common metadata patterns (titles, dates, IDs) and separates
metadata from table data.

Implements Requirements 4.1, 4.3, 4.4, 16.6 from the intelligent 
spreadsheet parsing spec.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import re
from datetime import datetime

from ..parsing_models import (
    DocumentSection,
    SectionType,
    ContentType
)

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extracts metadata from spreadsheets.
    
    Capabilities:
    - Extract and structure key-value metadata
    - Identify common metadata patterns (titles, dates, IDs)
    - Separate metadata from table data
    - Locate metadata regardless of position in document
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.logger = logging.getLogger(__name__)
        
        # Common metadata field patterns
        self.metadata_patterns = {
            'invoice_number': [
                r'invoice\s*#?:?\s*(\S+)',
                r'invoice\s+number:?\s*(\S+)',
                r'inv\s*#?:?\s*(\S+)',
            ],
            'date': [
                r'date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'date:?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'invoice\s+date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            ],
            'total': [
                r'total:?\s*\$?\s*([\d,]+\.?\d*)',
                r'grand\s+total:?\s*\$?\s*([\d,]+\.?\d*)',
                r'amount\s+due:?\s*\$?\s*([\d,]+\.?\d*)',
            ],
            'customer_id': [
                r'customer\s*#?:?\s*(\S+)',
                r'customer\s+id:?\s*(\S+)',
                r'client\s*#?:?\s*(\S+)',
            ],
            'order_number': [
                r'order\s*#?:?\s*(\S+)',
                r'order\s+number:?\s*(\S+)',
                r'po\s*#?:?\s*(\S+)',
            ],
            'company': [
                r'company:?\s*(.+)',
                r'business\s+name:?\s*(.+)',
                r'organization:?\s*(.+)',
            ],
        }
    
    def extract_metadata(
        self,
        df: pd.DataFrame,
        sections: Optional[List[DocumentSection]] = None
    ) -> Dict[str, Any]:
        """
        Extract all metadata from spreadsheet.
        
        Args:
            df: Full pandas DataFrame
            sections: Optional list of document sections for context
        
        Returns:
            Dictionary of extracted metadata
        
        Validates: Requirements 4.1, 4.3, 4.4, 16.6
        """
        metadata = {}
        
        # If sections provided, focus on HEADER and METADATA sections
        if sections:
            for section in sections:
                if section.section_type in [SectionType.HEADER, SectionType.METADATA]:
                    section_df = df.iloc[section.start_row:section.end_row + 1]
                    section_metadata = self._extract_from_region(section_df)
                    metadata.update(section_metadata)
        
        # Also scan top rows for metadata (common location)
        top_rows = df.iloc[:min(20, len(df))]
        top_metadata = self._extract_from_region(top_rows)
        
        # Merge, preferring section-based extraction
        for key, value in top_metadata.items():
            if key not in metadata:
                metadata[key] = value
        
        # Extract key-value pairs
        kv_pairs = self._extract_key_value_pairs(df)
        metadata.update(kv_pairs)
        
        # Identify common patterns
        pattern_metadata = self._identify_common_patterns(df)
        
        # Merge pattern-based metadata
        for key, value in pattern_metadata.items():
            if key not in metadata:
                metadata[key] = value
        
        self.logger.info(f"Extracted {len(metadata)} metadata fields")
        return metadata
    
    def identify_metadata_sections(
        self,
        df: pd.DataFrame
    ) -> List[DocumentSection]:
        """
        Identify sections of the spreadsheet that contain metadata.
        
        Metadata sections typically have:
        - Low fill ratio (<50%)
        - Key-value patterns
        - Located at top or bottom of document
        - Different structure from data tables
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            List of DocumentSection objects for metadata regions
        
        Validates: Requirement 4.1
        """
        metadata_sections = []
        
        # Analyze each row
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Check if row looks like metadata
            if self._is_metadata_row(row, df, idx):
                # Start or extend metadata section
                if metadata_sections and metadata_sections[-1].end_row == idx - 1:
                    # Extend existing section
                    metadata_sections[-1].end_row = idx
                else:
                    # Start new section
                    section = DocumentSection(
                        section_type=SectionType.METADATA,
                        start_row=idx,
                        end_row=idx,
                        content_type=ContentType.KEY_VALUE,
                        metadata={}
                    )
                    metadata_sections.append(section)
        
        self.logger.debug(f"Identified {len(metadata_sections)} metadata sections")
        return metadata_sections
    
    def extract_title(
        self,
        df: pd.DataFrame
    ) -> Optional[str]:
        """
        Extract document title if present.
        
        Title is typically:
        - In first few rows
        - Spans multiple columns (merged cell)
        - Contains text, not numbers
        - Larger font (we can't detect this, but position helps)
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            Title string or None
        
        Validates: Requirement 4.4
        """
        # Check first 5 rows
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            
            # Look for a cell with text that spans or is prominent
            non_null = row.dropna()
            
            if len(non_null) == 1:
                # Single cell with value - likely a title
                val = non_null.iloc[0]
                if isinstance(val, str) and len(val) > 5:
                    # Check if it looks like a title (not a field label)
                    if ':' not in val and not val.lower().startswith(('date', 'invoice', 'customer')):
                        self.logger.debug(f"Found title: {val}")
                        return val.strip()
            
            elif len(non_null) > 0:
                # Check first cell
                first_val = non_null.iloc[0]
                if isinstance(first_val, str) and len(first_val) > 10:
                    # Long text in first cell might be title
                    if ':' not in first_val:
                        self.logger.debug(f"Found title: {first_val}")
                        return first_val.strip()
        
        return None
    
    def extract_dates(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[str, Any]]:
        """
        Extract all date fields from spreadsheet.
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            List of (field_name, date_value) tuples
        
        Validates: Requirement 4.4
        """
        dates = []
        
        # Search top rows for date patterns
        for idx in range(min(20, len(df))):
            row = df.iloc[idx]
            
            for col_idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                
                val_str = str(val).lower()
                
                # Check if this looks like a date field
                if 'date' in val_str:
                    # Next cell might have the date value
                    if col_idx + 1 < len(row):
                        date_val = row.iloc[col_idx + 1]
                        if pd.notna(date_val):
                            parsed_date = self._parse_date(date_val)
                            if parsed_date:
                                field_name = str(val).strip().rstrip(':')
                                dates.append((field_name, parsed_date))
        
        self.logger.debug(f"Extracted {len(dates)} date fields")
        return dates
    
    def extract_identifiers(
        self,
        df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Extract document identifiers (invoice #, order #, customer ID, etc.).
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            Dictionary of identifier fields
        
        Validates: Requirement 4.4
        """
        identifiers = {}
        
        # Search top rows for identifier patterns
        for idx in range(min(20, len(df))):
            row = df.iloc[idx]
            
            for col_idx, val in enumerate(row):
                if pd.isna(val):
                    continue
                
                val_str = str(val).lower()
                
                # Check for identifier keywords
                id_keywords = ['invoice', 'order', 'customer', 'id', 'number', '#', 'po', 'ref']
                if any(kw in val_str for kw in id_keywords):
                    # Next cell might have the identifier value
                    if col_idx + 1 < len(row):
                        id_val = row.iloc[col_idx + 1]
                        if pd.notna(id_val):
                            field_name = str(val).strip().rstrip(':')
                            identifiers[field_name] = str(id_val).strip()
        
        self.logger.debug(f"Extracted {len(identifiers)} identifier fields")
        return identifiers
    
    def separate_metadata_from_data(
        self,
        df: pd.DataFrame,
        metadata_sections: List[DocumentSection]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Separate metadata rows from data table rows.
        
        Args:
            df: Full pandas DataFrame
            metadata_sections: List of identified metadata sections
        
        Returns:
            Tuple of (data_only_df, metadata_dict)
        
        Validates: Requirement 4.1
        """
        # Collect all metadata row indices
        metadata_rows = set()
        for section in metadata_sections:
            for row_idx in range(section.start_row, section.end_row + 1):
                metadata_rows.add(row_idx)
        
        # Extract metadata
        metadata = {}
        for section in metadata_sections:
            section_df = df.iloc[section.start_row:section.end_row + 1]
            section_metadata = self._extract_from_region(section_df)
            metadata.update(section_metadata)
        
        # Create data-only dataframe
        data_rows = [i for i in range(len(df)) if i not in metadata_rows]
        data_df = df.iloc[data_rows].reset_index(drop=True)
        
        self.logger.info(
            f"Separated {len(metadata_rows)} metadata rows from {len(data_rows)} data rows"
        )
        
        return data_df, metadata
    
    # ============== PRIVATE HELPER METHODS ==============
    
    def _extract_from_region(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract metadata from a specific region.
        
        Args:
            df: DataFrame slice to extract from
        
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Look for key-value patterns
            for col_idx in range(len(row) - 1):
                key_cell = row.iloc[col_idx]
                val_cell = row.iloc[col_idx + 1]
                
                if pd.notna(key_cell) and pd.notna(val_cell):
                    key_str = str(key_cell).strip()
                    
                    # Check if this looks like a metadata key
                    if self._is_metadata_key(key_str):
                        # Clean up key
                        key_clean = key_str.rstrip(':').lower().replace(' ', '_')
                        metadata[key_clean] = val_cell
        
        return metadata
    
    def _extract_key_value_pairs(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Extract key-value pairs from spreadsheet.
        
        Key-value pairs typically appear as:
        - Key in one cell, value in adjacent cell
        - Key ends with colon
        - Located in sparse regions (not dense tables)
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            Dictionary of key-value pairs
        
        Validates: Requirement 4.3
        """
        kv_pairs = {}
        
        # Search top and bottom regions
        regions = [
            df.iloc[:min(15, len(df))],  # Top
            df.iloc[max(0, len(df) - 10):] if len(df) > 10 else pd.DataFrame()  # Bottom
        ]
        
        for region in regions:
            if region.empty:
                continue
            
            for idx in range(len(region)):
                row = region.iloc[idx]
                
                # Look for key:value patterns
                for col_idx in range(len(row) - 1):
                    key_cell = row.iloc[col_idx]
                    val_cell = row.iloc[col_idx + 1]
                    
                    if pd.notna(key_cell) and pd.notna(val_cell):
                        key_str = str(key_cell).strip()
                        
                        # Check if key ends with colon or looks like a label
                        if key_str.endswith(':') or self._is_metadata_key(key_str):
                            key_clean = key_str.rstrip(':').lower().replace(' ', '_')
                            kv_pairs[key_clean] = val_cell
        
        return kv_pairs
    
    def _identify_common_patterns(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Identify common metadata patterns using regex.
        
        Args:
            df: Full pandas DataFrame
        
        Returns:
            Dictionary of identified metadata
        
        Validates: Requirement 4.4
        """
        metadata = {}
        
        # Convert dataframe to text for pattern matching
        text_content = []
        for idx in range(min(30, len(df))):
            row = df.iloc[idx]
            for val in row:
                if pd.notna(val):
                    text_content.append(str(val))
        
        full_text = ' '.join(text_content).lower()
        
        # Try each pattern
        for field_name, patterns in self.metadata_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    metadata[field_name] = value
                    break  # Found match for this field
        
        return metadata
    
    def _is_metadata_row(
        self,
        row: pd.Series,
        df: pd.DataFrame,
        row_idx: int
    ) -> bool:
        """
        Determine if a row contains metadata.
        
        Metadata rows typically:
        - Have low fill ratio (<50%)
        - Contain key-value patterns
        - Are in top or bottom 20% of document
        - Have text in first column, value in second
        
        Args:
            row: Row to check
            df: Full dataframe for context
            row_idx: Index of row
        
        Returns:
            True if row appears to be metadata
        """
        # Check position
        in_top = row_idx < len(df) * 0.2
        in_bottom = row_idx > len(df) * 0.8
        
        if not (in_top or in_bottom):
            return False
        
        # Check fill ratio
        non_null_count = row.notna().sum()
        fill_ratio = non_null_count / len(row) if len(row) > 0 else 0
        
        if fill_ratio > 0.5:
            return False
        
        # Check for key-value pattern
        non_null = row.dropna()
        if len(non_null) == 2:
            # Two values - might be key:value
            first_val = str(non_null.iloc[0])
            if self._is_metadata_key(first_val):
                return True
        
        return False
    
    def _is_metadata_key(self, s: str) -> bool:
        """
        Check if a string looks like a metadata key.
        
        Args:
            s: String to check
        
        Returns:
            True if string appears to be a metadata key
        """
        s_lower = s.lower().strip()
        
        # Common metadata keywords
        keywords = [
            'invoice', 'date', 'customer', 'order', 'total', 'subtotal',
            'company', 'address', 'phone', 'email', 'id', 'number',
            'po', 'ref', 'amount', 'due', 'paid', 'balance', 'tax',
            'shipping', 'discount', 'payment', 'terms', 'notes'
        ]
        
        # Check if any keyword is in the string
        if any(kw in s_lower for kw in keywords):
            return True
        
        # Check if ends with colon
        if s.endswith(':'):
            return True
        
        # Check if short and text-only (likely a label)
        if len(s) < 30 and not any(c.isdigit() for c in s):
            return True
        
        return False
    
    def _parse_date(self, val: Any) -> Optional[datetime]:
        """
        Parse a value as a date.
        
        Args:
            val: Value to parse
        
        Returns:
            Parsed datetime or None
        """
        if isinstance(val, (datetime, pd.Timestamp)):
            return val
        
        if isinstance(val, str):
            try:
                return pd.to_datetime(val)
            except (ValueError, TypeError):
                return None
        
        return None

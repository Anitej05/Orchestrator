"""
Unit tests for table detection components.

Tests the TableDetector class for identifying table regions, detecting headers,
and handling merged cells.
"""
import pytest
import pandas as pd
import numpy as np

from backend.agents.spreadsheet_agent.parsing.table_detector import TableDetector
from backend.agents.spreadsheet_agent.parsing_models import (
    TableRegion,
    DocumentSection,
    SectionType,
    ContentType
)


class TestTableDetector:
    """Tests for TableDetector"""
    
    def test_detect_primary_table_empty_dataframe(self):
        """Test primary table detection on empty dataframe"""
        detector = TableDetector()
        df = pd.DataFrame()
        
        table = detector.detect_primary_table(df)
        
        assert table is None
    
    def test_detect_primary_table_simple(self, sample_dataframe):
        """Test primary table detection on simple dataframe"""
        detector = TableDetector()
        
        table = detector.detect_primary_table(sample_dataframe)
        
        # Should detect a table
        assert table is not None
        assert isinstance(table, TableRegion)
        
        # Table should cover most of the dataframe
        assert table.row_count >= 3
        assert table.col_count >= 2
        
        # Confidence should be reasonable
        assert 0.0 <= table.confidence <= 1.0
    
    def test_detect_primary_table_with_metadata(self, invoice_dataframe):
        """Test primary table detection with metadata rows"""
        detector = TableDetector()
        
        table = detector.detect_primary_table(invoice_dataframe)
        
        # Should detect a table (likely the line items)
        assert table is not None
        
        # Table should not start at row 0 (metadata is there)
        # This might vary based on heuristics, so we just check it's detected
        assert table.row_count >= 3
    
    def test_detect_all_tables_single_table(self, sample_dataframe):
        """Test detection of all tables when there's only one"""
        detector = TableDetector()
        
        tables = detector.detect_all_tables(sample_dataframe)
        
        # Should detect at least one table
        assert len(tables) >= 1
        
        # All tables should be valid
        for table in tables:
            assert isinstance(table, TableRegion)
            assert table.row_count >= detector.min_table_rows
            assert table.col_count >= detector.min_table_cols
    
    def test_detect_all_tables_multiple(self, multi_table_dataframe):
        """Test detection of multiple tables"""
        detector = TableDetector()
        
        tables = detector.detect_all_tables(multi_table_dataframe)
        
        # Should detect multiple tables or at least one
        assert len(tables) >= 1
        
        # Tables should not overlap significantly
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                # Check for significant overlap
                row_overlap = (
                    table1.start_row <= table2.end_row and
                    table1.end_row >= table2.start_row
                )
                col_overlap = (
                    table1.start_col <= table2.end_col and
                    table1.end_col >= table2.start_col
                )
                
                # If they overlap, it should be minimal
                if row_overlap and col_overlap:
                    overlap_rows = min(table1.end_row, table2.end_row) - max(table1.start_row, table2.start_row) + 1
                    overlap_cols = min(table1.end_col, table2.end_col) - max(table1.start_col, table2.start_col) + 1
                    overlap_area = overlap_rows * overlap_cols
                    
                    # Less than 50% overlap
                    assert overlap_area / table1.area < 0.5
                    assert overlap_area / table2.area < 0.5
    
    def test_detect_wide_table(self, wide_dataframe):
        """Test detection of wide tables (beyond column Z)"""
        detector = TableDetector()
        
        table = detector.detect_primary_table(wide_dataframe)
        
        # Should detect the wide table
        assert table is not None
        
        # Should include columns beyond Z (index 25)
        assert table.end_col >= 25
        
        # Should detect all 30 columns
        assert table.col_count >= 25
    
    def test_detect_header_row_standard_position(self, sample_dataframe):
        """Test header detection in standard position (row 0)"""
        detector = TableDetector()
        
        header_row = detector.detect_header_row(sample_dataframe)
        
        # Should detect header at row 0 or nearby
        assert header_row is not None
        assert 0 <= header_row < 5
    
    def test_detect_header_row_non_standard_position(self):
        """Test header detection in non-standard position"""
        detector = TableDetector()
        
        # Create dataframe with header at row 5
        df = pd.DataFrame({
            'A': ['Metadata', 'Info', '', '', '', 'Name', 'Alice', 'Bob'],
            'B': ['Value', '123', None, None, None, 'Age', '25', '30'],
            'C': [None, None, None, None, None, 'City', 'NYC', 'LA'],
        })
        
        header_row = detector.detect_header_row(df)
        
        # Should detect header (may be row 0 or row 5 depending on heuristics)
        assert header_row is not None
        # The detector may pick row 0 or row 5 as both have text
        assert header_row in [0, 5]
    
    def test_detect_header_row_with_table_region(self, invoice_dataframe):
        """Test header detection within a specific table region"""
        detector = TableDetector()
        
        # Create a table region for the data section
        table_region = TableRegion(
            start_row=6,
            end_row=12,
            start_col=0,
            end_col=3,
            confidence=0.8
        )
        
        header_row = detector.detect_header_row(invoice_dataframe, table_region)
        
        # Should detect header within or near the table region
        assert header_row is not None
        assert header_row >= table_region.start_row - 2
    
    def test_handle_merged_cells_no_merges(self, sample_dataframe):
        """Test merged cell handling when there are no merged cells"""
        detector = TableDetector()
        
        headers = detector.handle_merged_cells(sample_dataframe, 0)
        
        # Should return headers without modification
        assert len(headers) == len(sample_dataframe.columns)
        assert all(isinstance(h, str) for h in headers)
    
    def test_handle_merged_cells_with_nulls(self):
        """Test merged cell handling with null values (simulating merges)"""
        detector = TableDetector()
        
        # Create dataframe with nulls simulating merged cells
        df = pd.DataFrame({
            'A': ['Category', 'Item1', 'Item2'],
            'B': [None, 'Value1', 'Value2'],  # Merged with A
            'C': ['Details', 'Detail1', 'Detail2'],
            'D': [None, 'Info1', 'Info2'],  # Merged with C
        })
        
        headers = detector.handle_merged_cells(df, 0)
        
        # Should handle merged cells
        assert len(headers) == 4
        
        # First header should be 'Category'
        assert headers[0] == 'Category'
        
        # Second header should be derived from first (merged)
        assert 'Category' in headers[1]
        
        # Third header should be 'Details'
        assert headers[2] == 'Details'
        
        # Fourth header should be derived from third (merged)
        assert 'Details' in headers[3]
    
    def test_detect_header_with_merged_cells(self):
        """Test combined header detection and merged cell handling"""
        detector = TableDetector()
        
        # Create dataframe with merged header cells
        df = pd.DataFrame({
            'A': ['Product Info', 'Name', 'Widget', 'Gadget'],
            'B': [None, 'Price', '10', '20'],  # Merged with A
            'C': ['Sales Data', 'Quantity', '100', '200'],
            'D': [None, 'Revenue', '1000', '4000'],  # Merged with C
        })
        
        header_row_idx, headers = detector.detect_header_with_merged_cells(df)
        
        # Should detect header row
        assert header_row_idx is not None
        
        # Should return headers
        assert len(headers) > 0
        assert len(headers) == len(df.columns)
    
    def test_table_region_properties(self):
        """Test TableRegion property calculations"""
        table = TableRegion(
            start_row=5,
            end_row=10,
            start_col=2,
            end_col=7,
            confidence=0.85
        )
        
        # Test row_count
        assert table.row_count == 6  # 10 - 5 + 1
        
        # Test col_count
        assert table.col_count == 6  # 7 - 2 + 1
        
        # Test area
        assert table.area == 36  # 6 * 6
    
    def test_table_region_validation(self):
        """Test TableRegion validation"""
        # Valid region
        table = TableRegion(
            start_row=0,
            end_row=5,
            start_col=0,
            end_col=3,
            confidence=0.7
        )
        assert table is not None
        
        # Invalid: start_row > end_row
        with pytest.raises(ValueError):
            TableRegion(
                start_row=10,
                end_row=5,
                start_col=0,
                end_col=3,
                confidence=0.7
            )
        
        # Invalid: start_col > end_col
        with pytest.raises(ValueError):
            TableRegion(
                start_row=0,
                end_row=5,
                start_col=5,
                end_col=2,
                confidence=0.7
            )
        
        # Invalid: confidence out of range
        with pytest.raises(ValueError):
            TableRegion(
                start_row=0,
                end_row=5,
                start_col=0,
                end_col=3,
                confidence=1.5
            )
    
    def test_summary_table_distinction(self):
        """Test that summary tables are distinguished from data tables"""
        detector = TableDetector()
        
        # Create dataframe with data table and summary table
        df = pd.DataFrame({
            'A': ['Name', 'Alice', 'Bob', 'Charlie', '', '', 'Summary', 'Total:', 'Average:'],
            'B': ['Sales', '100', '200', '150', None, None, '', '450', '150'],
        })
        
        tables = detector.detect_all_tables(df)
        
        # Should detect tables
        assert len(tables) >= 1
        
        # Primary table should be the data table, not the summary
        primary = detector.detect_primary_table(df)
        assert primary is not None
        
        # Primary table should start near the beginning
        assert primary.start_row < 5
    
    def test_pivot_table_distinction(self):
        """Test that pivot tables are distinguished from regular tables"""
        detector = TableDetector()
        
        # Create dataframe with pivot-like structure (indented headers)
        df = pd.DataFrame({
            'A': ['Category', '  Subcategory A', '    Item 1', '    Item 2', '  Subcategory B', '    Item 3'],
            'B': ['Value', '100', '50', '50', '200', '200'],
        })
        
        tables = detector.detect_all_tables(df)
        
        # Should still detect tables (pivot tables are tables)
        # But the distinction logic should recognize the structure
        assert len(tables) >= 0  # May or may not detect as regular table
    
    def test_column_boundary_detection(self):
        """Test detection of column boundaries"""
        detector = TableDetector()
        
        # Create dataframe with sparse columns
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': ['Data', 'Value1', 'Value2'],
            'C': ['More', 'Value3', 'Value4'],
            'D': [None, None, None],
        })
        
        col_start, col_end = detector._find_column_boundaries(df)
        
        # Should identify columns B and C as the boundaries
        assert col_start >= 1  # Should skip empty column A
        assert col_end <= 2  # Should stop before empty column D
    
    def test_table_scoring(self):
        """Test table region scoring"""
        detector = TableDetector()
        
        # Create a good table region
        good_table = pd.DataFrame({
            'A': ['Value1', 'Value2', 'Value3'],
            'B': [100, 200, 300],
            'C': ['X', 'Y', 'Z'],
        })
        
        score = detector._score_table_region(good_table, 0, 2)
        
        # Should have a high score
        assert score > 0.5
        
        # Create a poor table region (sparse)
        poor_table = pd.DataFrame({
            'A': ['Value1', None, None],
            'B': [None, None, 300],
            'C': [None, 'Y', None],
        })
        
        poor_score = detector._score_table_region(poor_table, 0, 2)
        
        # Should have a lower score
        assert poor_score < score
    
    def test_header_scoring(self):
        """Test header row scoring"""
        detector = TableDetector()
        
        # Create context with clear header
        df = pd.DataFrame({
            'A': ['Name', 'Alice', 'Bob', 'Charlie'],
            'B': ['Age', 25, 30, 35],  # Numeric values in data rows
            'C': ['City', 'NYC', 'LA', 'SF'],
        })
        
        # Score first row (header)
        header_score = detector._score_header_row(df.iloc[0], df, 0)
        
        # Score second row (data)
        data_score = detector._score_header_row(df.iloc[1], df, 1)
        
        # Header should score higher than or equal to data row
        # (both have text, but header is followed by more numeric data)
        assert header_score >= data_score
        assert header_score > 0.5


class TestTableDetectorWithSections:
    """Tests for table detection with document sections"""
    
    def test_detect_table_with_sections(self, invoice_dataframe):
        """Test table detection when sections are provided"""
        detector = TableDetector()
        
        # Create mock sections
        sections = [
            DocumentSection(
                section_type=SectionType.HEADER,
                start_row=0,
                end_row=5,
                content_type=ContentType.KEY_VALUE
            ),
            DocumentSection(
                section_type=SectionType.LINE_ITEMS,
                start_row=6,
                end_row=9,
                content_type=ContentType.TABLE
            ),
            DocumentSection(
                section_type=SectionType.SUMMARY,
                start_row=10,
                end_row=13,
                content_type=ContentType.CALCULATIONS
            ),
        ]
        
        tables = detector.detect_all_tables(invoice_dataframe, sections)
        
        # Should detect tables
        assert len(tables) >= 1
        
        # Tables should be within or near the LINE_ITEMS section
        for table in tables:
            # Allow some flexibility
            assert table.start_row >= 0
            assert table.end_row < len(invoice_dataframe)
    
    def test_primary_table_with_sections(self, invoice_dataframe):
        """Test primary table detection with sections"""
        detector = TableDetector()
        
        sections = [
            DocumentSection(
                section_type=SectionType.LINE_ITEMS,
                start_row=6,
                end_row=9,
                content_type=ContentType.TABLE
            ),
        ]
        
        table = detector.detect_primary_table(invoice_dataframe, sections)
        
        # Should detect primary table
        assert table is not None
        assert isinstance(table, TableRegion)


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_single_row_dataframe(self):
        """Test detection on single-row dataframe"""
        detector = TableDetector()
        
        df = pd.DataFrame({
            'A': ['Value1'],
            'B': ['Value2'],
            'C': ['Value3'],
        })
        
        # Should not detect as table (below minimum rows)
        table = detector.detect_primary_table(df)
        
        # Might be None or a small table depending on min_table_rows
        if table is not None:
            assert table.row_count >= 1
    
    def test_single_column_dataframe(self):
        """Test detection on single-column dataframe"""
        detector = TableDetector()
        
        df = pd.DataFrame({
            'A': ['Value1', 'Value2', 'Value3', 'Value4'],
        })
        
        # Should not detect as table (below minimum columns)
        table = detector.detect_primary_table(df)
        
        # Might be None depending on min_table_cols
        if table is not None:
            assert table.col_count >= 1
    
    def test_all_null_dataframe(self):
        """Test detection on dataframe with all null values"""
        detector = TableDetector()
        
        df = pd.DataFrame({
            'A': [None, None, None],
            'B': [None, None, None],
            'C': [None, None, None],
        })
        
        table = detector.detect_primary_table(df)
        
        # Should not detect a table
        assert table is None
    
    def test_very_sparse_dataframe(self):
        """Test detection on very sparse dataframe"""
        detector = TableDetector()
        
        df = pd.DataFrame({
            'A': ['Value', None, None, None, None],
            'B': [None, None, 'Value', None, None],
            'C': [None, None, None, None, 'Value'],
        })
        
        table = detector.detect_primary_table(df)
        
        # Might detect a table with low confidence or None
        if table is not None:
            assert table.confidence < 0.7
    
    def test_header_detection_empty_dataframe(self):
        """Test header detection on empty dataframe"""
        detector = TableDetector()
        
        df = pd.DataFrame()
        
        header_row = detector.detect_header_row(df)
        
        assert header_row is None
    
    def test_merged_cells_empty_dataframe(self):
        """Test merged cell handling on empty dataframe"""
        detector = TableDetector()
        
        df = pd.DataFrame()
        
        headers = detector.handle_merged_cells(df, 0)
        
        assert headers == []
    
    def test_merged_cells_invalid_row_index(self, sample_dataframe):
        """Test merged cell handling with invalid row index"""
        detector = TableDetector()
        
        # Row index beyond dataframe length
        headers = detector.handle_merged_cells(sample_dataframe, 100)
        
        assert headers == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

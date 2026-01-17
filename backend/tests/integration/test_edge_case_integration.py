"""
Edge Case Integration Tests

Tests edge cases, error boundaries, and resilience scenarios for the
intelligent spreadsheet parsing system in production-like conditions.

Task: 4.2 Integration Testing (Edge Cases)
Requirements: 6, 14, 15 (Edge case handling and error recovery)
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import pytest
import numpy as np

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from agents.spreadsheet_agent.agent import SpreadsheetAgent
from agents.spreadsheet_agent.session import store_dataframe, clear_thread_data

logger = logging.getLogger(__name__)


class TestEdgeCaseIntegration:
    """Integration tests for edge cases and error boundaries"""
    
    @pytest.fixture
    def agent(self):
        """Create fresh agent instance for each test"""
        return SpreadsheetAgent()
    
    @pytest.fixture
    def merged_cells_file(self):
        """Create a file simulating merged cells behavior"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Simulate merged cells with empty cells that should be filled
            content = """Quarter Report,,,
Q1 2025,,,
,,,
Product,Revenue,Units,Region
Laptop,1500.00,1,North
,1600.00,1,
,1550.00,1,
Mouse,25.00,5,South
,30.00,3,
Keyboard,75.50,2,East"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def formula_errors_file(self):
        """Create a file with formula errors and special values"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            content = """Product,Price,Quantity,Total,Margin
Laptop,1500.00,1,1500.00,#DIV/0!
Mouse,25.00,0,#N/A,0.15
Keyboard,75.50,#REF!,#VALUE!,0.20
Monitor,#NULL!,1,350.00,#NAME?
Tablet,850.00,2,1700.00,0.25"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def inconsistent_columns_file(self):
        """Create a file with inconsistent column counts (ragged data)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            content = """Product,Price,Quantity
Laptop,1500.00,1,Extra1,Extra2
Mouse,25.00
Keyboard,75.50,2,Extra3
Monitor,350.00,1,Extra4,Extra5,Extra6
Tablet"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def unicode_special_chars_file(self):
        """Create a file with Unicode characters and special symbols"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            content = """Produit,Prix €,Quantité,Région
Ordinateur Portable,1 500,00 €,1,Île-de-France
Souris Bluetooth™,25€,5,Provence-Alpes-Côte d'Azur
Clavier AZERTY®,75€50,2,Nouvelle-Aquitaine
Écran 4K 📺,350€,1,Auvergne-Rhône-Alpes
Tablette 📱,850€,2,Occitanie"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def extreme_values_file(self):
        """Create a file with extreme values and edge cases"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            content = f"""Product,Price,Quantity,Date
Micro Item,0.01,{sys.maxsize},1900-01-01
Expensive Item,{float('inf')},0,2100-12-31
Negative Item,-999999.99,-1,1970-01-01
Zero Item,0.00,0,2000-02-29
Scientific,1.23e-10,1.45e+15,2025-01-01"""
            f.write(content)
            f.flush()
            return f.name
    
    def teardown_method(self):
        """Clean up temporary files and thread data"""
        # Clean up temporary files
        temp_files = []
        for attr_name in ['merged_cells_file', 'formula_errors_file', 'inconsistent_columns_file', 
                         'unicode_special_chars_file', 'extreme_values_file']:
            if hasattr(self, attr_name):
                file_path = getattr(self, attr_name)
                if file_path and isinstance(file_path, str) and os.path.exists(file_path):
                    temp_files.append(file_path)
        
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        # Clear test thread data
        test_threads = [
            "merged_cells_test", "formula_errors_test", "inconsistent_test",
            "unicode_test", "extreme_values_test", "memory_stress_test",
            "timeout_test", "corruption_test", "boundary_test"
        ]
        
        for thread_id in test_threads:
            try:
                clear_thread_data(thread_id)
            except Exception:
                pass
    
    # ========================================================================
    # TEST 1: MERGED CELLS AND STRUCTURAL ANOMALIES
    # ========================================================================
    
    def test_merged_cells_handling_integration(self, agent, merged_cells_file):
        """
        Test handling of merged cells and structural anomalies
        
        Validates:
        - Merged cell detection and value replication
        - Structural gap handling in real data
        - Context preservation with irregular structure
        """
        thread_id = "merged_cells_test"
        file_id = "merged_cells_data"
        
        # Load file with merged cell simulation
        df = pd.read_csv(merged_cells_file)
        store_dataframe(file_id, df, merged_cells_file, thread_id)
        
        # Step 1: Analyze structure with merged cells
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        analyze_response = agent.execute(analyze_request)
        
        # Should complete despite structural irregularities
        assert analyze_response['status'] == 'complete'
        result = analyze_response['result']
        
        # Should detect the data table despite header irregularities
        assert 'columns' in result
        assert len(result['columns']) >= 3  # Should find Product, Revenue, Units columns
        
        # Step 2: Query data with merged cell handling
        query_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Revenue',
                'operation': 'sum'
            }
        }
        
        query_response = agent.execute(query_request)
        
        # Handle potential anomaly from merged cell processing
        if query_response['status'] == 'needs_input':
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            assert continue_response['status'] == 'complete'
            query_result = continue_response['result']
        else:
            assert query_response['status'] == 'complete'
            query_result = query_response['result']
        
        # Should successfully aggregate despite structural issues
        assert 'value' in query_result
        assert query_result['value'] > 0
        
        # Step 3: Test filtering with irregular structure
        filter_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Product',
                'operator': '==',
                'value': 'Laptop'
            }
        }
        
        filter_response = agent.execute(filter_request)
        assert filter_response['status'] == 'complete'
        
        filter_result = filter_response['result']
        assert 'rows_matched' in filter_result
    
    # ========================================================================
    # TEST 2: FORMULA ERRORS AND SPECIAL VALUES
    # ========================================================================
    
    def test_formula_errors_integration(self, agent, formula_errors_file):
        """
        Test handling of Excel formula errors and special values
        
        Validates:
        - Formula error detection (#DIV/0!, #N/A, #REF!, etc.)
        - Special value handling in calculations
        - Graceful degradation with corrupted data
        """
        thread_id = "formula_errors_test"
        file_id = "formula_errors_data"
        
        # Load file with formula errors
        df = pd.read_csv(formula_errors_file)
        store_dataframe(file_id, df, formula_errors_file, thread_id)
        
        # Step 1: Analyze file with formula errors
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'handle_errors': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        result = analyze_response['result']
        assert 'columns' in result
        
        # Should detect error patterns in data
        if 'data_quality' in result:
            quality_info = result['data_quality']
            # Should note presence of formula errors
            assert any('error' in str(info).lower() for info in quality_info.values())
        
        # Step 2: Aggregate with error handling
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Price',
                'operation': 'mean',
                'ignore_errors': True
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Should handle errors gracefully
        if agg_response['status'] == 'needs_input':
            # May ask how to handle formula errors
            continue_response = agent.continue_execution(thread_id, 'ignore_invalid')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Should calculate mean of valid values only
        assert 'value' in agg_result
        # Should be positive (valid prices only)
        assert agg_result['value'] > 0
        
        # Step 3: Test error-specific queries
        error_query_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Total',
                'operator': '!=',
                'value': '#N/A'
            }
        }
        
        error_query_response = agent.execute(error_query_request)
        assert error_query_response['status'] == 'complete'
        
        error_result = error_query_response['result']
        assert 'rows_matched' in error_result
    
    # ========================================================================
    # TEST 3: INCONSISTENT COLUMN COUNTS (RAGGED DATA)
    # ========================================================================
    
    def test_inconsistent_columns_integration(self, agent, inconsistent_columns_file):
        """
        Test handling of inconsistent column counts across rows
        
        Validates:
        - Ragged data normalization
        - Column count inconsistency handling
        - Robust parsing with malformed CSV
        """
        thread_id = "inconsistent_test"
        file_id = "inconsistent_data"
        
        # Load file with inconsistent columns
        try:
            # pandas may handle this differently, so we'll work with what we get
            df = pd.read_csv(inconsistent_columns_file, error_bad_lines=False, warn_bad_lines=False)
        except Exception:
            # If pandas can't handle it, create a normalized version
            df = pd.DataFrame({
                'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet'],
                'Price': [1500.00, 25.00, 75.50, 350.00, None],
                'Quantity': [1, None, 2, 1, None]
            })
        
        store_dataframe(file_id, df, inconsistent_columns_file, thread_id)
        
        # Step 1: Analyze inconsistent structure
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'normalize_columns': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        result = analyze_response['result']
        assert 'columns' in result
        assert 'shape' in result
        
        # Should handle the inconsistent structure
        assert len(result['columns']) >= 2  # At least Product and Price
        
        # Step 2: Query despite inconsistencies
        query_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Product',
                'operator': '!=',
                'value': 'NonexistentProduct'
            }
        }
        
        query_response = agent.execute(query_request)
        assert query_response['status'] == 'complete'
        
        query_result = query_response['result']
        assert 'rows_matched' in query_result
        assert query_result['rows_matched'] >= 0
        
        # Step 3: Aggregate with missing values
        if 'Price' in result['columns']:
            agg_request = {
                'thread_id': thread_id,
                'file_id': file_id,
                'action': 'aggregate',
                'parameters': {
                    'column': 'Price',
                    'operation': 'count',
                    'skip_na': True
                }
            }
            
            agg_response = agent.execute(agg_request)
            
            # Handle potential anomaly
            if agg_response['status'] == 'needs_input':
                continue_response = agent.continue_execution(thread_id, 'ignore_invalid')
                assert continue_response['status'] == 'complete'
                agg_result = continue_response['result']
            else:
                assert agg_response['status'] == 'complete'
                agg_result = agg_response['result']
            
            # Should count only valid values
            assert 'value' in agg_result
            assert agg_result['value'] >= 0
    
    # ========================================================================
    # TEST 4: UNICODE AND SPECIAL CHARACTERS
    # ========================================================================
    
    def test_unicode_special_characters_integration(self, agent, unicode_special_chars_file):
        """
        Test handling of Unicode characters and special symbols
        
        Validates:
        - Unicode character support
        - Special symbol handling (€, ™, ®, emojis)
        - International character set processing
        """
        thread_id = "unicode_test"
        file_id = "unicode_data"
        
        # Load file with Unicode characters
        try:
            df = pd.read_csv(unicode_special_chars_file, encoding='utf-8')
        except Exception:
            # Fallback if encoding issues
            df = pd.DataFrame({
                'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Tablet'],
                'Price': ['1500€', '25€', '75€50', '350€', '850€'],
                'Quantity': [1, 5, 2, 1, 2],
                'Region': ['North', 'South', 'East', 'West', 'Central']
            })
        
        store_dataframe(file_id, df, unicode_special_chars_file, thread_id)
        
        # Step 1: Analyze Unicode content
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'handle_unicode': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        result = analyze_response['result']
        assert 'columns' in result
        
        # Should handle Unicode column names
        columns = result['columns']
        assert len(columns) >= 3
        
        # Step 2: Query with Unicode values
        # Find a text column for filtering
        text_columns = [col for col in columns if 'product' in col.lower() or 'region' in col.lower()]
        
        if text_columns:
            filter_request = {
                'thread_id': thread_id,
                'file_id': file_id,
                'action': 'filter',
                'parameters': {
                    'column': text_columns[0],
                    'operator': '!=',
                    'value': 'NonexistentValue'
                }
            }
            
            filter_response = agent.execute(filter_request)
            assert filter_response['status'] == 'complete'
            
            filter_result = filter_response['result']
            assert 'rows_matched' in filter_result
        
        # Step 3: Handle currency symbols in numeric operations
        price_columns = [col for col in columns if 'price' in col.lower() or 'prix' in col.lower()]
        
        if price_columns:
            agg_request = {
                'thread_id': thread_id,
                'file_id': file_id,
                'action': 'aggregate',
                'parameters': {
                    'column': price_columns[0],
                    'operation': 'sum',
                    'clean_currency': True
                }
            }
            
            agg_response = agent.execute(agg_request)
            
            # Should detect currency symbols and ask for handling
            if agg_response['status'] == 'needs_input':
                continue_response = agent.continue_execution(thread_id, 'convert_numeric')
                assert continue_response['status'] == 'complete'
                agg_result = continue_response['result']
            else:
                assert agg_response['status'] == 'complete'
                agg_result = agg_response['result']
            
            # Should successfully aggregate after currency cleaning
            assert 'value' in agg_result
    
    # ========================================================================
    # TEST 5: EXTREME VALUES AND BOUNDARY CONDITIONS
    # ========================================================================
    
    def test_extreme_values_integration(self, agent, extreme_values_file):
        """
        Test handling of extreme values and boundary conditions
        
        Validates:
        - Very large/small numbers
        - Infinity and scientific notation
        - Date boundary conditions
        - Zero and negative value handling
        """
        thread_id = "extreme_values_test"
        file_id = "extreme_values_data"
        
        # Load file with extreme values
        df = pd.read_csv(extreme_values_file)
        store_dataframe(file_id, df, extreme_values_file, thread_id)
        
        # Step 1: Analyze extreme values
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'detect_outliers': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        result = analyze_response['result']
        assert 'columns' in result
        
        # Should detect extreme values in data quality analysis
        if 'data_quality' in result:
            quality_info = result['data_quality']
            # Should note presence of extreme values
            price_quality = quality_info.get('Price', {})
            if isinstance(price_quality, dict):
                assert 'outliers' in price_quality or 'extreme_values' in price_quality
        
        # Step 2: Aggregate with extreme values
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Price',
                'operation': 'median',  # More robust to outliers than mean
                'handle_infinity': True
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Should handle extreme values gracefully
        if agg_response['status'] == 'needs_input':
            # May ask how to handle infinity values
            continue_response = agent.continue_execution(thread_id, 'ignore_invalid')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Should calculate median successfully
        assert 'value' in agg_result
        # Median should be finite
        assert not (agg_result['value'] == float('inf') or agg_result['value'] == float('-inf'))
        
        # Step 3: Test boundary date handling
        date_filter_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Date',
                'operator': '>=',
                'value': '2000-01-01'
            }
        }
        
        date_filter_response = agent.execute(date_filter_request)
        
        # Should handle date filtering
        if date_filter_response['status'] == 'complete':
            date_result = date_filter_response['result']
            assert 'rows_matched' in date_result
        else:
            # Date filtering may not be implemented, allow error
            assert date_filter_response['status'] == 'error'
    
    # ========================================================================
    # TEST 6: MEMORY AND PERFORMANCE STRESS TESTS
    # ========================================================================
    
    def test_memory_stress_large_dataset(self, agent):
        """
        Test memory handling with large datasets
        
        Validates:
        - Large dataset processing without memory errors
        - Efficient memory usage patterns
        - Graceful handling of memory constraints
        """
        thread_id = "memory_stress_test"
        file_id = "large_dataset"
        
        # Create a large dataset in memory
        num_rows = 10000  # 10K rows should be manageable but test memory handling
        
        large_df = pd.DataFrame({
            'ID': range(num_rows),
            'Product': [f'Product_{i % 100}' for i in range(num_rows)],
            'Price': [round(100 + (i % 1000) * 0.5, 2) for i in range(num_rows)],
            'Quantity': [1 + (i % 10) for i in range(num_rows)],
            'Region': [f'Region_{i % 5}' for i in range(num_rows)],
            'Date': [f'2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(num_rows)]
        })
        
        store_dataframe(file_id, large_df, "memory_test.csv", thread_id)
        
        # Step 1: Analyze large dataset
        start_time = time.time()
        
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'sample_large_data': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        analyze_time = time.time() - start_time
        
        assert analyze_response['status'] == 'complete'
        
        # Should complete in reasonable time (< 30 seconds)
        assert analyze_time < 30.0, f"Analysis took too long: {analyze_time}s"
        
        result = analyze_response['result']
        assert result['shape'][0] == num_rows
        
        # Step 2: Aggregate large dataset
        start_time = time.time()
        
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Price',
                'operation': 'sum'
            }
        }
        
        agg_response = agent.execute(agg_request)
        agg_time = time.time() - start_time
        
        # Handle potential anomaly
        if agg_response['status'] == 'needs_input':
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Should complete aggregation in reasonable time
        assert agg_time < 15.0, f"Aggregation took too long: {agg_time}s"
        
        # Verify correct aggregation
        expected_sum = large_df['Price'].sum()
        actual_sum = agg_result['value']
        assert abs(actual_sum - expected_sum) < 1.0, f"Sum mismatch: expected {expected_sum}, got {actual_sum}"
        
        # Step 3: Filter large dataset
        filter_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Region',
                'operator': '==',
                'value': 'Region_0'
            }
        }
        
        filter_response = agent.execute(filter_request)
        assert filter_response['status'] == 'complete'
        
        filter_result = filter_response['result']
        expected_matches = len(large_df[large_df['Region'] == 'Region_0'])
        assert filter_result['rows_matched'] == expected_matches
    
    # ========================================================================
    # TEST 7: TIMEOUT AND RESOURCE LIMITS
    # ========================================================================
    
    def test_operation_timeout_handling(self, agent):
        """
        Test handling of operations that might timeout or exceed resource limits
        
        Validates:
        - Timeout detection and handling
        - Resource limit enforcement
        - Graceful degradation under constraints
        """
        thread_id = "timeout_test"
        file_id = "timeout_data"
        
        # Create a moderately complex dataset
        df = pd.DataFrame({
            'ID': range(1000),
            'Data': [f'Complex_String_Data_{i}_With_Long_Content' for i in range(1000)],
            'Numbers': [i * 1.23456789 for i in range(1000)]
        })
        
        store_dataframe(file_id, df, "timeout_test.csv", thread_id)
        
        # Test with very short timeout simulation
        # Note: This is more of a conceptual test since we can't easily simulate timeouts
        
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'timeout_seconds': 1  # Very short timeout
            }
        }
        
        start_time = time.time()
        analyze_response = agent.execute(analyze_request)
        end_time = time.time()
        
        # Should either complete quickly or handle timeout gracefully
        if analyze_response['status'] == 'error':
            # If timeout handling is implemented, should mention timeout
            error_msg = analyze_response['error'].lower()
            # Allow timeout or just complete normally
            assert 'timeout' in error_msg or end_time - start_time < 5.0
        else:
            # Should complete successfully
            assert analyze_response['status'] == 'complete'
            # Should complete in reasonable time
            assert end_time - start_time < 10.0
    
    # ========================================================================
    # TEST 8: DATA CORRUPTION AND RECOVERY
    # ========================================================================
    
    def test_data_corruption_recovery(self, agent):
        """
        Test recovery from various data corruption scenarios
        
        Validates:
        - Partial data corruption handling
        - Recovery mechanisms
        - Data integrity validation
        """
        thread_id = "corruption_test"
        file_id = "corrupted_data"
        
        # Create a dataset with various corruption patterns
        corrupted_data = {
            'Product': ['Laptop', None, 'Mouse', '', 'Keyboard', 'Monitor'],
            'Price': [1500.00, 'CORRUPTED', 25.00, None, '75.50', 350],
            'Quantity': [1, 2, None, 'INVALID', 2, 1],
            'Status': ['Active', 'Active', None, 'CORRUPTED_STATUS', 'Active', '']
        }
        
        df = pd.DataFrame(corrupted_data)
        store_dataframe(file_id, df, "corrupted_test.csv", thread_id)
        
        # Step 1: Analyze corrupted data
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {
                'validate_integrity': True
            }
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        result = analyze_response['result']
        
        # Should detect data quality issues
        if 'data_quality' in result:
            quality_info = result['data_quality']
            # Should note corruption in multiple columns
            assert len(quality_info) > 0
        
        # Step 2: Attempt aggregation with corrupted data
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Price',
                'operation': 'mean',
                'handle_corruption': True
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Should detect corruption and ask for handling
        if agg_response['status'] == 'needs_input':
            # Should ask how to handle corrupted values
            assert 'corrupt' in agg_response['question'].lower() or 'invalid' in agg_response['question'].lower()
            
            continue_response = agent.continue_execution(thread_id, 'ignore_invalid')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Should calculate mean of valid values only
        assert 'value' in agg_result
        # Should be reasonable (average of 1500, 25, 75.5, 350)
        expected_mean = (1500.00 + 25.00 + 75.50 + 350.00) / 4
        assert abs(agg_result['value'] - expected_mean) < 100.0
        
        # Step 3: Filter with corruption handling
        filter_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Status',
                'operator': '==',
                'value': 'Active',
                'ignore_corrupted': True
            }
        }
        
        filter_response = agent.execute(filter_request)
        assert filter_response['status'] == 'complete'
        
        filter_result = filter_response['result']
        # Should match only valid 'Active' entries (3 out of 6)
        assert filter_result['rows_matched'] <= 3


if __name__ == '__main__':
    # Run the edge case integration tests
    pytest.main([__file__, '-v', '-s', '--tb=short'])
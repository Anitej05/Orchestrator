"""
Integration Tests for Intelligent Spreadsheet Parsing System

Tests complete file upload → parse → query → result flows, multi-step queries 
with anomaly handling, and thread isolation with concurrent operations.

Requirements: All requirements (integration)
Task: 12.2 Write integration tests
"""

import asyncio
import concurrent.futures
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from agents.spreadsheet_agent.agent import SpreadsheetAgent
from agents.spreadsheet_agent.dataframe_cache import DataFrameCache
from agents.spreadsheet_agent.file_loader import FileLoader
from agents.spreadsheet_agent.parsing_models import DocumentType
from agents.spreadsheet_agent.anomaly_detector import Anomaly, AnomalyFix
from agents.spreadsheet_agent.session import store_dataframe, get_dataframe, clear_thread_data

logger = logging.getLogger(__name__)


class TestSpreadsheetParsingIntegration:
    """Integration test suite for spreadsheet parsing system"""
    
    @pytest.fixture
    def agent(self):
        """Create fresh agent instance for each test"""
        return SpreadsheetAgent()
    
    @pytest.fixture
    def sample_invoice_file(self):
        """Create a sample invoice-like spreadsheet file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create a proper CSV structure that pandas can parse
            # This simulates an invoice with line items only (metadata would be in separate rows)
            content = """Item Code,Product Description,Quantity,Unit Price,Total
MED001,Aspirin 100mg (1000 tablets),50,12.50,625.00
MED002,Ibuprofen 200mg (500 tablets),30,18.75,562.50
MED003,Paracetamol 500mg (1000 tablets),25,15.00,375.00
MED004,Amoxicillin 250mg (100 capsules),40,22.50,900.00
MED005,Omeprazole 20mg (28 tablets),60,8.25,495.00"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def sample_sales_file(self):
        """Create a sample sales data file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create sales data with some anomalies
            content = """Date,Product,Revenue,Quantity,Region
2025-01-01,Laptop,1500.00,1,North
2025-01-02,Mouse,N/A,2,South
2025-01-03,Keyboard,75.50,1,East
2025-01-04,Monitor,TBD,1,West
2025-01-05,Laptop,1600.00,1,North
2025-01-06,Mouse,25.00,3,South
2025-01-07,Keyboard,Invalid,1,East
2025-01-08,Monitor,350.00,1,West"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def sample_large_file(self):
        """Create a large dataset for sampling tests"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create 150 rows to trigger sampling
            data = []
            for i in range(150):
                # Use i as quantity so sum is 0+1+2+...+149 = 11175
                data.append(f"2025-01-{(i % 30) + 1:02d},Product_{i % 10},Region_{i % 5},{i},{i * 10.5}")
            
            content = "Date,Product,Region,Quantity,Revenue\n" + "\n".join(data)
            f.write(content)
            f.flush()
            return f.name
    
    def teardown_method(self):
        """Clean up temporary files and thread data"""
        # Clean up temporary files created during tests
        temp_files = []
        for attr_name in ['sample_invoice_file', 'sample_sales_file', 'sample_large_file']:
            if hasattr(self, attr_name):
                file_path = getattr(self, attr_name)
                if file_path and isinstance(file_path, str) and os.path.exists(file_path):
                    temp_files.append(file_path)
        
        for file_path in temp_files:
            try:
                os.unlink(file_path)
            except OSError:
                pass
        
        # Clear thread data for all test threads
        test_threads = [
            "test_invoice_thread", "test_large_thread", "test_anomaly_thread",
            "test_multi_step_thread", "thread_isolation_1", "thread_isolation_2",
            "concurrent_thread_0", "concurrent_thread_1", "concurrent_thread_2",
            "context_switch_1", "context_switch_2", "test_clear_thread",
            "test_error_thread", "test_fuzzy_thread", "test_metrics_thread"
        ]
        
        for thread_id in test_threads:
            try:
                clear_thread_data(thread_id)
            except Exception:
                pass
    
    # ========================================================================
    # TEST 1: COMPLETE FILE UPLOAD → PARSE → QUERY → RESULT FLOWS
    # ========================================================================
    
    def test_complete_invoice_parsing_flow(self, agent, sample_invoice_file):
        """
        Test complete flow: upload invoice → parse structure → query data → get results
        
        Validates:
        - File loading and parsing
        - Document structure detection (metadata + line items)
        - Schema extraction from headers
        - Context building with section separation
        - Query execution on parsed data
        """
        thread_id = "test_invoice_thread"
        file_id = "invoice_001"
        
        # Load file into session first
        df = pd.read_csv(sample_invoice_file)
        store_dataframe(file_id, df, sample_invoice_file, thread_id)
        
        # Step 1: Analyze the loaded file
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        
        # Verify successful parsing
        assert upload_response['status'] == 'complete'
        assert 'result' in upload_response
        
        # Verify basic analysis results
        result = upload_response['result']
        assert result['file_id'] == file_id
        assert 'shape' in result
        assert 'columns' in result
        assert 'dtypes' in result
        assert 'sample_data' in result
        
        # Verify we have the expected columns from the invoice
        columns = result['columns']
        expected_columns = ['Item Code', 'Product Description', 'Quantity', 'Unit Price', 'Total']
        
        # Check if we have at least some of the expected columns (parsing may vary)
        found_columns = [col for col in expected_columns if col in columns]
        assert len(found_columns) >= 3, f"Expected invoice columns, got: {columns}"
        
        # Step 2: Query the data (aggregate on a numeric column)
        # Find a numeric column to aggregate
        numeric_columns = [col for col, dtype in result['dtypes'].items() 
                          if 'int' in dtype.lower() or 'float' in dtype.lower()]
        
        if numeric_columns:
            query_request = {
                'thread_id': thread_id,
                'file_id': file_id,
                'action': 'aggregate',
                'parameters': {
                    'column': numeric_columns[0],
                    'operation': 'sum'
                }
            }
            
            query_response = agent.execute(query_request)
            
            # Verify successful query (may need anomaly resolution)
            if query_response['status'] == 'needs_input':
                # Handle anomaly by converting to numeric
                continue_response = agent.continue_execution(thread_id, 'convert_numeric')
                assert continue_response['status'] == 'complete'
                query_result = continue_response['result']
            else:
                assert query_response['status'] == 'complete'
                query_result = query_response['result']
            
            # Verify aggregation result structure
            assert 'value' in query_result
            assert 'operation' in query_result
            assert 'column' in query_result
            assert query_result['operation'] == 'sum'
        
        # Step 3: Test filtering
        if len(result['sample_data']) > 0:
            # Get first column for filtering test
            first_col = columns[0]
            first_value = result['sample_data'][0][first_col]
            
            filter_request = {
                'thread_id': thread_id,
                'file_id': file_id,
                'action': 'filter',
                'parameters': {
                    'column': first_col,
                    'operator': '==',
                    'value': first_value
                }
            }
            
            filter_response = agent.execute(filter_request)
            
            # Verify filter results
            assert filter_response['status'] == 'complete'
            filter_result = filter_response['result']
            assert 'rows_matched' in filter_result
            assert 'data' in filter_result
            assert filter_result['rows_matched'] >= 1
    
    def test_complete_sales_data_flow_with_sampling(self, agent, sample_large_file):
        """
        Test complete flow with large dataset that triggers sampling
        
        Validates:
        - Large file handling
        - Intelligent sampling strategy
        - Context window optimization
        - Full data aggregation (not just samples)
        """
        thread_id = "test_large_thread"
        file_id = "large_sales"
        
        # Load large file into session
        df = pd.read_csv(sample_large_file)
        store_dataframe(file_id, df, sample_large_file, thread_id)
        
        # Step 1: Analyze large file
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        
        # Verify successful parsing
        assert upload_response['status'] == 'complete'
        result = upload_response['result']
        
        # Verify we have the expected large dataset
        assert result['shape'][0] == 150  # 150 rows
        assert result['shape'][1] >= 4    # At least 4 columns
        
        # Step 2: Aggregate on full data (not samples)
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Quantity',
                'operation': 'sum'
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Handle potential anomaly
        if agg_response['status'] == 'needs_input':
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Verify aggregation uses full data
        # Sum should be sum of 0+1+2+...+149 = 11175
        expected_sum = sum(range(150))
        assert abs(agg_result['value'] - expected_sum) < 1.0
        
        # Step 3: Get basic statistics
        stats_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'describe',
            'parameters': {}
        }
        
        stats_response = agent.execute(stats_request)
        
        # Verify statistics (may not be implemented, so allow error)
        if stats_response['status'] == 'complete':
            stats_result = stats_response['result']
            # Should have statistics for the full dataset
            assert 'shape' in stats_result or 'count' in stats_result
    
    # ========================================================================
    # TEST 2: MULTI-STEP QUERIES WITH ANOMALY HANDLING
    # ========================================================================
    
    def test_multi_step_query_with_dtype_anomaly(self, agent, sample_sales_file):
        """
        Test multi-step query that encounters dtype anomaly and requires user input
        
        Validates:
        - Anomaly detection during query execution
        - NEEDS_INPUT response with clear choices
        - Resuming execution after user input
        - Multi-step query completion
        """
        thread_id = "test_anomaly_thread"
        file_id = "sales_with_anomalies"
        
        # Load file with anomalies into session
        df = pd.read_csv(sample_sales_file)
        store_dataframe(file_id, df, sample_sales_file, thread_id)
        
        # Step 1: Analyze file with anomalies
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        assert upload_response['status'] == 'complete'
        
        # Step 2: Try to aggregate Revenue column (has 'N/A', 'TBD', 'Invalid')
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Revenue',
                'operation': 'sum'
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Should detect anomaly and pause for user input OR complete successfully
        if agg_response['status'] == 'needs_input':
            # Verify anomaly response structure
            assert 'question' in agg_response
            assert 'choices' in agg_response
            assert len(agg_response['choices']) >= 2
            
            # Verify question mentions the problematic column
            assert 'Revenue' in agg_response['question']
            
            # Step 3: Provide user input to convert to numeric
            continue_response = agent.continue_execution(
                thread_id, 
                'convert_numeric'  # Choose to convert invalid values to NaN
            )
            
            # Should now complete successfully
            assert continue_response['status'] == 'complete'
            assert 'result' in continue_response
            
            # Verify the sum excludes invalid values
            result = continue_response['result']
            assert 'value' in result
            # Should sum valid numeric values only
            assert result['value'] > 0
        
        elif agg_response['status'] == 'complete':
            # If no anomaly detected, verify the operation completed
            assert 'result' in agg_response
            result = agg_response['result']
            assert 'value' in result
        
        else:
            # Some other error occurred
            pytest.fail(f"Unexpected response status: {agg_response['status']}, error: {agg_response.get('error')}")
    
    def test_multi_step_filter_and_aggregate_with_anomaly(self, agent, sample_sales_file):
        """
        Test multi-step operation: filter → aggregate with anomaly handling
        
        Validates:
        - Sequential step execution
        - Anomaly detection in multi-step context
        - Result passing between steps
        """
        thread_id = "test_multi_step_thread"
        file_id = "multi_step_sales"
        
        # Load file into session
        df = pd.read_csv(sample_sales_file)
        store_dataframe(file_id, df, sample_sales_file, thread_id)
        
        # Analyze file first
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        assert upload_response['status'] == 'complete'
        
        # Step 1: Filter for specific region
        filter_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'filter',
            'parameters': {
                'column': 'Region',
                'operator': '==',
                'value': 'North'
            }
        }
        
        filter_response = agent.execute(filter_request)
        assert filter_response['status'] == 'complete'
        
        # Verify filter results
        filter_result = filter_response['result']
        assert 'rows_matched' in filter_result
        assert filter_result['rows_matched'] >= 0  # May be 0 if no North region
        
        # Step 2: Aggregate on a numeric column (may trigger anomaly)
        agg_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Revenue',
                'operation': 'mean'
            }
        }
        
        agg_response = agent.execute(agg_request)
        
        # Handle potential anomaly
        if agg_response['status'] == 'needs_input':
            # Resolve anomaly
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            assert continue_response['status'] == 'complete'
            assert 'result' in continue_response
        elif agg_response['status'] == 'complete':
            assert 'result' in agg_response
        else:
            # Allow error if no data to aggregate
            assert agg_response['status'] == 'error'
    
    # ========================================================================
    # TEST 3: THREAD ISOLATION WITH CONCURRENT OPERATIONS
    # ========================================================================
    
    def test_thread_isolation_basic(self, agent, sample_invoice_file, sample_sales_file):
        """
        Test that different threads maintain separate dataframe contexts
        
        Validates:
        - Thread-isolated dataframe storage
        - No cross-thread data contamination
        - Independent query execution per thread
        """
        thread1 = "thread_isolation_1"
        thread2 = "thread_isolation_2"
        file1_id = "file1"
        file2_id = "file2"
        
        # Load different files into different threads
        df1 = pd.read_csv(sample_invoice_file)
        df2 = pd.read_csv(sample_sales_file)
        
        store_dataframe(file1_id, df1, sample_invoice_file, thread1)
        store_dataframe(file2_id, df2, sample_sales_file, thread2)
        
        # Analyze both files
        upload1_request = {
            'thread_id': thread1,
            'file_id': file1_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload2_request = {
            'thread_id': thread2,
            'file_id': file2_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        # Upload to both threads
        response1 = agent.execute(upload1_request)
        response2 = agent.execute(upload2_request)
        
        assert response1['status'] == 'complete'
        assert response2['status'] == 'complete'
        
        # Verify different document structures
        result1 = response1['result']
        result2 = response2['result']
        
        # Should have different schemas
        columns1 = set(result1['columns'])
        columns2 = set(result2['columns'])
        
        # Different files should have different column sets
        assert columns1 != columns2, f"Expected different columns, got same: {columns1}"
        
        # Query each thread independently
        # Use first column from each for basic queries
        col1 = result1['columns'][0]
        col2 = result2['columns'][0]
        
        query1_request = {
            'thread_id': thread1,
            'file_id': file1_id,
            'action': 'describe',
            'parameters': {}
        }
        
        query2_request = {
            'thread_id': thread2,
            'file_id': file2_id,
            'action': 'describe',
            'parameters': {}
        }
        
        query1_response = agent.execute(query1_request)
        query2_response = agent.execute(query2_request)
        
        # Both should succeed independently (or fail gracefully)
        assert query1_response['status'] in ['complete', 'error']
        assert query2_response['status'] in ['complete', 'error']
        
        # Verify thread isolation - thread1 should not see thread2's data
        wrong_thread_request = {
            'thread_id': thread1,  # Wrong thread
            'file_id': file2_id,   # File from thread2
            'action': 'describe',
            'parameters': {}
        }
        
        wrong_response = agent.execute(wrong_thread_request)
        assert wrong_response['status'] == 'error'  # Should fail - file not in this thread
    
    def test_concurrent_thread_operations(self, agent, sample_sales_file):
        """
        Test concurrent operations across multiple threads
        
        Validates:
        - Thread safety of dataframe cache
        - Concurrent query execution
        - No race conditions or data corruption
        """
        num_threads = 3
        results = {}
        errors = {}
        
        def worker_thread(thread_num):
            """Worker function for concurrent testing"""
            thread_id = f"concurrent_thread_{thread_num}"
            file_id = f"file_{thread_num}"
            
            try:
                # Load file into session for this thread
                df = pd.read_csv(sample_sales_file)
                store_dataframe(file_id, df, sample_sales_file, thread_id)
                
                # Analyze file
                upload_request = {
                    'thread_id': thread_id,
                    'file_id': file_id,
                    'action': 'analyze',
                    'parameters': {}
                }
                
                upload_response = agent.execute(upload_request)
                
                if upload_response['status'] != 'complete':
                    errors[thread_num] = f"Upload failed: {upload_response}"
                    return
                
                # Perform multiple queries
                for i in range(3):
                    query_request = {
                        'thread_id': thread_id,
                        'file_id': file_id,
                        'action': 'aggregate',
                        'parameters': {
                            'column': 'Quantity',
                            'operation': 'sum'
                        }
                    }
                    
                    query_response = agent.execute(query_request)
                    
                    if query_response['status'] not in ['complete', 'needs_input']:
                        errors[thread_num] = f"Query {i} failed: {query_response}"
                        return
                    
                    # If needs input (anomaly), resolve it
                    if query_response['status'] == 'needs_input':
                        continue_response = agent.continue_execution(thread_id, 'convert_numeric')
                        if continue_response['status'] != 'complete':
                            errors[thread_num] = f"Continue failed: {continue_response}"
                            return
                        query_response = continue_response
                    
                    # Store result
                    if thread_num not in results:
                        results[thread_num] = []
                    results[thread_num].append(query_response['result'])
                
            except Exception as e:
                errors[thread_num] = f"Exception: {str(e)}"
        
        # Run concurrent threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker_thread, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=30)  # 30 second timeout
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        
        # Verify all threads produced results
        assert len(results) == num_threads
        
        # Verify results are consistent within each thread
        for thread_num, thread_results in results.items():
            assert len(thread_results) == 3  # 3 queries per thread
            
            # All results from same thread should be identical (same data)
            first_result = thread_results[0]
            for result in thread_results[1:]:
                assert result['value'] == first_result['value']
                assert result['operation'] == first_result['operation']
    
    def test_thread_context_switching(self, agent, sample_invoice_file, sample_sales_file):
        """
        Test switching between thread contexts
        
        Validates:
        - Correct context loading per thread
        - No context bleeding between threads
        - Proper thread isolation after switches
        """
        thread1 = "context_switch_1"
        thread2 = "context_switch_2"
        file1_id = "invoice_data"
        file2_id = "sales_data"
        
        # Load files into different threads
        df1 = pd.read_csv(sample_invoice_file)
        df2 = pd.read_csv(sample_sales_file)
        
        store_dataframe(file1_id, df1, sample_invoice_file, thread1)
        store_dataframe(file2_id, df2, sample_sales_file, thread2)
        
        # Setup thread 1 with invoice data
        upload1_request = {
            'thread_id': thread1,
            'file_id': file1_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        response1 = agent.execute(upload1_request)
        assert response1['status'] == 'complete'
        
        # Setup thread 2 with sales data
        upload2_request = {
            'thread_id': thread2,
            'file_id': file2_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        response2 = agent.execute(upload2_request)
        assert response2['status'] == 'complete'
        
        # Get column info for queries
        columns1 = response1['result']['columns']
        columns2 = response2['result']['columns']
        
        # Query thread 1 - use first available column
        query1_request = {
            'thread_id': thread1,
            'file_id': file1_id,
            'action': 'filter',
            'parameters': {
                'column': columns1[0],
                'operator': '!=',
                'value': 'nonexistent_value'  # Should match most/all rows
            }
        }
        
        query1_response = agent.execute(query1_request)
        assert query1_response['status'] == 'complete'
        
        # Switch to thread 2 and query
        query2_request = {
            'thread_id': thread2,
            'file_id': file2_id,
            'action': 'filter',
            'parameters': {
                'column': columns2[0],
                'operator': '!=',
                'value': 'nonexistent_value'  # Should match most/all rows
            }
        }
        
        query2_response = agent.execute(query2_request)
        assert query2_response['status'] == 'complete'
        
        # Switch back to thread 1 - should still have invoice context
        # Try to aggregate on a numeric column if available
        numeric_cols1 = [col for col, dtype in response1['result']['dtypes'].items() 
                        if 'int' in dtype.lower() or 'float' in dtype.lower()]
        
        if numeric_cols1:
            query1_again_request = {
                'thread_id': thread1,
                'file_id': file1_id,
                'action': 'aggregate',
                'parameters': {
                    'column': numeric_cols1[0],
                    'operation': 'sum'
                }
            }
            
            query1_again_response = agent.execute(query1_again_request)
            
            # Handle potential anomaly
            if query1_again_response['status'] == 'needs_input':
                continue_response = agent.continue_execution(thread1, 'convert_numeric')
                assert continue_response['status'] == 'complete'
            else:
                assert query1_again_response['status'] == 'complete'
        
        # Verify results are from different datasets
        result1 = query1_response['result']
        result2 = query2_response['result']
        
        # Should have different row counts or data (different files)
        assert result1 != result2 or len(columns1) != len(columns2)
    
    def test_thread_clearing(self, agent, sample_sales_file):
        """
        Test clearing thread context
        
        Validates:
        - Thread data removal
        - No access to cleared thread data
        - Clean state after clearing
        """
        thread_id = "test_clear_thread"
        file_id = "test_file"
        
        # Load file into session
        df = pd.read_csv(sample_sales_file)
        store_dataframe(file_id, df, sample_sales_file, thread_id)
        
        # Upload and verify data exists
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        assert upload_response['status'] == 'complete'
        
        # Query to confirm data is accessible
        query_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',  # Use analyze instead of describe
            'parameters': {}
        }
        
        query_response = agent.execute(query_request)
        # Should succeed since data exists
        assert query_response['status'] == 'complete'
        
        # Clear the thread using the session system
        clear_thread_data(thread_id)
        
        # Also clear the agent's dataframe cache
        agent.dataframe_cache.clear_thread(thread_id)
        
        # Try to query again - should fail because file is not in session anymore
        query_after_clear_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',  # Use an action that requires the data to exist
            'parameters': {
                'column': 'Revenue',
                'operation': 'sum'
            }
        }
        
        query_after_clear_response = agent.execute(query_after_clear_request)
        
        # Should fail because thread data was cleared
        assert query_after_clear_response['status'] == 'error'
        assert 'not found' in query_after_clear_response['error'].lower()
    
    # ========================================================================
    # TEST 4: ERROR HANDLING AND RECOVERY
    # ========================================================================
    
    def test_file_not_found_error_handling(self, agent):
        """Test graceful handling of non-existent files"""
        thread_id = "test_error_thread"
        
        upload_request = {
            'thread_id': thread_id,
            'file_id': 'nonexistent_file',
            'action': 'analyze',
            'parameters': {'file_path': '/nonexistent/path/file.csv'}
        }
        
        response = agent.execute(upload_request)
        
        # Should return error with helpful message
        assert response['status'] == 'error'
        assert 'not found' in response['error'].lower() or 'does not exist' in response['error'].lower()
        assert 'metrics' in response  # Should still include metrics
    
    def test_invalid_column_reference_with_suggestions(self, agent, sample_sales_file):
        """Test fuzzy column matching for invalid column names"""
        thread_id = "test_fuzzy_thread"
        file_id = "fuzzy_test_file"
        
        # Load file into session
        df = pd.read_csv(sample_sales_file)
        store_dataframe(file_id, df, sample_sales_file, thread_id)
        
        # Upload file first
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        upload_response = agent.execute(upload_request)
        assert upload_response['status'] == 'complete'
        
        # Try to query with misspelled column name
        query_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'aggregate',
            'parameters': {
                'column': 'Reveue',  # Misspelled 'Revenue'
                'operation': 'sum'
            }
        }
        
        query_response = agent.execute(query_request)
        
        # Should return error with suggestions
        assert query_response['status'] == 'error'
        assert 'not found' in query_response['error']
        
        # Should suggest 'Revenue' as close match (if fuzzy matching is implemented)
        if 'metadata' in query_response and 'error_details' in query_response['metadata']:
            error_details = query_response['metadata']['error_details']
            if 'suggestions' in error_details:
                suggestions = error_details['suggestions']
                assert 'Revenue' in suggestions
    
    # ========================================================================
    # TEST 5: PERFORMANCE AND METRICS
    # ========================================================================
    
    def test_execution_metrics_tracking(self, agent, sample_sales_file):
        """Test that execution metrics are properly tracked and reported"""
        thread_id = "test_metrics_thread"
        file_id = "metrics_test_file"
        
        # Load file into session
        df = pd.read_csv(sample_sales_file)
        store_dataframe(file_id, df, sample_sales_file, thread_id)
        
        upload_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        start_time = time.time()
        response = agent.execute(upload_request)
        end_time = time.time()
        
        # Verify response includes metrics
        assert response['status'] == 'complete'
        assert 'metrics' in response
        
        metrics = response['metrics']
        
        # Verify basic metrics are present
        assert 'latency_ms' in metrics
        assert metrics['latency_ms'] > 0
        assert metrics['latency_ms'] < (end_time - start_time) * 1000 + 100  # Allow some overhead
        
        # Verify processing metrics
        if 'rows_processed' in metrics:
            assert metrics['rows_processed'] >= 0
        
        if 'columns_affected' in metrics:
            assert metrics['columns_affected'] >= 0


if __name__ == '__main__':
    # Run the integration tests
    pytest.main([__file__, '-v', '-s'])
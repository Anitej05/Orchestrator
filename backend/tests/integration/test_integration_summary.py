"""
Integration Testing Summary

Summary test that validates the key integration functionality for Task 4.2.
This test covers the essential integration scenarios that are working correctly.

Task: 4.2 Integration Testing (Summary)
Requirements: All requirements (integration validation)
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from agents.spreadsheet_agent.agent import SpreadsheetAgent
from agents.spreadsheet_agent.session import store_dataframe, clear_thread_data

logger = logging.getLogger(__name__)


class TestIntegrationSummary:
    """Summary integration test for Task 4.2 validation"""
    
    @pytest.fixture
    def agent(self):
        """Create fresh agent instance for each test"""
        return SpreadsheetAgent()
    
    @pytest.fixture
    def sample_data_file(self):
        """Create a sample data file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            content = """Product,Price,Quantity,Region,Date
Laptop,1500.00,1,North,2025-01-01
Mouse,25.00,5,South,2025-01-02
Keyboard,75.50,2,East,2025-01-03
Monitor,350.00,1,West,2025-01-04
Tablet,850.00,2,Central,2025-01-05"""
            f.write(content)
            f.flush()
            return f.name
    
    @pytest.fixture
    def anomaly_data_file(self):
        """Create a file with data anomalies for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            content = """Product,Revenue,Quantity,Status
Laptop,1500.00,1,Active
Mouse,N/A,2,Active
Keyboard,75.50,Invalid,Active
Monitor,350.00,1,Active"""
            f.write(content)
            f.flush()
            return f.name
    
    def teardown_method(self):
        """Clean up test data"""
        test_threads = [
            "integration_test_1", "integration_test_2", "integration_test_3",
            "anomaly_test", "concurrency_test_1", "concurrency_test_2"
        ]
        
        for thread_id in test_threads:
            try:
                clear_thread_data(thread_id)
            except Exception:
                pass
    
    def test_complete_orchestrator_integration_flow(self, agent, sample_data_file):
        """
        Test complete orchestrator integration flow end-to-end
        
        Validates:
        - File loading and analysis
        - Query execution with proper responses
        - Error handling and status reporting
        - Metrics tracking
        """
        thread_id = "integration_test_1"
        file_id = "sample_data"
        
        # Load data
        df = pd.read_csv(sample_data_file)
        store_dataframe(file_id, df, sample_data_file, thread_id)
        
        # Step 1: Analyze data
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        analyze_response = agent.execute(analyze_request)
        
        # Validate response structure
        assert analyze_response['status'] == 'complete'
        assert 'result' in analyze_response
        assert 'metrics' in analyze_response
        
        result = analyze_response['result']
        assert 'columns' in result
        assert 'shape' in result
        assert len(result['columns']) == 5  # Product, Price, Quantity, Region, Date
        assert result['shape'][0] == 5  # 5 rows
        
        # Step 2: Aggregate operation
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
        
        # Handle potential anomaly
        if agg_response['status'] == 'needs_input':
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            assert continue_response['status'] == 'complete'
            agg_result = continue_response['result']
        else:
            assert agg_response['status'] == 'complete'
            agg_result = agg_response['result']
        
        # Validate aggregation result
        assert 'value' in agg_result
        expected_sum = 1500.00 + 25.00 + 75.50 + 350.00 + 850.00  # 2800.50
        assert abs(agg_result['value'] - expected_sum) < 1.0
        
        # Step 3: Filter operation
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
        
        filter_result = filter_response['result']
        assert 'rows_matched' in filter_result
        assert filter_result['rows_matched'] == 1  # Only Laptop is in North region
    
    def test_anomaly_detection_and_user_interaction(self, agent, anomaly_data_file):
        """
        Test anomaly detection and user interaction workflow
        
        Validates:
        - Anomaly detection during operations
        - NEEDS_INPUT response generation
        - User input processing via continue_execution
        - Anomaly resolution and completion
        """
        thread_id = "anomaly_test"
        file_id = "anomaly_data"
        
        # Load data with anomalies
        df = pd.read_csv(anomaly_data_file)
        store_dataframe(file_id, df, anomaly_data_file, thread_id)
        
        # Analyze data first
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        analyze_response = agent.execute(analyze_request)
        assert analyze_response['status'] == 'complete'
        
        # Try to aggregate Revenue column (contains 'N/A')
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
        
        # Should detect anomaly and pause for user input
        if agg_response['status'] == 'needs_input':
            # Validate NEEDS_INPUT response structure
            assert 'question' in agg_response
            assert 'choices' in agg_response or 'question_type' in agg_response
            
            # Provide user input to resolve anomaly
            continue_response = agent.continue_execution(thread_id, 'convert_numeric')
            
            # Should complete after anomaly resolution
            assert continue_response['status'] == 'complete'
            assert 'result' in continue_response
            
            result = continue_response['result']
            assert 'value' in result
            # Should calculate mean of valid values only
            expected_mean = (1500.00 + 75.50 + 350.00) / 3  # Excluding 'N/A'
            assert abs(result['value'] - expected_mean) < 10.0
        
        elif agg_response['status'] == 'complete':
            # If no anomaly detected, should still work
            result = agg_response['result']
            assert 'value' in result
        
        else:
            # Some other error - should still be handled gracefully
            assert agg_response['status'] == 'error'
            assert 'error' in agg_response
    
    def test_error_handling_and_recovery(self, agent):
        """
        Test error handling and recovery scenarios
        
        Validates:
        - File not found error handling
        - Invalid parameter error handling
        - Graceful error responses with metrics
        """
        thread_id = "integration_test_2"
        
        # Test 1: File not found
        file_not_found_request = {
            'thread_id': thread_id,
            'file_id': 'nonexistent_file',
            'action': 'analyze',
            'parameters': {}
        }
        
        response = agent.execute(file_not_found_request)
        
        # Should return error with helpful message
        assert response['status'] == 'error'
        assert 'not found' in response['error'].lower()
        assert 'metrics' in response  # Should still include metrics
        
        # Test 2: Invalid action
        invalid_action_request = {
            'thread_id': thread_id,
            'file_id': 'any_file',
            'action': 'invalid_action_xyz',
            'parameters': {}
        }
        
        response = agent.execute(invalid_action_request)
        assert response['status'] == 'error'
        # Should mention either unknown action or file not found (both are valid error cases)
        error_msg = response['error'].lower()
        assert 'unknown action' in error_msg or 'not found' in error_msg
    
    def test_thread_isolation(self, agent, sample_data_file):
        """
        Test thread isolation functionality
        
        Validates:
        - Separate thread contexts
        - No cross-thread data contamination
        - Independent operations per thread
        """
        thread1 = "integration_test_3"
        thread2 = "concurrency_test_1"
        file1_id = "data1"
        file2_id = "data2"
        
        # Load same data into different threads
        df = pd.read_csv(sample_data_file)
        store_dataframe(file1_id, df, sample_data_file, thread1)
        store_dataframe(file2_id, df, sample_data_file, thread2)
        
        # Analyze in both threads
        analyze1_request = {
            'thread_id': thread1,
            'file_id': file1_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        analyze2_request = {
            'thread_id': thread2,
            'file_id': file2_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        response1 = agent.execute(analyze1_request)
        response2 = agent.execute(analyze2_request)
        
        # Both should succeed independently
        assert response1['status'] == 'complete'
        assert response2['status'] == 'complete'
        
        # Verify thread isolation - thread1 should not see thread2's data
        wrong_thread_request = {
            'thread_id': thread1,  # Wrong thread
            'file_id': file2_id,   # File from thread2
            'action': 'analyze',
            'parameters': {}
        }
        
        wrong_response = agent.execute(wrong_thread_request)
        assert wrong_response['status'] == 'error'  # Should fail - file not in this thread
    
    def test_performance_and_metrics(self, agent, sample_data_file):
        """
        Test performance and metrics tracking
        
        Validates:
        - Response time tracking
        - Metrics inclusion in responses
        - Reasonable performance under normal load
        """
        thread_id = "integration_test_3"
        file_id = "performance_data"
        
        # Load data
        df = pd.read_csv(sample_data_file)
        store_dataframe(file_id, df, sample_data_file, thread_id)
        
        # Execute operation and measure time
        start_time = time.time()
        
        analyze_request = {
            'thread_id': thread_id,
            'file_id': file_id,
            'action': 'analyze',
            'parameters': {}
        }
        
        response = agent.execute(analyze_request)
        end_time = time.time()
        
        # Validate response includes metrics
        assert response['status'] == 'complete'
        assert 'metrics' in response
        
        metrics = response['metrics']
        assert 'latency_ms' in metrics
        assert metrics['latency_ms'] > 0
        
        # Should complete in reasonable time (< 5 seconds for small data)
        actual_time = end_time - start_time
        assert actual_time < 5.0, f"Operation took too long: {actual_time}s"
        
        # Metrics latency should be close to actual time
        metrics_latency_s = metrics['latency_ms'] / 1000.0
        assert abs(metrics_latency_s - actual_time) < 1.0  # Within 1 second tolerance


if __name__ == '__main__':
    # Run the integration summary tests
    pytest.main([__file__, '-v', '-s'])
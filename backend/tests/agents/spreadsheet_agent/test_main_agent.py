"""
Tests for main spreadsheet agent (Task 10)

Tests endpoint handlers, error handling, and fuzzy column matching.
Requirements: 9.1, 9.3, 14.1, 14.2, 14.3
"""

import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import pandas as pd
from agents.spreadsheet_agent.agent import SpreadsheetAgent


class TestSpreadsheetAgent:
    """Test suite for SpreadsheetAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance"""
        return SpreadsheetAgent()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame"""
        return pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Salary': [50000, 60000, 70000]
        })
    
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly"""
        assert agent is not None
        assert agent.dialogue_manager is not None
        assert agent.dataframe_cache is not None
        assert agent.query_executor is not None
        assert agent.anomaly_detector is not None
    
    def test_error_response_format(self, agent):
        """Test that error responses have correct format"""
        import time
        start_time = time.time()
        
        response = agent._error_response(
            "Test error message",
            start_time,
            error_details={"test": "details"}
        )
        
        assert response['status'] == 'error'
        assert response['error'] == 'Test error message'
        assert 'metrics' in response
        assert response['metrics']['latency_ms'] >= 0
    
    def test_column_not_found_with_suggestions(self, agent, sample_df):
        """Test fuzzy column matching when column not found (Requirement 14.2)"""
        import time
        start_time = time.time()
        
        # Test with similar column name
        response = agent._handle_column_not_found('Naem', sample_df, start_time)
        
        assert response['status'] == 'error'
        assert 'Name' in response['error']  # Should suggest 'Name'
        assert 'metadata' in response
        assert 'error_details' in response['metadata']
        assert 'suggestions' in response['metadata']['error_details']
        assert 'Name' in response['metadata']['error_details']['suggestions']
    
    def test_column_not_found_no_suggestions(self, agent, sample_df):
        """Test column not found with no similar columns"""
        import time
        start_time = time.time()
        
        # Test with completely different column name
        response = agent._handle_column_not_found('XYZ123', sample_df, start_time)
        
        assert response['status'] == 'error'
        assert 'not found' in response['error']
        assert 'Available columns' in response['error']
    
    def test_pandas_error_handling(self, agent, sample_df):
        """Test pandas exception handling (Requirement 14.1)"""
        import time
        start_time = time.time()
        
        # Create a KeyError - this will be handled by _handle_column_not_found
        error = KeyError("'NonExistentColumn'")
        response = agent._handle_pandas_error(error, start_time, sample_df, "test operation")
        
        assert response['status'] == 'error'
        assert 'not found' in response['error']
        # KeyError triggers column not found handling with suggestions
        assert 'metadata' in response
        assert 'error_details' in response['metadata']
    
    def test_execute_missing_file_id(self, agent):
        """Test execute with missing file_id"""
        request = {
            'thread_id': 'test_thread',
            'action': 'analyze',
            'parameters': {}
        }
        
        response = agent.execute(request)
        
        assert response['status'] == 'error'
        assert 'file_id is required' in response['error']
    
    def test_execute_missing_action_and_prompt(self, agent):
        """Test execute with missing action and prompt"""
        # Store a test dataframe first so file_id check passes
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        agent.dataframe_cache.store('test_thread', 'test_file', test_df, {})
        
        request = {
            'thread_id': 'test_thread',
            'file_id': 'test_file',
            'parameters': {}
        }
        
        response = agent.execute(request)
        
        assert response['status'] == 'error'
        assert 'action' in response['error'] or 'prompt' in response['error']
    
    def test_execute_unknown_action(self, agent):
        """Test execute with unknown action"""
        # Store a test dataframe first
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        agent.dataframe_cache.store('test_thread', 'test_file', test_df, {})
        
        request = {
            'thread_id': 'test_thread',
            'file_id': 'test_file',
            'action': 'unknown_action',
            'parameters': {}
        }
        
        response = agent.execute(request)
        
        assert response['status'] == 'error'
        assert 'Unknown action' in response['error']
    
    def test_handle_aggregate_missing_column(self, agent):
        """Test aggregate with missing column parameter"""
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        import time
        start_time = time.time()
        
        response = agent._handle_aggregate(
            test_df, 'test_file', 'test_thread', {}, start_time
        )
        
        assert response['status'] == 'error'
        assert 'column' in response['error']
    
    def test_handle_aggregate_column_not_found(self, agent):
        """Test aggregate with non-existent column"""
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        import time
        start_time = time.time()
        
        response = agent._handle_aggregate(
            test_df, 'test_file', 'test_thread',
            {'column': 'NonExistent', 'operation': 'sum'},
            start_time
        )
        
        assert response['status'] == 'error'
        assert 'not found' in response['error']
    
    def test_handle_aggregate_success(self, agent):
        """Test successful aggregation"""
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        agent.dataframe_cache.store('test_thread', 'test_file', test_df, {})
        import time
        start_time = time.time()
        
        response = agent._handle_aggregate(
            test_df, 'test_file', 'test_thread',
            {'column': 'A', 'operation': 'sum'},
            start_time
        )
        
        assert response['status'] == 'complete'
        assert response['result']['value'] == 6.0
        assert response['result']['operation'] == 'sum'
        assert response['result']['column'] == 'A'
    
    def test_handle_filter_success(self, agent):
        """Test successful filter operation"""
        test_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        import time
        start_time = time.time()
        
        response = agent._handle_filter(
            test_df, 'test_file', 'test_thread',
            {'column': 'A', 'operator': '>', 'value': 3},
            start_time
        )
        
        assert response['status'] == 'complete'
        assert response['result']['rows_matched'] == 2
        assert len(response['result']['data']) == 2
    
    def test_handle_sort_success(self, agent):
        """Test successful sort operation"""
        test_df = pd.DataFrame({'A': [3, 1, 2]})
        import time
        start_time = time.time()
        
        response = agent._handle_sort(
            test_df, 'test_file', 'test_thread',
            {'columns': ['A'], 'ascending': True},
            start_time
        )
        
        assert response['status'] == 'complete'
        assert response['result']['data'][0]['A'] == 1
        assert response['result']['data'][1]['A'] == 2
        assert response['result']['data'][2]['A'] == 3
    
    def test_continue_execution_no_state(self, agent):
        """Test continue with no pending dialogue"""
        response = agent.continue_execution('nonexistent_thread', 'user_input')
        
        assert response['status'] == 'error'
        assert 'No pending dialogue' in response['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Unit tests for Query Executor

Tests basic query operations: filter, aggregate, sort, search, retrieve
"""

import pytest
import pandas as pd
import numpy as np
from backend.agents.spreadsheet_agent.query_executor import (
    QueryExecutor, QueryPlan, QueryResult
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Salary': [50000, 60000, 70000, 80000, 90000],
        'Department': ['Sales', 'Engineering', 'Sales', 'Engineering', 'Sales']
    })


@pytest.fixture
def executor():
    """Create a QueryExecutor instance"""
    return QueryExecutor()


class TestFilterOperation:
    """Test filter operations"""
    
    def test_filter_equal(self, executor, sample_df):
        """Test filter with == operator"""
        query = QueryPlan(
            operation='filter',
            conditions={
                'column': 'Department',
                'operator': '==',
                'value': 'Sales'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 3
        assert all(result.data['Department'] == 'Sales')
    
    def test_filter_greater_than(self, executor, sample_df):
        """Test filter with > operator"""
        query = QueryPlan(
            operation='filter',
            conditions={
                'column': 'Age',
                'operator': '>',
                'value': 30
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 3
        assert all(result.data['Age'] > 30)
    
    def test_filter_contains(self, executor, sample_df):
        """Test filter with contains operator"""
        query = QueryPlan(
            operation='filter',
            conditions={
                'column': 'City',
                'operator': 'contains',
                'value': 'New'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 1
        assert result.data.iloc[0]['City'] == 'New York'
    
    def test_filter_invalid_column(self, executor, sample_df):
        """Test filter with non-existent column"""
        query = QueryPlan(
            operation='filter',
            conditions={
                'column': 'InvalidColumn',
                'operator': '==',
                'value': 'test'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert not result.success
        assert 'not found' in result.explanation.lower()


class TestAggregateOperation:
    """Test aggregation operations"""
    
    def test_aggregate_sum(self, executor, sample_df):
        """Test sum aggregation"""
        query = QueryPlan(
            operation='aggregate',
            conditions={
                'function': 'sum',
                'column': 'Salary'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert result.data == 350000
        assert result.metadata['total_rows'] == 5
    
    def test_aggregate_mean(self, executor, sample_df):
        """Test mean aggregation"""
        query = QueryPlan(
            operation='aggregate',
            conditions={
                'function': 'mean',
                'column': 'Age'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert result.data == 35.0
    
    def test_aggregate_all_columns(self, executor, sample_df):
        """Test aggregation on all numeric columns"""
        query = QueryPlan(
            operation='aggregate',
            conditions={
                'function': 'sum'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert isinstance(result.data, dict)
        assert 'Age' in result.data
        assert 'Salary' in result.data
        assert result.data['Age'] == 175
        assert result.data['Salary'] == 350000


class TestSortOperation:
    """Test sort operations"""
    
    def test_sort_ascending(self, executor, sample_df):
        """Test sort in ascending order"""
        query = QueryPlan(
            operation='sort',
            conditions={
                'columns': ['Age'],
                'ascending': True
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert result.data.iloc[0]['Age'] == 25
        assert result.data.iloc[-1]['Age'] == 45
    
    def test_sort_descending(self, executor, sample_df):
        """Test sort in descending order"""
        query = QueryPlan(
            operation='sort',
            conditions={
                'columns': ['Salary'],
                'ascending': False
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert result.data.iloc[0]['Salary'] == 90000
        assert result.data.iloc[-1]['Salary'] == 50000
    
    def test_sort_multiple_columns(self, executor, sample_df):
        """Test sort by multiple columns"""
        query = QueryPlan(
            operation='sort',
            conditions={
                'columns': ['Department', 'Age'],
                'ascending': [True, False]
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 5


class TestSearchOperation:
    """Test search operations"""
    
    def test_search_single_column(self, executor, sample_df):
        """Test search in specific column"""
        query = QueryPlan(
            operation='search',
            conditions={
                'text': 'Alice',
                'columns': ['Name']
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 1
        assert result.data.iloc[0]['Name'] == 'Alice'
    
    def test_search_all_columns(self, executor, sample_df):
        """Test search across all text columns"""
        query = QueryPlan(
            operation='search',
            conditions={
                'text': 'Sales'
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 3


class TestRetrieveOperation:
    """Test retrieve operations"""
    
    def test_retrieve_by_indices(self, executor, sample_df):
        """Test retrieve specific rows by index"""
        query = QueryPlan(
            operation='retrieve',
            conditions={
                'indices': [0, 2, 4]
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert result.success
        assert len(result.data) == 3
        assert result.data.iloc[0]['Name'] == 'Alice'
        assert result.data.iloc[1]['Name'] == 'Charlie'
        assert result.data.iloc[2]['Name'] == 'Eve'
    
    def test_retrieve_invalid_index(self, executor, sample_df):
        """Test retrieve with invalid index"""
        query = QueryPlan(
            operation='retrieve',
            conditions={
                'indices': [100]
            }
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert not result.success
        assert 'invalid indices' in result.explanation.lower()


class TestErrorHandling:
    """Test error handling"""
    
    def test_unknown_operation(self, executor, sample_df):
        """Test handling of unknown operation"""
        query = QueryPlan(
            operation='unknown_op',
            conditions={}
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert not result.success
        assert 'unknown operation' in result.explanation.lower()
    
    def test_missing_parameters(self, executor, sample_df):
        """Test handling of missing parameters"""
        query = QueryPlan(
            operation='filter',
            conditions={}
        )
        
        result = executor.execute_query(sample_df, query)
        
        assert not result.success

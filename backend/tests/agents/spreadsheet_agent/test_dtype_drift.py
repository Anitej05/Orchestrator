"""
Test dtype drift detection and user interaction flow.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
from agents.spreadsheet_agent.models import QueryResult

query_agent = SpreadsheetQueryAgent()


class TestDtypeDriftDetection:
    """Test anomaly detection for dtype drift"""
    
    @pytest.mark.asyncio
    async def test_dtype_drift_numeric_to_object(self):
        """Detect when numeric column becomes object due to string values"""
        # Create DataFrame with intentional dtype drift
        # Original: all numeric
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5],
            'Sales': [100, 200, 'invalid', 400, 500],  # String in numeric column
            'Region': ['North', 'South', 'East', 'West', 'Central']
        })
        
        # Force Sales to object dtype (simulating real-world data issue)
        df['Sales'] = df['Sales'].astype('object')
        
        # Query that would normally work on numeric data
        result = await query_agent.query(
            df=df,
            question="What is the total Sales?",
            max_iterations=3
        )
        
        # Should detect anomaly instead of completing
        assert result.status == "anomaly_detected", f"Expected anomaly_detected, got {result.status}"
        assert result.needs_user_input == True
        assert result.anomaly is not None
        assert result.anomaly.anomaly_type == "dtype_drift"
        assert 'Sales' in result.anomaly.affected_columns
        assert result.user_choices is not None
        assert len(result.user_choices) == 4  # convert, ignore, treat_as_text, cancel
    
    @pytest.mark.asyncio
    async def test_dtype_drift_user_choices(self):
        """Verify user choice options are correct"""
        df = pd.DataFrame({
            'Price': [10.5, 20.3, 'N/A', 40.1],
            'Quantity': [1, 2, 3, 4]
        })
        df['Price'] = df['Price'].astype('object')
        
        result = await query_agent.query(
            df=df,
            question="Calculate average Price",
            max_iterations=2
        )
        
        assert result.needs_user_input == True
        assert result.user_choices is not None
        
        # Verify all expected choices exist
        choice_ids = [c.id for c in result.user_choices]
        assert 'convert_numeric' in choice_ids
        assert 'ignore_rows' in choice_ids
        assert 'treat_as_text' in choice_ids
        assert 'cancel' in choice_ids
        
        # Verify choice details
        convert_choice = next(c for c in result.user_choices if c.id == 'convert_numeric')
        assert 'NaN' in convert_choice.description
        assert convert_choice.is_safe == True
    
    @pytest.mark.asyncio
    async def test_no_dtype_drift_on_clean_data(self):
        """Clean numeric data should not trigger drift detection"""
        df = pd.DataFrame({
            'Value': [100, 200, 300, 400, 500],
            'Category': ['A', 'B', 'A', 'B', 'A']
        })
        
        result = await query_agent.query(
            df=df,
            question="What is the total Value?",
            max_iterations=2
        )
        
        # Should complete normally, no anomaly
        assert result.status == "completed"
        assert result.needs_user_input == False
        assert result.anomaly is None
        assert result.success == True
    
    @pytest.mark.asyncio
    async def test_dtype_drift_multiple_columns(self):
        """Detect drift in multiple columns"""
        df = pd.DataFrame({
            'Sales': [100, 'invalid', 300],
            'Cost': [50, 'N/A', 150],
            'Product': ['A', 'B', 'C']
        })
        df['Sales'] = df['Sales'].astype('object')
        df['Cost'] = df['Cost'].astype('object')
        
        result = await query_agent.query(
            df=df,
            question="Calculate profit (Sales - Cost)",
            max_iterations=2
        )
        
        assert result.status == "anomaly_detected"
        assert len(result.anomaly.affected_columns) >= 1  # At least one detected
        assert 'Sales' in result.anomaly.affected_columns or 'Cost' in result.anomaly.affected_columns
    
    @pytest.mark.asyncio
    async def test_dtype_drift_message_clarity(self):
        """Anomaly message should be clear and actionable"""
        df = pd.DataFrame({
            'Revenue': [1000, 2000, 'TBD', 4000],
            'Year': [2020, 2021, 2022, 2023]
        })
        df['Revenue'] = df['Revenue'].astype('object')
        
        result = await query_agent.query(
            df=df,
            question="Sum all Revenue",
            max_iterations=2
        )
        
        assert result.needs_user_input == True
        assert result.anomaly is not None
        
        # Message should mention the column and the issue
        message = result.anomaly.message.lower()
        assert 'revenue' in message
        assert 'string' in message or 'text' in message or 'drift' in message
        assert 'number' in message or 'numeric' in message
    
    @pytest.mark.asyncio
    async def test_dtype_drift_sample_values(self):
        """Anomaly should include sample problematic values"""
        df = pd.DataFrame({
            'Score': [95, 87, 'ABSENT', 76, 'N/A'],
            'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        })
        df['Score'] = df['Score'].astype('object')
        
        result = await query_agent.query(
            df=df,
            question="Average score",
            max_iterations=2
        )
        
        assert result.anomaly is not None
        assert result.anomaly.sample_values is not None
        assert 'Score' in result.anomaly.sample_values
        
        # Should contain the non-numeric values
        score_samples = result.anomaly.sample_values['Score']
        assert 'ABSENT' in score_samples or 'N/A' in score_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

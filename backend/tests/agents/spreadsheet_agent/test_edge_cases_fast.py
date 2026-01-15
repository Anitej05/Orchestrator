"""
Fast smoke tests for critical edge cases - optimized for speed.
Run these for quick validation before full test suite.

Usage: pytest test_edge_cases_fast.py -v --asyncio-mode=auto
"""

import asyncio
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
from tests.spreadsheet_agent.test_edge_case_helpers import (
    create_empty_dataframe,
    create_unicode_dataframe,
    create_edge_numerics_dataframe,
    create_multiindex_result_dataframe,
)

query_agent = SpreadsheetQueryAgent()


class TestCriticalEdgeCasesFast:
    """Fast smoke tests - max_iterations=2 for speed"""
    
    @pytest.mark.asyncio
    async def test_empty_dataframe_count(self):
        """Empty DataFrame should handle count query"""
        df = create_empty_dataframe()
        result = await query_agent.query(
            df=df,
            question="How many rows?",
            max_iterations=2
        )
        assert result.success, f"Failed: {result.error}"
    
    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        """Division by zero should not crash"""
        df = create_edge_numerics_dataframe()
        result = await query_agent.query(
            df=df,
            question="Calculate Value divided by Denominator",
            max_iterations=2
        )
        assert result.success, f"Failed: {result.error}"
    
    @pytest.mark.asyncio
    async def test_unicode_data(self):
        """Unicode characters should work"""
        df = create_unicode_dataframe()
        result = await query_agent.query(
            df=df,
            question="How many products?",
            max_iterations=2
        )
        assert result.success, f"Failed: {result.error}"
    
    @pytest.mark.asyncio
    async def test_multiindex_groupby(self):
        """Multi-column groupby should work (the bug we fixed)"""
        df = create_multiindex_result_dataframe()
        result = await query_agent.query(
            df=df,
            question="Total Sales by Region and Product",
            max_iterations=2
        )
        assert result.success, f"Failed: {result.error}"
    
    @pytest.mark.asyncio
    async def test_infinity_values(self):
        """Infinity values should be handled"""
        df = create_edge_numerics_dataframe()
        result = await query_agent.query(
            df=df,
            question="What is the total Value?",
            max_iterations=2
        )
        assert result.success, f"Failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])

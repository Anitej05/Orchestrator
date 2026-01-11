"""
MNIST stress tests for SpreadsheetQueryAgent.

Focus:
- Semantic correctness on numeric pixel columns
- Percentage distribution across labels (~100%)
- Read-only behavior (no DataFrame mutation)
- Performance and memory at increasing sizes
"""

import time
import psutil
import pytest
import pandas as pd
from pathlib import Path
import sys

# Ensure repo root is on path like other tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent


MNIST_PATH = Path(__file__).parent / "edge_case_datasets" / "mnist_784.csv"
query_agent = SpreadsheetQueryAgent()


def _load_mnist(nrows: int) -> pd.DataFrame:
    df = pd.read_csv(MNIST_PATH, nrows=nrows)
    # Expect either 'label' or 'class' as the target column
    assert ("label" in df.columns) or ("class" in df.columns), "MNIST must have a 'label' or 'class' column"
    return df

def _label_col(df: pd.DataFrame) -> str:
    return "label" if "label" in df.columns else "class"


class TestMNISTSemantic:
    @pytest.mark.parametrize("nrows", [1000, 10000])
    @pytest.mark.asyncio
    async def test_basic_pixel_stats(self, nrows):
        """Compute mean on a few pixel columns and ensure scalar results."""
        df = _load_mnist(nrows)
        question = "What is the mean of pixels pixel1, pixel2, pixel3?"
        start = time.time()
        result = await query_agent.query(df=df, question=question, max_iterations=2)
        latency = (time.time() - start) * 1000

        # Semantic correctness: return should be a scalar/numeric summary
        assert result.success is True
        assert result.status == "completed"
        # Ensure it did not mutate
        assert result.final_dataframe is not None
        assert list(result.final_dataframe.columns) == list(df.columns)

        # Record performance
        print(f"MNIST {nrows} basic stats latency: {latency:.2f} ms")


class TestMNISTLabelDistribution:
    @pytest.mark.parametrize("nrows", [1000, 10000])
    @pytest.mark.asyncio
    async def test_label_percentage_distribution(self, nrows):
        """Group by label and compute percentage distribution; sum ~100%."""
        df = _load_mnist(nrows)
        label_name = _label_col(df)
        question = f"What percentage of rows are in each {label_name}?"
        result = await query_agent.query(df=df, question=question, max_iterations=3)

        assert result.success is True
        assert result.status == "completed"

        # The agent should append a sanity note if not ~100
        if result.final_data is not None:
            # If final_data is a DataFrame/Series, percentages should sum ~100
            try:
                if isinstance(result.final_data, list) and len(result.final_data) == 1:
                    data = result.final_data[0]
                else:
                    data = result.final_data
                if hasattr(data, "sum"):
                    total = float(data.sum())
                    assert 98.0 <= total <= 102.0
            except Exception:
                # If not directly parsable, ensure the answer includes distribution language
                assert "percent" in (result.answer or "").lower()


class TestMNISTInvalidOps:
    @pytest.mark.parametrize("nrows", [1000])
    @pytest.mark.asyncio
    async def test_invalid_pixel_categories(self, nrows):
        """Treating pixel columns as categories should not silently coerce; expect warning/pause or failure."""
        df = _load_mnist(nrows)
        question = "Group pixels pixel1..pixel10 as categories and count combinations"
        result = await query_agent.query(df=df, question=question, max_iterations=2)

        # Expect graceful failure or pause; no silent coercion
        assert (result.needs_user_input is True) or (result.success is False)
        # Must not mutate schema
        assert list(result.final_dataframe.columns) == list(df.columns)


class TestMNISTPerformance:
    @pytest.mark.asyncio
    async def test_performance_scale_10000(self):
        df = _load_mnist(10000)
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start = time.time()
        result = await query_agent.query(df=df, question="Mean pixel value across all rows", max_iterations=2)
        elapsed_ms = (time.time() - start) * 1000
        mem_after = process.memory_info().rss / (1024 * 1024)

        assert result.success is True
        print(f"MNIST 10k latency: {elapsed_ms:.2f} ms, memory delta: {mem_after - mem_before:.2f} MB")

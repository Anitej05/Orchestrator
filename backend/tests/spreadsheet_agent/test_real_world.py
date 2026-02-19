"""
Spreadsheet Agent — Real-World Task Tests

Sends natural-language prompts end-to-end, progressively from simple to complex.
These tests evaluate the agent's LLM planning + execution against realistic scenarios.

Run:
  pytest backend/tests/spreadsheet_agent/test_real_world.py -v -s
"""

import sys
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.tests.conftest import start_agent

PORT = 9000
AGENT_ID = "spreadsheet_agent"
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
CSV_PATH = str(TEST_DATA_DIR / "sales_data.csv")
XLSX_PATH = str(TEST_DATA_DIR / "employees.xlsx")


@pytest.fixture(scope="module")
def spreadsheet_server():
    yield from start_agent(AGENT_ID)


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(timeout=120.0) as c:
        yield c


async def _nl(client, port, prompt, thread):
    """Send a natural-language prompt and return the parsed response."""
    resp = await client.post(
        f"http://localhost:{port}/execute",
        json={"prompt": prompt, "thread_id": thread},
    )
    resp.raise_for_status()
    return resp.json()


def _assert_success(result: Dict[str, Any], task_desc: str):
    """Standard success assertion with helpful failure message."""
    assert result.get("success") is True, (
        f"❌ Task FAILED: '{task_desc}'\n"
        f"   error={result.get('error')}\n"
        f"   message={result.get('message')}\n"
        f"   full response={result}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE TASKS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_simple_load_and_total_revenue(spreadsheet_server, client):
    """
    SIMPLE: Load a CSV from a path and compute total Revenue.
    Validates: file loading + basic numeric aggregation via NL prompt.
    """
    port = spreadsheet_server
    thread = "rw_simple_1"
    prompt = (
        f"Load the file at {CSV_PATH} and tell me the total revenue across all rows."
    )
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Simple 1] {prompt[:60]}…")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Load CSV + total revenue")
    # The answer should contain a number
    answer = str(result.get("data", {}).get("answer", result.get("message", "")))
    assert any(c.isdigit() for c in answer), f"Expected numeric answer, got: {answer}"
    print(f"   ✅ Answer: {answer[:150]}")


@pytest.mark.asyncio
async def test_simple_top_product(spreadsheet_server, client):
    """
    SIMPLE: Which product had the highest units sold?
    Validates: grouping by product + max aggregation.
    """
    port = spreadsheet_server
    thread = "rw_simple_2"
    # Load file first
    await client.post(
        f"http://localhost:{port}/execute",
        json={
            "prompt": f"Load file at {CSV_PATH}",
            "thread_id": thread,
            "action": "load_file",
            "payload": {"file_path": CSV_PATH, "thread_id": thread, "file_id": "sales_data"},
        },
    )
    prompt = "Which product had the highest total units sold?"
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Simple 2] {prompt}")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Top product by units")
    answer = str(result.get("data", {}).get("answer", ""))
    assert answer, "Expected non-empty answer"
    print(f"   ✅ Answer: {answer[:150]}")


# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM TASKS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_medium_filter_sort(spreadsheet_server, client):
    """
    MEDIUM: Filter rows where Region is 'North', then sort by Revenue descending.
    Validates: multi-step NL task (filter + sort chained).
    """
    port = spreadsheet_server
    thread = "rw_medium_1"
    await client.post(
        f"http://localhost:{port}/execute",
        json={
            "action": "load_file",
            "payload": {"file_path": CSV_PATH, "thread_id": thread, "file_id": "sales_data"},
            "thread_id": thread,
            "prompt": "",
        },
    )
    prompt = (
        "Filter the data to show only rows where Region is 'North', "
        "then sort the results by Revenue from highest to lowest."
    )
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Medium 1] {prompt}")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Filter North + sort Revenue DESC")
    print(f"   ✅ shape={result.get('data', {}).get('shape')}")


@pytest.mark.asyncio
async def test_medium_groupby_average(spreadsheet_server, client):
    """
    MEDIUM: Group by Product, calculate average Revenue per product.
    Validates: groupby aggregation via pure NL.
    """
    port = spreadsheet_server
    thread = "rw_medium_2"
    await client.post(
        f"http://localhost:{port}/execute",
        json={
            "action": "load_file",
            "payload": {"file_path": CSV_PATH, "thread_id": thread, "file_id": "sales_data"},
            "thread_id": thread,
            "prompt": "",
        },
    )
    prompt = "Group the data by Product and show the average Revenue for each product."
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Medium 2] {prompt}")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Group by Product, avg Revenue")
    print(f"   ✅ data={str(result.get('data', {}))[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPLEX TASKS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_complex_multi_step_pipeline(spreadsheet_server, client):
    """
    COMPLEX: Load employees.xlsx, fill missing department names with 'Unknown',
             merge Employee sheet with Departments sheet, export result as CSV.
    Validates: multi-step planning involving load → fill_missing → merge → export.
    """
    port = spreadsheet_server
    thread = "rw_complex_1"
    prompt = (
        f"Load the Excel file at {XLSX_PATH}. "
        "Fill any missing DepartmentID values with 0. "
        "Then give me a count of employees per department."
    )
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Complex 1] Multi-step: fill missing + aggregate")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Fill missing + count by dept")
    print(f"   ✅ data keys={list(result.get('data', {}).keys())}")


@pytest.mark.asyncio
async def test_complex_add_column_filter_aggregate(spreadsheet_server, client):
    """
    COMPLEX: Load sales CSV, add a Profit column (20% of Revenue),
             filter out rows where Profit < 100, then aggregate by Region.
    Validates: add_column → filter → aggregate chained through planning.
    """
    port = spreadsheet_server
    thread = "rw_complex_2"
    prompt = (
        f"Load the CSV at {CSV_PATH}. "
        "Add a 'Profit' column as 20% of Revenue. "
        "Remove rows where Profit is less than 100. "
        "Then group by Region and show me the total Profit per region."
    )
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Complex 2] Add Profit col → filter → agg by Region")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Add col + filter + groupby pipeline")
    print(f"   ✅ data={str(result.get('data', {}))[:300]}")


@pytest.mark.asyncio
async def test_complex_export_pipeline(spreadsheet_server, client):
    """
    COMPLEX: Transform data (Revenue to thousands), sort desc, export to XLSX.
    Validates: transform → sort → export end-to-end.
    """
    port = spreadsheet_server
    thread = "rw_complex_3"
    # Load first
    await client.post(
        f"http://localhost:{port}/execute",
        json={
            "action": "load_file",
            "payload": {"file_path": CSV_PATH, "thread_id": thread, "file_id": "sales_data"},
            "thread_id": thread,
            "prompt": "",
        },
    )
    prompt = (
        "Convert the Revenue column to thousands (divide by 1000). "
        "Sort by Revenue descending. "
        "Export the result as an Excel file named 'revenue_report'."
    )
    result = await _nl(client, port, prompt, thread)
    print(f"\n🌍 [Complex 3] Transform → sort → export XLSX")
    print(f"   → {result.get('message')}")
    _assert_success(result, "Transform + sort + export")
    print(f"   ✅ data={str(result.get('data', {}))[:300]}")

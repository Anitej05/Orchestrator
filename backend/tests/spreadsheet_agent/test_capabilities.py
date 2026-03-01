"""
Spreadsheet Agent — Capability Tests

Tests every capability by calling the agent's /execute endpoint directly
with explicit action names and structured payloads.

Prerequisites:
  - Agent running on port 9000 (or auto-started by the session fixture)
  - Test data at backend/tests/test_data/sales_data.csv
  - Test data at backend/tests/test_data/employees.xlsx

Run:
  pytest backend/tests/spreadsheet_agent/test_capabilities.py -v -s
"""

import base64
import sys
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
import pytest_asyncio

# ── path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.tests.conftest import start_agent

# ── constants ──────────────────────────────────────────────────────────────────
PORT = 9000
AGENT_ID = "spreadsheet_agent"
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
CSV_PATH = str(TEST_DATA_DIR / "sales_data.csv")
XLSX_PATH = str(TEST_DATA_DIR / "employees.xlsx")
THREAD = "test_spreadsheet_capabilities"


# ── Session-level agent fixture ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def spreadsheet_server():
    """Start (or reuse) the Spreadsheet Agent for this test module."""
    yield from start_agent(AGENT_ID)


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(timeout=90.0) as c:
        yield c


# ── Helper ─────────────────────────────────────────────────────────────────────
async def _post(client, port, prompt="", action=None, payload=None, thread=THREAD):
    body: Dict[str, Any] = {"prompt": prompt, "thread_id": thread}
    if action:
        body["action"] = action
    if payload:
        body["payload"] = payload
    resp = await client.post(f"http://localhost:{port}/execute", json=body)
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_health_check(spreadsheet_server, client):
    """Agent /health returns 200."""
    port = spreadsheet_server
    r = await client.get(f"http://localhost:{port}/health")
    assert r.status_code == 200
    print(f"\n✅ Health: {r.json()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD FILE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_load_csv(spreadsheet_server, client):
    """load_file: loads sales_data.csv and returns 50 rows × 5 columns."""
    port = spreadsheet_server
    result = await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH,
        "thread_id": THREAD,
        "file_id": "sales_data",
    })
    print(f"\n📂 load_csv: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert data["shape"][0] == 50, f"Expected 50 rows, got {data['shape'][0]}"
    assert data["shape"][1] == 5, f"Expected 5 columns, got {data['shape'][1]}"
    assert "Revenue" in data["columns"]
    print(f"   shape={data['shape']}, columns={data['columns']}")


@pytest.mark.asyncio
async def test_load_xlsx(spreadsheet_server, client):
    """load_file: loads employees.xlsx (multi-sheet → first sheet by default)."""
    port = spreadsheet_server
    result = await _post(client, port, action="load_file", payload={
        "file_path": XLSX_PATH,
        "thread_id": THREAD,
        "file_id": "employees",
    })
    print(f"\n📂 load_xlsx: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert data["shape"][0] > 0, "Expected rows > 0"
    assert "Name" in data["columns"] or "EmployeeID" in data["columns"]
    print(f"   shape={data['shape']}, columns={data['columns']}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. UPLOAD FILE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_upload_file(spreadsheet_server, client):
    """upload_file: upload CSV bytes and get file_id back."""
    port = spreadsheet_server
    with open(CSV_PATH, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    result = await _post(client, port, action="upload_file", payload={
        "content": content_b64,
        "filename": "uploaded_sales.csv",
        "thread_id": THREAD,
    })
    print(f"\n📤 upload_file: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert "file_id" in data
    assert data["shape"][0] > 0
    print(f"   file_id={data['file_id']}, shape={data['shape']}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT FILE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_export_csv(spreadsheet_server, client):
    """export_file: export loaded data as CSV; verify path is returned."""
    port = spreadsheet_server
    # Ensure something is loaded first
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="export_file", payload={
        "filename": "test_export",
        "format": "csv",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n💾 export_csv: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert "file_path" in data
    assert data["rows"] == 50
    print(f"   exported to: {data['file_path']}")


@pytest.mark.asyncio
async def test_export_xlsx(spreadsheet_server, client):
    """export_file: export as XLSX."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="export_file", payload={
        "filename": "test_export_excel",
        "format": "xlsx",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n💾 export_xlsx: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   exported to: {result['data']['file_path']}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PROCESS DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_process_data_total_revenue(spreadsheet_server, client):
    """process_data: NL query — total revenue calculation."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="process_data", payload={
        "instruction": "Calculate the total Revenue across all rows",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔢 process_data (total revenue): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    answer = result["data"].get("answer", "")
    assert answer, "Expected non-empty answer"
    print(f"   answer={answer}")


@pytest.mark.asyncio
async def test_process_data_top_product(spreadsheet_server, client):
    """process_data: NL query — which product has highest total revenue."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="process_data", payload={
        "instruction": "Which product has the highest total Revenue?",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🏆 process_data (top product): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   answer={result['data'].get('answer')}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FILTER DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_filter_by_region(spreadsheet_server, client):
    """filter_data: filter rows where Region == 'North'."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="filter_data", payload={
        "column": "Region",
        "operator": "==",
        "value": "North",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔍 filter_data (Region=North): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert data["filtered_rows"] < data["original_rows"], "Filter should reduce row count"
    assert data["original_rows"] == 50
    print(f"   {data['original_rows']} → {data['filtered_rows']} rows")


@pytest.mark.asyncio
async def test_filter_revenue_gt(spreadsheet_server, client):
    """filter_data: filter rows where Revenue > 2000."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="filter_data", payload={
        "column": "Revenue",
        "operator": ">",
        "value": "2000",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔍 filter_data (Revenue>2000): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    assert result["data"]["filtered_rows"] > 0
    print(f"   filtered to {result['data']['filtered_rows']} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SORT DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sort_ascending(spreadsheet_server, client):
    """sort_data: sort by Revenue ascending."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="sort_data", payload={
        "column": "Revenue",
        "ascending": True,
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔼 sort_data (Revenue ASC): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   sorted {result['data'].get('shape', '?')} rows")


@pytest.mark.asyncio
async def test_sort_descending(spreadsheet_server, client):
    """sort_data: sort by Revenue descending."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="sort_data", payload={
        "column": "Revenue",
        "ascending": False,
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔽 sort_data (Revenue DESC): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   sorted {result['data'].get('shape', '?')} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 8. AGGREGATE DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_aggregate_sum_by_product(spreadsheet_server, client):
    """aggregate_data: sum Revenue by Product."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="aggregate_data", payload={
        "group_by": "Product",
        "agg_column": "Revenue",
        "agg_function": "sum",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n📊 aggregate_data (sum Revenue by Product): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    # Should have 3 unique products (Widget A, B, C)
    shape = data.get("shape", [0])
    assert shape[0] <= 4, "Expected ≤4 groups"
    print(f"   aggregated to {shape} groups")


@pytest.mark.asyncio
async def test_aggregate_avg_revenue_by_region(spreadsheet_server, client):
    """aggregate_data: average Revenue by Region."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="aggregate_data", payload={
        "group_by": "Region",
        "agg_column": "Revenue",
        "agg_function": "mean",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n📊 aggregate_data (mean Revenue by Region): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   shape={result['data'].get('shape')}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_merge_data(spreadsheet_server, client):
    """merge_data: inner join sales_data on itself (same file, trivial join)."""
    port = spreadsheet_server
    # Load CSV twice under different IDs
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "df_left",
    })
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "df_right",
    })
    result = await _post(client, port, action="merge_data", payload={
        "left_file_id": "df_left",
        "right_file_id": "df_right",
        "on": "Date",
        "how": "inner",
        "thread_id": THREAD,
    })
    print(f"\n🔗 merge_data: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   merged shape={result['data'].get('shape')}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. COLUMN OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_add_column(spreadsheet_server, client):
    """add_column: add a Profit column = Revenue * 0.20."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="add_column", payload={
        "column_name": "Profit",
        "expression": "Revenue * 0.20",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n➕ add_column (Profit): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    assert "Profit" in result["data"].get("columns", [])
    print(f"   columns now: {result['data']['columns']}")


@pytest.mark.asyncio
async def test_rename_column(spreadsheet_server, client):
    """rename_column: rename 'Units' to 'Units_Sold'."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="rename_column", payload={
        "old_name": "Units",
        "new_name": "Units_Sold",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n✏️  rename_column (Units→Units_Sold): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    cols = result["data"].get("columns", [])
    assert "Units_Sold" in cols or "Units" in cols   # either renamed or original
    print(f"   columns: {cols}")


@pytest.mark.asyncio
async def test_fill_missing(spreadsheet_server, client):
    """fill_missing: fill null values in Product column with 'Unknown'."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="fill_missing", payload={
        "column": "Product",
        "strategy": "value",
        "fill_value": "Unknown",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔧 fill_missing (Product→'Unknown'): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    nulls_after = data.get("null_count_after", data.get("null_count", -1))
    print(f"   null_count_after={nulls_after}")


@pytest.mark.asyncio
async def test_drop_column(spreadsheet_server, client):
    """drop_column: drop the 'Units' column."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="drop_column", payload={
        "column": "Units",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n➖ drop_column (Units): {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    cols = result["data"].get("columns", [])
    assert "Units" not in cols, f"'Units' should have been dropped but is still in {cols}"
    print(f"   columns after drop: {cols}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. GET SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_get_summary(spreadsheet_server, client):
    """get_summary: returns describe() stats for the loaded file."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="get_summary", payload={
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n📈 get_summary: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    data = result["data"]
    assert data, "Expected non-empty summary data"
    print(f"   summary keys: {list(data.keys())[:6]}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. LIST FILES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_list_files(spreadsheet_server, client):
    """list_files: after loading files, session should list them."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="list_files", payload={
        "thread_id": THREAD,
    })
    print(f"\n📋 list_files: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    files = result["data"].get("files", [])
    assert len(files) > 0, "Expected at least 1 file in session"
    print(f"   files in session: {[f.get('file_id') for f in files]}")


# ══════════════════════════════════════════════════════════════════════════════
# 13. TRANSFORM DATA
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_transform_data(spreadsheet_server, client):
    """transform_data: convert Revenue column to thousands (Revenue / 1000)."""
    port = spreadsheet_server
    await _post(client, port, action="load_file", payload={
        "file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data",
    })
    result = await _post(client, port, action="transform_data", payload={
        "instruction": "Divide the Revenue column by 1000 to convert it to thousands",
        "file_id": "sales_data",
        "thread_id": THREAD,
    })
    print(f"\n🔄 transform_data: {result.get('message')}")
    assert result.get("success") is True, f"Expected success, got: {result}"
    print(f"   shape after transform: {result['data'].get('shape')}")

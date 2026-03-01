import httpx
import json
import sys

BASE = "http://localhost:9000"
TEST_DATA = r"D:\Internship\Orbimesh\backend\tests\test_data"
CSV_PATH = TEST_DATA + r"\sales_data.csv"
XLSX_PATH = TEST_DATA + r"\employees.xlsx"
THREAD = "manual_test"

def post(action=None, payload=None, prompt=""):
    body = {"prompt": prompt, "thread_id": THREAD}
    if action:
        body["action"] = action
    if payload:
        body["payload"] = payload
    r = httpx.post(f"{BASE}/execute", json=body, timeout=90)
    r.raise_for_status()
    return r.json()

def get(path):
    r = httpx.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()

def show(label, result):
    status = "[PASS]" if result.get("success") or result.get("status") == "success" else "[FAIL]"
    print(f"\n{status} [{label}]")

    print(f"  message : {result.get('message') or result.get('summary') or result.get('result','')}")
    if result.get("error_message"):
        print(f"  ERROR   : {result['error_message']}")
    if result.get("error"):
        print(f"  ERROR   : {result['error']}")
    if result.get("data"):
        d = result["data"]
        print(f"  data    : {json.dumps(d, default=str)[:300]}")
    if status.startswith("[FAIL]"):
        print(f"  FULL    : {json.dumps(result, default=str)[:500]}")


# ─────────────────────────────────────────────
test = sys.argv[1] if len(sys.argv) > 1 else "health"

if test == "health":
    r = get("/health")
    print("✅ /health:", json.dumps(r, indent=2))

elif test == "capabilities":
    r = get("/capabilities")
    caps = r.get("capabilities", [])
    print(f"✅ {len(caps)} capabilities registered:")
    for c in caps:
        print(f"   • {c['name']}: {c['description'][:70]}")

elif test == "load_csv":
    r = post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    show("load_file CSV", r)

elif test == "load_xlsx":
    r = post("load_file", {"file_path": XLSX_PATH, "thread_id": THREAD, "file_id": "employees"})
    show("load_file XLSX", r)

elif test == "upload_file":
    import base64
    with open(CSV_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = post("upload_file", {"content": b64, "filename": "uploaded_sales.csv", "thread_id": THREAD})
    show("upload_file", r)

elif test == "export_csv":
    # load first
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("export_file", {"filename": "test_export", "format": "csv", "file_id": "sales_data", "thread_id": THREAD})
    show("export_file CSV", r)

elif test == "export_xlsx":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("export_file", {"filename": "test_export_excel", "format": "xlsx", "file_id": "sales_data", "thread_id": THREAD})
    show("export_file XLSX", r)

elif test == "process_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("process_data", {"instruction": "Calculate the total Revenue across all rows", "file_id": "sales_data", "thread_id": THREAD})
    show("process_data", r)

elif test == "filter_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("filter_data", {"column": "Region", "operator": "==", "value": "North", "file_id": "sales_data", "thread_id": THREAD})
    show("filter_data (Region=North)", r)

elif test == "sort_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("sort_data", {"columns": ["Revenue"], "ascending": False, "file_id": "sales_data", "thread_id": THREAD})
    show("sort_data (Revenue DESC)", r)

elif test == "aggregate_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("aggregate_data", {"group_by": "Product", "agg_column": "Revenue", "agg_function": "sum", "file_id": "sales_data", "thread_id": THREAD})
    show("aggregate_data", r)

elif test == "merge_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "df_left"})
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "df_right"})
    r = post("merge_data", {"file_ids": ["df_left", "df_right"], "on": "Date", "how": "inner", "thread_id": THREAD})
    show("merge_data", r)

elif test == "add_column":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("add_column", {"name": "Profit", "expression": "df['Revenue'] * 0.20", "file_id": "sales_data", "thread_id": THREAD})
    show("add_column (Profit)", r)

elif test == "rename_column":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("rename_column", {"old_name": "Units", "new_name": "Units_Sold", "file_id": "sales_data", "thread_id": THREAD})
    show("rename_column", r)

elif test == "fill_missing":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("fill_missing", {"column": "Revenue", "method": "mean", "file_id": "sales_data", "thread_id": THREAD})
    show("fill_missing (Product)", r)

elif test == "drop_column":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("drop_column", {"column": "Units", "file_id": "sales_data", "thread_id": THREAD})
    show("drop_column (Units)", r)

elif test == "get_summary":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("get_summary", {"file_id": "sales_data", "thread_id": THREAD})
    show("get_summary", r)

elif test == "transform_data":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("transform_data", {"instruction": "Divide the Revenue column by 1000", "file_id": "sales_data", "thread_id": THREAD})
    show("transform_data", r)

elif test == "list_files":
    post("load_file", {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"})
    r = post("list_files", {"thread_id": THREAD})
    show("list_files", r)

elif test == "nl_simple1":
    r = post(prompt=f"Load the file at {CSV_PATH} and tell me the total revenue across all rows.")
    show("NL: total revenue", r)

elif test == "nl_simple2":
    post("load_file", {"file_path": CSV_PATH, "file_id": "sales_data", "thread_id": THREAD})
    r = post(prompt="Which product had the highest total units sold?")
    show("NL: top product by units", r)

elif test == "nl_medium1":
    post("load_file", {"file_path": CSV_PATH, "file_id": "sales_data", "thread_id": THREAD})
    r = post(prompt="Filter only rows where Region is North, then sort by Revenue from highest to lowest")
    show("NL: filter+sort", r)

elif test == "nl_medium2":
    post("load_file", {"file_path": CSV_PATH, "file_id": "sales_data", "thread_id": THREAD})
    r = post(prompt="Group by Product and show me the average Revenue per product")
    show("NL: groupby avg Revenue", r)

elif test == "nl_complex1":
    r = post(prompt=f"Load {XLSX_PATH}, fill missing DepartmentID values with 0, then count employees per department.")
    show("NL: complex XLSX pipeline", r)

elif test == "nl_complex2":
    post("load_file", {"file_path": CSV_PATH, "file_id": "sales_data", "thread_id": THREAD})
    r = post(prompt="Add a Profit column as 20% of Revenue, remove rows where Profit is less than 100, then group by Region and show total Profit per region")
    show("NL: add col + filter + groupby", r)

# ─── STRESS TESTS ───────────────────────────────
elif test == "stress1":
    # 5 steps: load → add_column → filter → sort → export
    r = post(prompt=f"Load the CSV file at {CSV_PATH}, add a Profit column calculated as Revenue times 0.15, filter rows where Profit is greater than 200, sort by Profit from highest to lowest, then export the result as profit_report.xlsx")
    show("STRESS1: profit pipeline (5 steps)", r)

elif test == "stress2":
    # 5 steps: load → fill_missing → rename → drop → get_summary
    r = post(prompt=f"Load {XLSX_PATH}, fill missing DepartmentID values with 0, rename the 'Salary' column to 'Annual_Salary', drop the 'Status' column, then give me a summary of the cleaned data")
    show("STRESS2: data cleaning (5 steps)", r)

elif test == "stress3":
    # 3-4 steps: load → aggregate → synthesize → export
    r = post(prompt=f"Load the CSV at {CSV_PATH}, calculate the average revenue per region, tell me which region has the highest average revenue, and export the aggregated results as a CSV file")
    show("STRESS3: analysis + export (4 steps)", r)

else:
    print(f"Unknown test: {test}. Available: health, capabilities, load_csv, load_xlsx, upload_file, export_csv, export_xlsx, process_data, filter_data, sort_data, aggregate_data, merge_data, add_column, rename_column, fill_missing, drop_column, get_summary, transform_data, list_files, nl_simple1, nl_simple2, nl_medium1, nl_medium2, nl_complex1, nl_complex2, stress1, stress2, stress3")

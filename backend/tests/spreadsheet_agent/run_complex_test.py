import httpx, json

BASE = "http://localhost:9000"
THREAD = "complex_test_v5"

prompt = (
    "I need a complete sales analysis report. Here is what I need you to do:\n"
    "1. Load the sales data from D:\\Internship\\Orbimesh\\backend\\tests\\test_data\\sales_data.csv\n"
    "2. Add a new column called Revenue_Per_Unit calculated as Revenue divided by Units\n"
    "3. Filter to only keep rows from the North and West regions\n"
    "4. Sort the filtered data by Revenue from highest to lowest\n"
    "5. Give me a detailed summary of this filtered and sorted dataset\n"
    "6. Export the final result as an Excel file called priority_markets_analysis.xlsx"
)

body = {"prompt": prompt, "thread_id": THREAD}
print("Sending complex 6-step real-world test...")
print(f"Prompt:\n{prompt}\n")

r = httpx.post(f"{BASE}/execute", json=body, timeout=180)
r.raise_for_status()
result = r.json()

status = "[PASS]" if result.get("success") or result.get("status") == "success" else "[FAIL]"
print(f"\n{status} [COMPLEX REAL-WORLD TEST: 6 steps]")

msg = result.get("message") or result.get("summary") or result.get("result", "")
print(f"  message : {msg}")

if result.get("error_message"):
    print(f"  ERROR   : {result['error_message']}")

if result.get("data"):
    data_str = json.dumps(result["data"], indent=2, default=str)
    print(f"  data    : {data_str[:3000]}")

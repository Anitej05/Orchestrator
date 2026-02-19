"""Raw response inspector — shows full JSON from /execute"""
import httpx, json, sys

BASE = "http://localhost:9000"
TEST_DATA = r"D:\Internship\Orbimesh\backend\tests\test_data"
CSV_PATH = TEST_DATA + r"\sales_data.csv"
THREAD = "raw_inspect"

body = {
    "prompt": "",
    "thread_id": THREAD,
    "action": "load_file",
    "payload": {"file_path": CSV_PATH, "thread_id": THREAD, "file_id": "sales_data"},
}

r = httpx.post(f"{BASE}/execute", json=body, timeout=120)
print("HTTP STATUS:", r.status_code)
try:
    data = r.json()
    print("FULL JSON:")
    print(json.dumps(data, indent=2, default=str)[:3000])
except Exception as e:
    print("PARSE ERROR:", e)
    print("RAW:", r.text[:2000])

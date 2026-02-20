import requests
import json
import time

BASE_URL = "http://localhost:8050"

def test_full_workflow():
    """Complete workflow: create -> edit -> analyze -> extract -> versions."""
    
    # 1. CREATE
    print("=== [1] CREATE ===")
    res = requests.post(f"{BASE_URL}/create", json={
        "content": "Quarterly Sales Report Q3 2026\n\nRegion: North America\nTotal Revenue: $2.5M\nTop Product: Widget Pro ($800K)\nGrowth: 15% QoQ\n\nRegion: Europe\nTotal Revenue: $1.8M\nTop Product: Widget Basic ($600K)\nGrowth: 8% QoQ\n\nKey Findings:\n- Widget Pro outperformed in NA\n- Europe showed slower growth due to supply chain delays\n- New markets in Asia pending Q4 launch",
        "file_name": "Q3_Report.docx",
        "file_type": "docx"
    })
    print(f"  Status: {res.status_code}, Success: {res.json().get('success')}")
    file_path = res.json().get("file_path")
    print(f"  File: {file_path}")
    time.sleep(1)

    # 2. EDIT
    print("\n=== [2] EDIT ===")
    res = requests.post(f"{BASE_URL}/edit", json={
        "file_path": file_path,
        "instruction": "Add a 'Recommendations' section at the end with: 1) Increase Widget Pro marketing budget by 20%. 2) Expedite Asia launch to early Q4. 3) Investigate European supply chain alternatives."
    })
    print(f"  Status: {res.status_code}, Message: {res.json().get('message')}")
    time.sleep(1)

    # 3. ANALYZE
    print("\n=== [3] ANALYZE ===")
    res = requests.post(f"{BASE_URL}/analyze", json={
        "file_path": file_path,
        "query": "What was the total revenue across all regions and which product performed best?"
    })
    print(f"  Status: {res.status_code}")
    print(f"  Answer: {res.json().get('answer', 'N/A')[:500]}")
    time.sleep(1)

    # 4. EXTRACT
    print("\n=== [4] EXTRACT (structured) ===")
    res = requests.post(f"{BASE_URL}/extract", json={
        "file_path": file_path,
        "extraction_type": "structured"
    })
    print(f"  Status: {res.status_code}")
    data = res.json().get("extracted_data", "")
    print(f"  Data: {str(data)[:500]}")
    time.sleep(1)

    # 5. VERSIONS
    print("\n=== [5] VERSION HISTORY ===")
    res = requests.post(f"{BASE_URL}/versions", json={
        "file_path": file_path
    })
    print(f"  Status: {res.status_code}")
    versions = res.json().get("versions", [])
    print(f"  Versions: {len(versions)}")
    for v in versions:
        print(f"    - {v.get('version_id')}: {v.get('description', 'N/A')}")

    # 6. METRICS
    print("\n=== [6] METRICS ===")
    res = requests.get(f"{BASE_URL}/metrics")
    print(f"  Status: {res.status_code}")
    metrics = res.json().get("metrics", res.json())
    print(f"  API calls: {metrics.get('api_calls', 'N/A')}")

if __name__ == "__main__":
    test_full_workflow()

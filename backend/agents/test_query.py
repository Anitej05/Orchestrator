import requests
import pandas as pd
import time

BASE_URL = "http://localhost:8041"

def test_query_assignment():
    # 1. Upload
    df = pd.DataFrame({
        'Feature1': [10, 20, 30],
        'Feature2': [1, 2, 3],
        'Feature3': [5, 5, 5],
        'Feature4': [2, 2, 2],
        'Feature5': [1, 1, 1]
    })
    csv_content = df.to_csv(index=False)
    files = {'file': ('test_assign.csv', csv_content, 'text/csv')}
    
    print("Uploading file...")
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    if resp.status_code != 200:
        print("Upload failed:", resp.text)
        return
    file_id = resp.json()['result']['file_id']
    print(f"File ID: {file_id}")

    # 2. Test Assignment via /query
    query = "Total = Feature1 + Feature2 + Feature3 + Feature4 + Feature5"
    print(f"\nQuery: {query}")
    resp = requests.post(f"{BASE_URL}/query", data={"file_id": file_id, "query": query})
    result = resp.json()
    
    if result.get('success'):
        print("✅ SUCCESS: Assignment worked!")
        # Check if 'Total' column exists
        if result.get('result', {}).get('query_result'):
            first_row = result['result']['query_result'][0]
            if 'Total' in first_row:
                print(f"   Total column found! First row Total = {first_row['Total']}")
                expected = 10 + 1 + 5 + 2 + 1
                if first_row['Total'] == expected:
                    print(f"   Value is correct: {expected}")
                else:
                    print(f"   Value mismatch: expected {expected}, got {first_row['Total']}")
            else:
                print("   ❌ Total column NOT found in result:", first_row.keys())
    else:
        print("❌ FAILED:", result.get('error'))

if __name__ == "__main__":
    test_query_assignment()

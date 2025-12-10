import requests
import pandas as pd

BASE_URL = "http://localhost:8041"

def test_slno_addition():
    """Test adding Sl.No. column - this requires special handling as it's not a simple column expression"""
    # 1. Upload
    df = pd.DataFrame({
        'Feature1': [10, 20, 30],
        'Feature2': [1, 2, 3],
    })
    csv_content = df.to_csv(index=False)
    files = {'file': ('test_slno.csv', csv_content, 'text/csv')}
    
    print("Uploading file...")
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    file_id = resp.json()['result']['file_id']
    print(f"File ID: {file_id}")

    # 2. Test using execute_pandas for Sl.No. (complex operation)
    print("\nTesting /execute_pandas for adding Sl.No. column...")
    resp = requests.post(f"{BASE_URL}/execute_pandas", data={
        "file_id": file_id,
        "instruction": "Add a Sl.No. column starting from 1"
    })
    result = resp.json()
    
    if result.get('success'):
        print("✅ SUCCESS: Sl.No. column added!")
        if result.get('result', {}).get('data'):
            first_row = result['result']['data'][1] if len(result['result']['data']) > 1 else None
            if first_row:
                print(f"   First data row: {first_row}")
    else:
        print("❌ FAILED:", result.get('error'))

if __name__ == "__main__":
    test_slno_addition()

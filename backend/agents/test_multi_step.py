import requests
import pandas as pd
import time

BASE_URL = "http://localhost:8041"

def test_multi_step():
    """Test that changes persist across multiple operations"""
    # 1. Upload
    df = pd.DataFrame({
        'Feature1': [10, 20, 30],
        'Feature2': [1, 2, 3],
        'Feature3': [5, 5, 5]
    })
    csv_content = df.to_csv(index=False)
    files = {'file': ('multi_step_test.csv', csv_content, 'text/csv')}
    
    print("1. Uploading file...")
    resp = requests.post(f"{BASE_URL}/upload", files=files)
    file_id = resp.json()['result']['file_id']
    print(f"   File ID: {file_id}")

    # 2. Add Sl.No. column
    print("\n2. Adding Sl.No. column...")
    resp = requests.post(f"{BASE_URL}/execute_pandas", data={
        "file_id": file_id,
        "instruction": "Add a Sl.No. column starting from 1"
    })
    result = resp.json()
    if result.get('success'):
        print("   ✅ Sl.No. added")
    else:
        print(f"   ❌ Failed: {result.get('error')}")
        return

    # 3. Add Total column
    print("\n3. Adding Total column...")
    resp = requests.post(f"{BASE_URL}/execute_pandas", data={
        "file_id": file_id,
        "instruction": "Add a Total column that sums Feature1, Feature2, Feature3"
    })
    result = resp.json()
    if result.get('success'):
        print("   ✅ Total added")
    else:
        print(f"   ❌ Failed: {result.get('error')}")
        return

    # 4. Rename columns
    print("\n4. Renaming Feature columns to Subject...")
    resp = requests.post(f"{BASE_URL}/execute_pandas", data={
        "file_id": file_id,
        "instruction": "Rename Feature1 to Subject1, Feature2 to Subject2, Feature3 to Subject3"
    })
    result = resp.json()
    if result.get('success'):
        print("   ✅ Columns renamed")
    else:
        print(f"   ❌ Failed: {result.get('error')}")
        return

    # 5. Display final state
    print("\n5. Displaying final spreadsheet...")
    resp = requests.post(f"{BASE_URL}/display", data={"file_id": file_id})
    result = resp.json()
    if result.get('success') and result.get('result', {}).get('data'):
        headers = result['result']['data'][0]
        print(f"   Final columns: {headers}")
        
        # Check if all expected columns are present
        expected = ['Sl.No.', 'Subject1', 'Subject2', 'Subject3', 'Total']
        missing = [col for col in expected if col not in headers]
        extra = [col for col in headers if col not in expected]
        
        if not missing:
            print("\n   🎉 SUCCESS: All columns preserved!")
        else:
            print(f"\n   ❌ FAILED: Missing columns: {missing}")
        if extra:
            print(f"   Extra columns: {extra}")
    else:
        print(f"   ❌ Display failed: {result.get('error')}")

if __name__ == "__main__":
    test_multi_step()

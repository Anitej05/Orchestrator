# backend/run_agent_tests.py
import httpx
import asyncio
import os

BASE_URL = "http://localhost:8041"
test_file_id = None

async def run_tests():
    global test_file_id
    
    print("--- Starting Spreadsheet Agent Tests ---")
    
    try:
        async with httpx.AsyncClient() as client:
            # 1. Test Health Check
            print("\n[TEST] /health ...")
            response = await client.get(f"{BASE_URL}/health")
            assert response.status_code == 200, f"Health check failed with status {response.status_code}"
            assert response.json()["status"] == "healthy", "Health check did not return 'healthy'"
            print("✅ PASSED: Health check is successful.")

            # 2. Test File Upload
            print("\n[TEST] /upload ...")
            file_path = os.path.join(os.path.dirname(__file__), 'tests', 'test_data.csv')
            if not os.path.exists(file_path):
                print(f"❌ FAILED: Test data file not found at {file_path}")
                return

            with open(file_path, "rb") as f:
                files = {"file": ("test_data.csv", f, "text/csv")}
                response = await client.post(f"{BASE_URL}/upload", files=files)

            assert response.status_code == 200, f"Upload failed with status {response.status_code}"
            json_response = response.json()
            assert json_response["success"] is True, "Upload API call was not successful"
            assert "file_id" in json_response["result"], "file_id not in upload response"
            test_file_id = json_response["result"]["file_id"]
            print(f"✅ PASSED: File uploaded successfully. File ID: {test_file_id}")

            # 3. Test Get Summary
            print("\n[TEST] /get_summary ...")
            data = {"file_id": test_file_id}
            response = await client.post(f"{BASE_URL}/get_summary", data=data)
            assert response.status_code == 200, f"Get summary failed with status {response.status_code}"
            json_response = response.json()
            assert json_response["success"] is True, "Get summary API call was not successful"
            result = json_response["result"]
            assert result["headers"] == ["name", "age", "city"]
            assert len(result["rows"]) == 3
            print("✅ PASSED: Get summary returned correct data.")

            # 4. Test Query Data
            print("\n[TEST] /query ...")
            data = {"file_id": test_file_id, "query": "age > 30"}
            response = await client.post(f"{BASE_URL}/query", data=data)
            assert response.status_code == 200, f"Query failed with status {response.status_code}"
            json_response = response.json()
            assert json_response["success"] is True, "Query API call was not successful"
            result = json_response["result"]["query_result"]
            assert len(result) == 1
            assert result[0]["name"] == "Charlie"
            print("✅ PASSED: Query returned correct filtered data.")
            
            # 5. Test Get Column Stats
            print("\n[TEST] /get_column_stats ...")
            data = {"file_id": test_file_id, "column_name": "age"}
            response = await client.post(f"{BASE_URL}/get_column_stats", data=data)
            assert response.status_code == 200, f"Get column stats failed with status {response.status_code}"
            json_response = response.json()
            assert json_response["success"] is True, "Get column stats API call was not successful"
            stats = json_response["result"]["column_stats"]
            assert stats["mean"] == 30.0
            print("✅ PASSED: Get column stats returned correct statistics.")

    except Exception as e:
        print(f"\n❌ An error occurred during testing: {e}")
    
    print("\n--- All tests completed ---")

if __name__ == "__main__":
    asyncio.run(run_tests())

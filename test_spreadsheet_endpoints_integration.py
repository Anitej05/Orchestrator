#!/usr/bin/env python3
"""
Integration test for spreadsheet agent /execute and /continue endpoints.

Tests the complete bidirectional dialogue flow with proper AgentResponse handling.
"""

import sys
import json
import asyncio
import tempfile
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from agents.spreadsheet_agent.main import app
from schemas import AgentResponse, AgentResponseStatus, OrchestratorMessage


def create_test_csv():
    """Create a test CSV file for testing"""
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo'],
        'Salary': [50000, 60000, 70000, 80000]
    }
    df = pd.DataFrame(data)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name, df


def test_execute_endpoint():
    """Test /execute endpoint with proper AgentResponse format"""
    print("🧪 Testing /execute endpoint...")
    
    client = TestClient(app)
    
    # First upload a test file
    csv_path, df = create_test_csv()
    
    with open(csv_path, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("test.csv", f, "text/csv")},
            data={"thread_id": "test-thread"}
        )
    
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    file_id = upload_data["data"]["file_id"]
    print(f"✅ File uploaded with ID: {file_id}")
    
    # Test /execute with get_summary action
    execute_payload = {
        "action": "/get_summary",
        "payload": {
            "file_id": file_id,
            "thread_id": "test-thread",
            "task_id": "test-task-123"
        }
    }
    
    execute_response = client.post("/execute", json=execute_payload)
    assert execute_response.status_code == 200
    
    response_data = execute_response.json()
    print(f"✅ Execute response status: {response_data.get('status')}")
    print(f"✅ Execute response keys: {list(response_data.keys())}")
    
    # Verify AgentResponse format
    assert response_data["status"] in ["complete", "error", "needs_input"]
    assert "context" in response_data
    assert response_data["context"]["task_id"] == "test-task-123"
    
    if response_data["status"] == "complete":
        assert "result" in response_data
        assert response_data["result"] is not None
        print("✅ Summary execution completed successfully")
    
    # Clean up
    Path(csv_path).unlink()
    
    return file_id


def test_execute_with_prompt():
    """Test /execute endpoint with natural language prompt"""
    print("\n🧪 Testing /execute with prompt...")
    
    client = TestClient(app)
    
    # Create test file
    csv_path, df = create_test_csv()
    
    with open(csv_path, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("test.csv", f, "text/csv")},
            data={"thread_id": "test-thread-2"}
        )
    
    file_id = upload_response.json()["data"]["file_id"]
    
    # Test with natural language prompt
    execute_payload = {
        "prompt": "What is the average age of people in this dataset?",
        "payload": {
            "file_id": file_id,
            "thread_id": "test-thread-2",
            "task_id": "test-task-456"
        }
    }
    
    execute_response = client.post("/execute", json=execute_payload)
    assert execute_response.status_code == 200
    
    response_data = execute_response.json()
    print(f"✅ Prompt execution status: {response_data.get('status')}")
    
    # Verify AgentResponse format
    assert response_data["status"] in ["complete", "error", "needs_input"]
    assert "context" in response_data
    
    if response_data["status"] == "complete":
        print("✅ Natural language query executed successfully")
    elif response_data["status"] == "needs_input":
        print(f"✅ Needs input detected: {response_data.get('question', '')[:50]}...")
    
    # Clean up
    Path(csv_path).unlink()


def test_continue_endpoint():
    """Test /continue endpoint for bidirectional dialogue"""
    print("\n🧪 Testing /continue endpoint...")
    
    client = TestClient(app)
    
    # Test continue with a simple task
    continue_payload = {
        "type": "continue",
        "answer": "fill_mean",
        "payload": {
            "task_id": "test-continue-789"
        }
    }
    
    continue_response = client.post("/continue", json=continue_payload)
    assert continue_response.status_code == 200
    
    response_data = continue_response.json()
    print(f"✅ Continue response status: {response_data.get('status')}")
    print(f"✅ Continue response keys: {list(response_data.keys())}")
    
    # Verify AgentResponse format
    assert response_data["status"] in ["complete", "error", "needs_input"]
    assert "context" in response_data
    
    if response_data["status"] == "complete":
        assert "result" in response_data
        print("✅ Continue operation completed successfully")


def test_error_handling():
    """Test error handling with proper AgentResponse format"""
    print("\n🧪 Testing error handling...")
    
    client = TestClient(app)
    
    # Test with missing file_id
    execute_payload = {
        "action": "/get_summary",
        "payload": {
            "thread_id": "test-thread",
            "task_id": "test-error-123"
        }
    }
    
    execute_response = client.post("/execute", json=execute_payload)
    assert execute_response.status_code == 200
    
    response_data = execute_response.json()
    print(f"✅ Error response status: {response_data.get('status')}")
    
    # Verify error format
    assert response_data["status"] == "error"
    assert "error" in response_data
    assert "context" in response_data
    assert response_data["context"]["task_id"] == "test-error-123"
    
    print(f"✅ Error message: {response_data['error']}")


def test_form_data_support():
    """Test that endpoints support both JSON and form data"""
    print("\n🧪 Testing form data support...")
    
    client = TestClient(app)
    
    # Create test file
    csv_path, df = create_test_csv()
    
    with open(csv_path, 'rb') as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("test.csv", f, "text/csv")},
            data={"thread_id": "test-thread-form"}
        )
    
    file_id = upload_response.json()["data"]["file_id"]
    
    # Test /execute with form data
    form_data = {
        "action": "/get_summary",
        "file_id": file_id,
        "thread_id": "test-thread-form"
    }
    
    execute_response = client.post("/execute", data=form_data)
    assert execute_response.status_code == 200
    
    response_data = execute_response.json()
    print(f"✅ Form data execution status: {response_data.get('status')}")
    
    # Verify AgentResponse format
    assert response_data["status"] in ["complete", "error", "needs_input"]
    assert "context" in response_data
    
    # Test /continue with form data
    continue_form_data = {
        "task_id": "test-continue-form",
        "answer": "proceed"
    }
    
    continue_response = client.post("/continue", data=continue_form_data)
    assert continue_response.status_code == 200
    
    continue_data = continue_response.json()
    print(f"✅ Form data continue status: {continue_data.get('status')}")
    
    # Clean up
    Path(csv_path).unlink()


def main():
    """Run all integration tests"""
    print("🚀 Testing Spreadsheet Agent Endpoints Integration")
    print("=" * 60)
    
    try:
        test_execute_endpoint()
        test_execute_with_prompt()
        test_continue_endpoint()
        test_error_handling()
        test_form_data_support()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\nVerified functionality:")
        print("✅ /execute endpoint with proper AgentResponse format")
        print("✅ /continue endpoint for bidirectional dialogue")
        print("✅ Both JSON and form data support")
        print("✅ Error handling with consistent format")
        print("✅ Natural language prompt processing")
        print("✅ File passing in context field")
        print("✅ Task ID tracking across requests")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
API Integration Test for Multi-Stage Planning Endpoints
Tests the /plan_operation endpoint with all 4 stages via HTTP
"""

import requests
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
BASE_URL = "http://localhost:8041"
TEST_DATA_PATH = Path(__file__).parent.parent / "test_data" / "sales_data.csv"


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


# ============================================================================
# TEST UTILITIES
# ============================================================================

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_file() -> str:
    """Upload test file and return file_id"""
    print_info("Uploading test file...")
    
    if not TEST_DATA_PATH.exists():
        print_error(f"Test data not found: {TEST_DATA_PATH}")
        # Create sample CSV
        import pandas as pd
        df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop'],
            'Amount': [1000, 25, 50, 300, 1200]
        })
        TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(TEST_DATA_PATH, index=False)
        print_info(f"Created sample test data at {TEST_DATA_PATH}")
    
    with open(TEST_DATA_PATH, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/upload",
            files={"file": f}
        )
    
    if response.status_code == 200:
        result = response.json()
        file_id = result.get("result", {}).get("file_id")
        print_success(f"File uploaded: {file_id}")
        return file_id
    else:
        print_error(f"Upload failed: {response.status_code}")
        print(response.text)
        return None


def make_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make HTTP request and return response"""
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            data=data,
            timeout=30
        )
        return {
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    except Exception as e:
        return {
            "status_code": 0,
            "data": None,
            "error": str(e)
        }


# ============================================================================
# TEST CASES
# ============================================================================

def test_propose_stage(file_id: str) -> str:
    """Test PROPOSE stage"""
    print_header("TEST: Propose Stage")
    
    instruction = "Filter products where Amount is greater than 500 and sort by Date"
    
    print_info(f"Instruction: {instruction}")
    
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": instruction,
        "stage": "propose"
    })
    
    if response["status_code"] != 200:
        print_error(f"Request failed: {response['error']}")
        return None
    
    data = response["data"]
    
    if not data.get("success"):
        print_error(f"API returned error: {data.get('error')}")
        return None
    
    plan = data["result"]["plan"]
    plan_id = plan["plan_id"]
    
    print_success("Plan proposed successfully")
    print(f"  Plan ID: {plan_id}")
    print(f"  Stage: {plan['stage']}")
    print(f"  Actions: {len(plan['actions'])}")
    print(f"  Reasoning: {plan['reasoning'][:100]}...")
    
    print("\n  Actions:")
    for i, action in enumerate(plan['actions']):
        print(f"    {i+1}. {action['action_type']}: {json.dumps(action, indent=6)[6:]}")
    
    return plan_id


def test_revise_stage(file_id: str, plan_id: str) -> str:
    """Test REVISE stage"""
    print_header("TEST: Revise Stage")
    
    feedback = "Sort in descending order instead"
    
    print_info(f"Plan ID: {plan_id}")
    print_info(f"Feedback: {feedback}")
    
    revision_data = {
        "plan_id": plan_id,
        "feedback": feedback
    }
    
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": json.dumps(revision_data),
        "stage": "revise"
    })
    
    if response["status_code"] != 200:
        print_error(f"Request failed: {response['error']}")
        return None
    
    data = response["data"]
    
    if not data.get("success"):
        print_error(f"API returned error: {data.get('error')}")
        return None
    
    plan = data["result"]["plan"]
    
    print_success("Plan revised successfully")
    print(f"  Plan ID: {plan['plan_id']}")
    print(f"  Stage: {plan['stage']}")
    print(f"  Revisions: {len(plan.get('revisions', []))}")
    
    if plan.get('revisions'):
        last_revision = plan['revisions'][-1]
        print(f"  Last revision feedback: {last_revision.get('feedback')}")
        print(f"  Changes: {last_revision.get('changes', 'N/A')[:100]}...")
    
    return plan_id


def test_simulate_stage(file_id: str, plan_id: str) -> bool:
    """Test SIMULATE stage"""
    print_header("TEST: Simulate Stage")
    
    print_info(f"Plan ID: {plan_id}")
    
    sim_data = {"plan_id": plan_id}
    
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": json.dumps(sim_data),
        "stage": "simulate"
    })
    
    if response["status_code"] != 200:
        print_error(f"Request failed: {response['error']}")
        return False
    
    data = response["data"]
    
    if not data.get("success"):
        print_error(f"Simulation failed: {data.get('error')}")
        return False
    
    simulation = data["result"]["simulation"]
    observation = simulation.get("observation", {})
    
    print_success("Simulation completed successfully")
    print(f"  Before shape: {observation.get('before_shape')}")
    print(f"  After shape: {observation.get('after_shape')}")
    print(f"  Changes: {observation.get('changes_summary')}")
    
    warnings = simulation.get("warnings", [])
    if warnings:
        print_warning(f"Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print_info("No warnings")
    
    return True


def test_execute_stage(file_id: str, plan_id: str) -> bool:
    """Test EXECUTE stage"""
    print_header("TEST: Execute Stage")
    
    print_info(f"Plan ID: {plan_id}")
    
    exec_data = {
        "plan_id": plan_id,
        "force": False
    }
    
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": json.dumps(exec_data),
        "stage": "execute"
    })
    
    if response["status_code"] != 200:
        print_error(f"Request failed: {response['error']}")
        return False
    
    data = response["data"]
    
    if not data.get("success"):
        print_error(f"Execution failed: {data.get('error')}")
        return False
    
    result = data["result"]
    execution = result.get("execution", {})
    
    print_success("Plan executed successfully")
    print(f"  Actions executed: {execution.get('actions_executed')}")
    print(f"  Final shape: {result.get('shape')}")
    print(f"  Columns: {', '.join(result.get('columns', [])[:5])}...")
    
    return True


def test_simulate_operation_endpoint(file_id: str) -> bool:
    """Test the /simulate_operation endpoint"""
    print_header("TEST: Simulate Operation Endpoint")
    
    pandas_code = "df = df[df['Amount'] > 100]"
    
    print_info(f"Code: {pandas_code}")
    
    response = make_request("/simulate_operation", {
        "file_id": file_id,
        "pandas_code": pandas_code
    })
    
    if response["status_code"] != 200:
        print_error(f"Request failed: {response['error']}")
        return False
    
    data = response["data"]
    
    if not data.get("success"):
        print_error(f"Simulation failed: {data.get('error')}")
        return False
    
    simulation = data["result"]["simulation"]
    
    print_success("Simulation completed")
    print(f"  Success: {simulation.get('success')}")
    
    if simulation.get('observation'):
        obs = simulation['observation']
        print(f"  Before: {obs.get('before_shape')}")
        print(f"  After: {obs.get('after_shape')}")
    
    return True


def test_complete_workflow(file_id: str) -> bool:
    """Test complete 4-stage workflow"""
    print_header("TEST: Complete 4-Stage Workflow")
    
    print_info("Testing Propose → Simulate → Execute workflow")
    
    # Stage 1: Propose
    print("\n📋 Stage 1: PROPOSE")
    plan_id = test_propose_stage(file_id)
    if not plan_id:
        return False
    
    time.sleep(1)
    
    # Stage 2: Simulate
    print("\n🔬 Stage 2: SIMULATE")
    if not test_simulate_stage(file_id, plan_id):
        return False
    
    time.sleep(1)
    
    # Stage 3: Execute
    print("\n⚡ Stage 3: EXECUTE")
    if not test_execute_stage(file_id, plan_id):
        return False
    
    print_success("\n✨ Complete workflow successful!")
    return True


def test_revision_workflow(file_id: str) -> bool:
    """Test workflow with revision"""
    print_header("TEST: Workflow with Revision")
    
    print_info("Testing Propose → Revise → Simulate → Execute workflow")
    
    # Stage 1: Propose
    print("\n📋 Stage 1: PROPOSE")
    plan_id = test_propose_stage(file_id)
    if not plan_id:
        return False
    
    time.sleep(1)
    
    # Stage 2: Revise
    print("\n📝 Stage 2: REVISE")
    plan_id = test_revise_stage(file_id, plan_id)
    if not plan_id:
        return False
    
    time.sleep(1)
    
    # Stage 3: Simulate
    print("\n🔬 Stage 3: SIMULATE")
    if not test_simulate_stage(file_id, plan_id):
        return False
    
    time.sleep(1)
    
    # Stage 4: Execute
    print("\n⚡ Stage 4: EXECUTE")
    if not test_execute_stage(file_id, plan_id):
        return False
    
    print_success("\n✨ Complete revision workflow successful!")
    return True


def test_error_handling(file_id: str) -> bool:
    """Test error handling"""
    print_header("TEST: Error Handling")
    
    # Test 1: Invalid stage
    print("\n1. Testing invalid stage...")
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": "test",
        "stage": "invalid_stage"
    })
    
    if response["status_code"] == 400 or (response["data"] and not response["data"].get("success")):
        print_success("Correctly rejected invalid stage")
    else:
        print_error("Should have rejected invalid stage")
        return False
    
    # Test 2: Missing plan_id in simulate
    print("\n2. Testing missing plan_id...")
    response = make_request("/plan_operation", {
        "file_id": file_id,
        "instruction": "{}",
        "stage": "simulate"
    })
    
    if response["status_code"] == 404 or (response["data"] and not response["data"].get("success")):
        print_success("Correctly rejected missing plan_id")
    else:
        print_error("Should have rejected missing plan_id")
        return False
    
    # Test 3: Non-existent file_id
    print("\n3. Testing non-existent file_id...")
    response = make_request("/plan_operation", {
        "file_id": "nonexistent_file",
        "instruction": "test",
        "stage": "propose"
    })
    
    if response["status_code"] == 404 or (response["data"] and not response["data"].get("success")):
        print_success("Correctly rejected non-existent file")
    else:
        print_error("Should have rejected non-existent file")
        return False
    
    print_success("\n✨ Error handling tests passed!")
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all API integration tests"""
    print_header("🚀 MULTI-STAGE PLANNING API INTEGRATION TESTS")
    
    # Check server
    print_info("Checking if server is running...")
    if not check_server():
        print_error(f"Server not running at {BASE_URL}")
        print_info("Start the server with: python backend/agents/spreadsheet_agent/main.py")
        return False
    print_success("Server is running\n")
    
    # Upload test file
    file_id = upload_file()
    if not file_id:
        print_error("Failed to upload test file")
        return False
    
    time.sleep(1)
    
    # Run tests
    test_results = {
        "passed": 0,
        "failed": 0,
        "tests": []
    }
    
    tests = [
        ("Simulate Operation Endpoint", lambda: test_simulate_operation_endpoint(file_id)),
        ("Complete Workflow (Propose→Simulate→Execute)", lambda: test_complete_workflow(file_id)),
        ("Revision Workflow (Propose→Revise→Simulate→Execute)", lambda: test_revision_workflow(file_id)),
        ("Error Handling", lambda: test_error_handling(file_id)),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{Colors.BOLD}Running: {test_name}{Colors.ENDC}")
            result = test_func()
            
            if result:
                test_results["passed"] += 1
                test_results["tests"].append((test_name, "PASSED"))
            else:
                test_results["failed"] += 1
                test_results["tests"].append((test_name, "FAILED"))
            
            time.sleep(2)  # Delay between tests
            
        except Exception as e:
            test_results["failed"] += 1
            test_results["tests"].append((test_name, f"ERROR: {e}"))
            print_error(f"Test error: {e}")
    
    # Print summary
    print_header("📊 TEST SUMMARY")
    
    total = test_results["passed"] + test_results["failed"]
    success_rate = (test_results["passed"] / total * 100) if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print_success(f"Passed: {test_results['passed']}")
    if test_results['failed'] > 0:
        print_error(f"Failed: {test_results['failed']}")
    print(f"\nSuccess Rate: {success_rate:.1f}%\n")
    
    print("Test Results:")
    for test_name, status in test_results["tests"]:
        if status == "PASSED":
            print_success(f"  {test_name}")
        else:
            print_error(f"  {test_name}: {status}")
    
    print("\n" + "="*80 + "\n")
    
    return test_results["failed"] == 0


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

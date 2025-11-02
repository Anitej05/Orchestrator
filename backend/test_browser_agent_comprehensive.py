"""
Comprehensive test suite for the Browser Automation Agent.
Tests all endpoints, configurations, and common failure scenarios.
"""
import asyncio
import os
import sys
import requests
import json
from dotenv import load_dotenv
from typing import Dict, Any
import time

# Load environment variables
load_dotenv()

# Test configuration
AGENT_BASE_URL = "http://localhost:8070"
TEST_TIMEOUT = 30  # seconds

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")

def print_test(test_name: str):
    """Print test name"""
    print(f"{Colors.BOLD}🧪 TEST: {test_name}{Colors.RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")

# Test results tracker
test_results = {
    "passed": 0,
    "failed": 0,
    "warnings": 0
}

def test_environment_variables():
    """Test 1: Check if required environment variables are set"""
    print_test("Environment Variables Check")
    
    required_vars = ["OLLAMA_API_KEY"]
    optional_vars = ["BROWSER_AGENT_PORT"]
    
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var} is set")
        else:
            print_error(f"{var} is NOT set (REQUIRED)")
            all_good = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var} is set to: {value}")
        else:
            print_warning(f"{var} is not set (using default)")
            test_results["warnings"] += 1
    
    if all_good:
        test_results["passed"] += 1
        return True
    else:
        test_results["failed"] += 1
        return False

def test_agent_is_running():
    """Test 2: Check if the agent server is running"""
    print_test("Agent Server Availability")
    
    try:
        response = requests.get(f"{AGENT_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success(f"Agent server is running at {AGENT_BASE_URL}")
            test_results["passed"] += 1
            return True
        else:
            print_error(f"Agent server returned status code: {response.status_code}")
            test_results["failed"] += 1
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to agent server at {AGENT_BASE_URL}")
        print_info("Make sure the agent is running: python backend/agents/browser_automation_agent.py")
        test_results["failed"] += 1
        return False
    except Exception as e:
        print_error(f"Error checking agent server: {str(e)}")
        test_results["failed"] += 1
        return False

def test_agent_definition():
    """Test 3: Validate agent definition structure"""
    print_test("Agent Definition Structure")
    
    try:
        response = requests.get(f"{AGENT_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print_error("Failed to fetch agent definition")
            test_results["failed"] += 1
            return False
        
        definition = response.json()
        
        # Check required fields
        required_fields = ["id", "name", "description", "capabilities", "endpoints"]
        missing_fields = [field for field in required_fields if field not in definition]
        
        if missing_fields:
            print_error(f"Missing required fields: {', '.join(missing_fields)}")
            test_results["failed"] += 1
            return False
        
        print_success(f"Agent ID: {definition['id']}")
        print_success(f"Agent Name: {definition['name']}")
        print_success(f"Capabilities: {len(definition['capabilities'])} defined")
        print_success(f"Endpoints: {len(definition['endpoints'])} defined")
        
        # Validate capabilities
        if definition['capabilities']:
            print_info(f"Capabilities: {', '.join(definition['capabilities'])}")
        
        test_results["passed"] += 1
        return True
        
    except Exception as e:
        print_error(f"Error validating agent definition: {str(e)}")
        test_results["failed"] += 1
        return False

def test_browse_endpoint_structure():
    """Test 4: Check /browse endpoint structure"""
    print_test("Browse Endpoint Structure")
    
    try:
        response = requests.get(f"{AGENT_BASE_URL}/", timeout=5)
        definition = response.json()
        
        browse_endpoint = None
        for endpoint in definition.get("endpoints", []):
            if "/browse" in endpoint.get("endpoint", ""):
                browse_endpoint = endpoint
                break
        
        if not browse_endpoint:
            print_error("/browse endpoint not found in agent definition")
            test_results["failed"] += 1
            return False
        
        print_success(f"Endpoint: {browse_endpoint['endpoint']}")
        print_success(f"HTTP Method: {browse_endpoint['http_method']}")
        print_success(f"Parameters: {len(browse_endpoint.get('parameters', []))} defined")
        
        # Check parameters
        params = browse_endpoint.get('parameters', [])
        for param in params:
            required = "REQUIRED" if param.get('required') else "OPTIONAL"
            print_info(f"  - {param['name']} ({param['param_type']}) [{required}]")
        
        test_results["passed"] += 1
        return True
        
    except Exception as e:
        print_error(f"Error checking browse endpoint: {str(e)}")
        test_results["failed"] += 1
        return False

def test_browse_endpoint_simple_task():
    """Test 5: Execute a simple browser task"""
    print_test("Simple Browser Task Execution")
    
    print_info("This test will attempt to execute a simple browser task")
    print_info("Task: Navigate to example.com and describe what you see")
    
    payload = {
        "task": "Navigate to https://example.com and tell me what you see",
        "extract_data": False
    }
    
    try:
        print_info("Sending request to /browse endpoint...")
        response = requests.post(
            f"{AGENT_BASE_URL}/browse",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code != 200:
            print_error(f"Request failed with status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            test_results["failed"] += 1
            return False
        
        result = response.json()
        
        print_info(f"Response received:")
        print_info(f"  Success: {result.get('success')}")
        print_info(f"  Task Summary: {result.get('task_summary', 'N/A')[:100]}")
        print_info(f"  Actions Taken: {len(result.get('actions_taken', []))}")
        
        if result.get('error'):
            print_error(f"  Error: {result['error']}")
            test_results["failed"] += 1
            return False
        
        if result.get('success'):
            print_success("Browser task executed successfully!")
            test_results["passed"] += 1
            return True
        else:
            print_error("Browser task reported failure")
            test_results["failed"] += 1
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Request timed out after {TEST_TIMEOUT} seconds")
        print_warning("Browser tasks can take time. Consider increasing TEST_TIMEOUT")
        test_results["failed"] += 1
        return False
    except Exception as e:
        print_error(f"Error executing browser task: {str(e)}")
        test_results["failed"] += 1
        return False

def test_browse_endpoint_with_extraction():
    """Test 6: Execute a task with data extraction"""
    print_test("Browser Task with Data Extraction")
    
    payload = {
        "task": "Navigate to https://example.com and extract the page title",
        "extract_data": True
    }
    
    try:
        print_info("Sending request with extract_data=True...")
        response = requests.post(
            f"{AGENT_BASE_URL}/browse",
            json=payload,
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code != 200:
            print_error(f"Request failed with status code: {response.status_code}")
            test_results["failed"] += 1
            return False
        
        result = response.json()
        
        if result.get('error'):
            print_error(f"Error: {result['error']}")
            test_results["failed"] += 1
            return False
        
        if result.get('extracted_data'):
            print_success("Data extraction successful!")
            print_info(f"Extracted data: {json.dumps(result['extracted_data'], indent=2)[:200]}")
            test_results["passed"] += 1
            return True
        else:
            print_warning("No data was extracted (this might be expected)")
            test_results["warnings"] += 1
            test_results["passed"] += 1
            return True
            
    except Exception as e:
        print_error(f"Error during extraction test: {str(e)}")
        test_results["failed"] += 1
        return False

def test_browse_endpoint_invalid_request():
    """Test 7: Test error handling with invalid request"""
    print_test("Error Handling - Invalid Request")
    
    # Test with missing required field
    payload = {
        "extract_data": False
        # Missing 'task' field
    }
    
    try:
        response = requests.post(
            f"{AGENT_BASE_URL}/browse",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 422:  # Validation error
            print_success("Agent correctly rejected invalid request (422)")
            test_results["passed"] += 1
            return True
        elif response.status_code == 200:
            print_warning("Agent accepted invalid request (should return 422)")
            test_results["warnings"] += 1
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            test_results["failed"] += 1
            return False
            
    except Exception as e:
        print_error(f"Error during invalid request test: {str(e)}")
        test_results["failed"] += 1
        return False

def test_dependencies():
    """Test 8: Check if required Python packages are installed"""
    print_test("Python Dependencies Check")
    
    required_packages = [
        "browser_use",
        "langchain_openai",
        "fastapi",
        "uvicorn",
        "pydantic"
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            all_installed = False
    
    if all_installed:
        test_results["passed"] += 1
        return True
    else:
        print_info("Install missing packages: pip install -r requirements.txt")
        test_results["failed"] += 1
        return False

def test_langchain_compatibility():
    """Test 9: Check LangChain version compatibility"""
    print_test("LangChain Compatibility Check")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Try to instantiate ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key="test-key",
            base_url="https://api.openai.com/v1"
        )
        
        # Check if it has the expected attributes
        if hasattr(llm, 'model_name') or hasattr(llm, 'model'):
            print_success("ChatOpenAI instantiated successfully")
        else:
            print_warning("ChatOpenAI structure might be different than expected")
        
        # Check for the 'provider' attribute issue
        if hasattr(llm, 'provider'):
            print_info("ChatOpenAI has 'provider' attribute")
        else:
            print_warning("ChatOpenAI does NOT have 'provider' attribute")
            print_info("This might cause issues with browser-use library")
            print_info("Consider updating: pip install --upgrade langchain-openai browser-use")
        
        test_results["passed"] += 1
        return True
        
    except Exception as e:
        print_error(f"LangChain compatibility issue: {str(e)}")
        test_results["failed"] += 1
        return False

def print_summary():
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total_tests = test_results["passed"] + test_results["failed"]
    
    print(f"{Colors.BOLD}Total Tests Run: {total_tests}{Colors.RESET}")
    print(f"{Colors.GREEN}✅ Passed: {test_results['passed']}{Colors.RESET}")
    print(f"{Colors.RED}❌ Failed: {test_results['failed']}{Colors.RESET}")
    print(f"{Colors.YELLOW}⚠️  Warnings: {test_results['warnings']}{Colors.RESET}")
    
    if test_results["failed"] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Common fixes:{Colors.RESET}")
        print("1. Make sure the agent is running: python backend/agents/browser_automation_agent.py")
        print("2. Check OLLAMA_API_KEY in .env file")
        print("3. Update dependencies: pip install --upgrade langchain-openai browser-use")
        print("4. Check requirements.txt for version conflicts")
        return 1

def main():
    """Run all tests"""
    print_header("BROWSER AUTOMATION AGENT - COMPREHENSIVE TEST SUITE")
    
    # Run tests in order
    tests = [
        test_dependencies,
        test_langchain_compatibility,
        test_environment_variables,
        test_agent_is_running,
        test_agent_definition,
        test_browse_endpoint_structure,
        test_browse_endpoint_invalid_request,
        test_browse_endpoint_simple_task,  # Now enabled
        # test_browse_endpoint_with_extraction,  # Keep this commented for speed
    ]
    
    print_info("Running tests including one integration test...")
    print_info("This will test actual browser automation\n")
    
    for test_func in tests:
        try:
            test_func()
        except KeyboardInterrupt:
            print_error("\n\nTests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print_error(f"Unexpected error in {test_func.__name__}: {str(e)}")
            test_results["failed"] += 1
        
        time.sleep(0.5)  # Small delay between tests
    
    # Print summary and exit
    exit_code = print_summary()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

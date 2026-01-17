#!/usr/bin/env python3
"""
Comprehensive validation script for the intelligent spreadsheet parsing system.
Tests against real-world files and edge cases to ensure robust handling.
"""

import os
import sys
import pandas as pd
import traceback
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.spreadsheet_agent.agent import SpreadsheetAgent
from services.file_processor import FileProcessor

def test_file_parsing(file_path: str, agent: SpreadsheetAgent) -> dict:
    """Test parsing a single file and return results."""
    try:
        print(f"\n=== Testing: {os.path.basename(file_path)} ===")
        
        # Create a mock file_id for testing
        file_id = f"test_{os.path.basename(file_path)}"
        
        # Create request for analysis
        request = {
            "thread_id": "validation_thread",
            "file_id": file_id,
            "action": "analyze",
            "prompt": "Analyze this spreadsheet and provide a summary",
            "parameters": {"file_path": file_path}
        }
        
        # Execute the request
        response = agent.execute(request)
        
        result = {
            "file": os.path.basename(file_path),
            "status": response.get("status", "UNKNOWN"),
            "success": response.get("status") in ["COMPLETE", "NEEDS_INPUT"],
            "error": None,
            "explanation": response.get("explanation", "")[:200] + "..." if len(response.get("explanation", "")) > 200 else response.get("explanation", "")
        }
        
        if response.get("status") == "ERROR":
            result["error"] = response.get("explanation", "Unknown error")
            
        print(f"Status: {response.get('status', 'UNKNOWN')}")
        print(f"Success: {result['success']}")
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Summary: {result['explanation']}")
            
        return result
        
    except Exception as e:
        error_msg = f"Exception: {str(e)}"
        print(f"FAILED: {error_msg}")
        traceback.print_exc()
        return {
            "file": os.path.basename(file_path),
            "status": "EXCEPTION",
            "success": False,
            "error": error_msg,
            "explanation": ""
        }

def test_context_preservation(agent: SpreadsheetAgent) -> dict:
    """Test anti-hallucination measures and context preservation."""
    print("\n=== Testing Context Preservation & Anti-Hallucination ===")
    
    try:
        # Test with a known file
        test_file = "backend/tests/test_data/sales_data.csv"
        if not os.path.exists(test_file):
            return {"success": False, "error": "Test file not found"}
            
        file_id = "context_test"
        
        # First, analyze the file
        analyze_request = {
            "thread_id": "context_thread",
            "file_id": file_id,
            "action": "analyze",
            "parameters": {"file_path": test_file}
        }
        
        analyze_response = agent.execute(analyze_request)
        
        if analyze_response.get("status") != "COMPLETE":
            return {"success": False, "error": f"Analysis failed: {analyze_response.get('explanation', 'Unknown error')}"}
        
        # Test specific queries that could lead to hallucination
        test_queries = [
            "How many rows are in this dataset?",
            "What are the column names?",
            "What is the sum of all numeric columns?",
            "Show me the first 3 rows"
        ]
        
        results = []
        for query in test_queries:
            query_request = {
                "thread_id": "context_thread",
                "file_id": file_id,
                "action": "query",
                "prompt": query,
                "parameters": {}
            }
            
            query_response = agent.execute(query_request)
            results.append({
                "query": query,
                "status": query_response.get("status", "UNKNOWN"),
                "success": query_response.get("status") == "COMPLETE"
            })
            
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        
        return {
            "success": success_rate >= 0.75,  # At least 75% success rate
            "success_rate": success_rate,
            "results": results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Run comprehensive validation."""
    print("Starting Comprehensive Validation of Intelligent Spreadsheet Parsing System")
    print("=" * 80)
    
    # Initialize the agent
    agent = SpreadsheetAgent()
    
    # Test files to validate against
    test_files = []
    
    # Add real-world test data files
    test_data_dir = Path("backend/tests/test_data")
    if test_data_dir.exists():
        for file_path in test_data_dir.glob("*.xlsx"):
            test_files.append(str(file_path))
        for file_path in test_data_dir.glob("*.csv"):
            test_files.append(str(file_path))
    
    # Add edge case files
    edge_case_dir = Path("backend/tests/spreadsheet_agent/edge_case_datasets")
    if edge_case_dir.exists():
        for file_path in edge_case_dir.glob("*.xlsx"):
            test_files.append(str(file_path))
        for file_path in edge_case_dir.glob("*.csv"):
            test_files.append(str(file_path))
    
    if not test_files:
        print("No test files found!")
        return False
    
    print(f"Found {len(test_files)} test files to validate")
    
    # Test each file
    results = []
    for file_path in test_files:
        if os.path.exists(file_path):
            result = test_file_parsing(file_path, agent)
            results.append(result)
    
    # Test context preservation
    context_result = test_context_preservation(agent)
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    successful_files = [r for r in results if r["success"]]
    failed_files = [r for r in results if not r["success"]]
    
    print(f"Total files tested: {len(results)}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Success rate: {len(successful_files)/len(results)*100:.1f}%")
    
    if failed_files:
        print("\nFailed files:")
        for result in failed_files:
            print(f"  - {result['file']}: {result['error']}")
    
    print(f"\nContext preservation test: {'PASSED' if context_result['success'] else 'FAILED'}")
    if not context_result['success']:
        print(f"  Error: {context_result.get('error', 'Unknown error')}")
    
    # Overall success criteria
    overall_success = (
        len(successful_files) / len(results) >= 0.8 and  # 80% file success rate
        context_result['success']  # Context preservation must pass
    )
    
    print(f"\nOVERALL VALIDATION: {'PASSED' if overall_success else 'FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify the agent endpoints handle missing parameters gracefully.
"""

import requests
import json
import pandas as pd

def test_agent_endpoints():
    """Test the spreadsheet agent endpoints with missing parameters"""
    
    base_url = "http://localhost:8041"
    
    # First, we need to upload a file to get a file_id
    # For this test, we'll assume the agent is running and has a file
    
    print("🧪 Testing Spreadsheet Agent Endpoint Fixes")
    
    # Test data that matches the actual dataset structure
    test_file_id = "test_file_123"  # This would normally come from upload
    
    # Test 1: /execute_pandas with missing instruction
    print("\n1. Testing /execute_pandas with missing instruction...")
    try:
        response = requests.post(f"{base_url}/execute_pandas", data={
            "file_id": test_file_id
            # No instruction provided
        })
        print(f"Status: {response.status_code}")
        if response.status_code != 422:
            print("✅ /execute_pandas handles missing instruction gracefully")
        else:
            print("❌ /execute_pandas still requires instruction")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing /execute_pandas: {e}")
    
    # Test 2: /transform with missing operation/params
    print("\n2. Testing /transform with missing operation/params...")
    try:
        response = requests.post(f"{base_url}/transform", data={
            "file_id": test_file_id
            # No operation or params provided
        })
        print(f"Status: {response.status_code}")
        if response.status_code != 422:
            print("✅ /transform handles missing parameters gracefully")
        else:
            print("❌ /transform still requires parameters")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing /transform: {e}")
    
    # Test 3: /plan_operation with missing instruction
    print("\n3. Testing /plan_operation with missing instruction...")
    try:
        response = requests.post(f"{base_url}/plan_operation", data={
            "file_id": test_file_id
            # No instruction provided
        })
        print(f"Status: {response.status_code}")
        if response.status_code != 422:
            print("✅ /plan_operation handles missing instruction gracefully")
        else:
            print("❌ /plan_operation still requires instruction")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing /plan_operation: {e}")
    
    # Test 4: /execute endpoint (new)
    print("\n4. Testing /execute endpoint...")
    try:
        response = requests.post(f"{base_url}/execute", data={
            "instruction": "List all categories and their total quantities",
            "file_id": test_file_id
        })
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ /execute endpoint works")
        else:
            print("❌ /execute endpoint has issues")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing /execute: {e}")

def test_expected_results():
    """Show what the expected results should be"""
    print("\n📊 Expected Results from Actual Dataset:")
    try:
        df = pd.read_excel('backend/tests/test_data/retail_sales_dataset.xlsx')
        result = df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False)
        print("Categories and total quantities:")
        for category, total in result.items():
            print(f"  {category}: {total}")
    except Exception as e:
        print(f"Could not load test data: {e}")

if __name__ == "__main__":
    test_expected_results()
    print("\n" + "="*50)
    test_agent_endpoints()
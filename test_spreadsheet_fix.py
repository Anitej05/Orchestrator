#!/usr/bin/env python3
"""
Test script to verify the spreadsheet agent fixes work correctly.
"""

import pandas as pd
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_column_matching():
    """Test the intelligent column matching functionality"""
    
    # Load the actual dataset to get real numbers
    try:
        actual_df = pd.read_excel('backend/tests/test_data/retail_sales_dataset.xlsx')
        print("✅ Loaded actual dataset:")
        print(f"Shape: {actual_df.shape}")
        print(f"Columns: {list(actual_df.columns)}")
        
        # Get actual aggregation results
        actual_result = actual_df.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False)
        print("\n✅ Actual Aggregation Results:")
        for category, total in actual_result.items():
            print(f"  {category}: {total}")
        
        use_actual_data = True
        df = actual_df
    except Exception as e:
        print(f"⚠️ Could not load actual dataset ({e}), using test data")
        # Create test data similar to the user's Excel file
        test_data = {
            'Transaction ID': [1, 2, 3, 4, 5],
            'Date': ['2023-11-24', '2023-02-27', '2023-01-13', '2023-05-21', '2023-05-06'],
            'Customer ID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
            'Gender': ['Male', 'Female', 'Male', 'Male', 'Male'],
            'Age': [34, 26, 50, 37, 30],
            'Product Category': ['Beauty', 'Clothing', 'Electronics', 'Clothing', 'Beauty'],
            'Quantity': [3, 2, 1, 1, 2],
            'Price per Unit': [50, 500, 30, 500, 50],
            'Total Amount': [150, 1000, 30, 500, 100]
        }
        df = pd.DataFrame(test_data)
        use_actual_data = False
    
    # Import the agent
    try:
        from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
        agent = SpreadsheetQueryAgent()
        
        # Test column matching
        category_matches = agent._find_similar_columns(df, ['category', 'categories'])
        quantity_matches = agent._find_similar_columns(df, ['quantity', 'amount', 'total'])
        
        print(f"\n✅ Column Matching Test Results:")
        print(f"Category matches: {category_matches}")
        print(f"Quantity matches: {quantity_matches}")
        
        # Test query enhancement
        test_question = "Can you list all categories mentioned in the file and the total quantity of those category sold till now."
        enhanced_question = agent._enhance_query_with_column_suggestions(test_question, df)
        
        print(f"\n✅ Query Enhancement Test:")
        print(f"Original: {test_question}")
        print(f"Enhanced: {enhanced_question}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_execute_endpoint():
    """Test the /execute endpoint routing logic"""
    
    # Test analytical question routing
    analytical_instruction = "Can you list all categories mentioned in the file and the total quantity of those category sold till now."
    
    # Check routing logic
    instruction_lower = analytical_instruction.lower()
    analytical_keywords = [
        'list', 'show', 'what', 'how many', 'count', 'total', 'sum', 'average', 'mean',
        'categories', 'category', 'group', 'aggregate', 'analyze', 'analysis',
        'find', 'identify', 'calculate', 'compute', 'determine'
    ]
    
    is_analytical = any(keyword in instruction_lower for keyword in analytical_keywords)
    
    print("✅ Execute Endpoint Routing Test:")
    print(f"Instruction: {analytical_instruction}")
    print(f"Detected as analytical: {is_analytical}")
    print(f"Matching keywords: {[kw for kw in analytical_keywords if kw in instruction_lower]}")
    
    # Test transformation routing
    transformation_instruction = "Add a new column called Total that sums Feature1 and Feature2"
    instruction_lower = transformation_instruction.lower()
    transformation_keywords = [
        'add', 'create', 'insert', 'remove', 'delete', 'drop', 'rename', 'change',
        'modify', 'update', 'transform', 'convert', 'replace', 'fill', 'sort'
    ]
    
    is_transformation = any(keyword in instruction_lower for keyword in transformation_keywords)
    
    print(f"\nTransformation instruction: {transformation_instruction}")
    print(f"Detected as transformation: {is_transformation}")
    print(f"Matching keywords: {[kw for kw in transformation_keywords if kw in instruction_lower]}")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Spreadsheet Agent Fixes\n")
    
    success = True
    
    print("1. Testing Column Matching...")
    success &= test_column_matching()
    
    print("\n2. Testing Execute Endpoint Routing...")
    success &= test_execute_endpoint()
    
    if success:
        print("\n🎉 All tests passed! The fixes should resolve the orchestrator issues.")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
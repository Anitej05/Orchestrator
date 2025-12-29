import asyncio
import logging
import sys
import pandas as pd
from pathlib import Path
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import spreadsheet agent components directly
from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
from agents.spreadsheet_agent.code_generator import generate_modification_code
from agents.spreadsheet_agent.utils import load_dataframe
from agents.spreadsheet_agent.session import store_dataframe, get_dataframe

# Create global query agent instance
query_agent_instance = SpreadsheetQueryAgent()

async def test_analyze_spreadsheet():
    """Test 1: Analyze spreadsheet data with natural language query."""
    print("\n\n" + "="*60)
    print("📊 TEST 1: Analyze Spreadsheet Data")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    # Example: backend/tests/test_data/sales_data.csv
    test_file = "backend/tests/test_data/sales_data.csv"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("📁 Please add test files to: backend/tests/test_data/")
        print("   Supported formats: CSV, XLSX, XLS")
        print("   Example CSV content:")
        print("   Date,Product,Amount")
        print("   2025-01-01,ProductA,1500")
        print("   2025-01-02,ProductB,2300")
        return
    
    try:
        # Load the spreadsheet
        df = load_dataframe(test_file)
        thread_id = "test-spreadsheet-1"
        
        # Store in session
        store_dataframe(thread_id, df, test_file)
        
        # Query using LLM agent - call query method with proper kwargs
        query = "What is the total sales amount and which product sold the most?"
        result = await query_agent_instance.query(df, query, thread_id=thread_id, max_iterations=3)
        
        print(f"✅ Analysis Result: {result.answer if hasattr(result, 'answer') else result}")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_execute_pandas():
    """Test 2: Execute pandas operations on spreadsheet."""
    print("\n\n" + "="*60)
    print("🐼 TEST 2: Execute Pandas Operations")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    test_file = "backend/tests/test_data/sales_data.csv"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("📁 Please add a CSV/XLSX file to: backend/tests/test_data/")
        return
    
    try:
        # Load the spreadsheet
        df = load_dataframe(test_file)
        thread_id = "test-spreadsheet-2"
        
        # Store in session
        store_dataframe(thread_id, df, test_file)
        
        # Generate and execute code - pass DataFrame not thread_id
        instruction = "Calculate the average of the Amount column and show the sum"
        code = await generate_modification_code(df, instruction)
        
        print(f"✅ Generated Code:")
        if code:
            print(f"   {code[:200]}...")
        else:
            print(f"   No code generated")
        print(f"📊 Original DataFrame shape: {df.shape}")
        
        # Execute the code (simplified for testing)
        if code and ('mean()' in code or 'average' in code.lower()):
            print(f"✅ Code generation successful - would calculate averages")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_get_info():
    """Test 3: Get spreadsheet metadata and structure."""
    print("\n\n" + "="*60)
    print("ℹ️  TEST 3: Get Spreadsheet Info")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    test_file = "backend/tests/test_data/sales_data.csv"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("📁 Please add a CSV/XLSX file to: backend/tests/test_data/")
        return
    
    try:
        # Load the spreadsheet
        df = load_dataframe(test_file)
        
        print(f"✅ Spreadsheet Info:")
        print(f"   📏 Rows: {len(df)}")
        print(f"   📊 Columns: {len(df.columns)}")
        print(f"   📋 Column names: {list(df.columns)}")
        print(f"   🔢 Data types: {dict(df.dtypes)}")
        print(f"   📑 Sample data (first 3 rows):")
        print(f"{df.head(3).to_string()}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_data_transformation():
    """Test 4: Complex data transformation with natural language."""
    print("\n\n" + "="*60)
    print("🔄 TEST 4: Data Transformation")
    print("="*60 + "\n")
    
    test_file = "backend/tests/test_data/sales_data.csv"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return
    
    try:
        # Load the spreadsheet
        df = load_dataframe(test_file)
        thread_id = "test-spreadsheet-4"
        
        # Store in session
        store_dataframe(thread_id, df, test_file)
        
        # Generate transformation code - pass DataFrame
        instruction = """Filter rows where Amount > 1000, group by Product, and calculate total sales"""
        code = await generate_modification_code(df, instruction)
        
        print(f"✅ Generated Transformation Code:")
        if code:
            print(f"   {code[:300]}...")
        else:
            print(f"   No code generated")
        print(f"📊 Original data rows: {len(df)}")
        
        # Simple transformation example
        if 'Amount' in df.columns:
            filtered = df[df['Amount'] > 1000] if pd.api.types.is_numeric_dtype(df['Amount']) else df
            print(f"📊 Would filter to {len(filtered)} rows where Amount > 1000")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all spreadsheet agent tests."""
    print("\n\n" + "🟢"*30)
    print("🚀 SPREADSHEET AGENT MANUAL TESTS")
    print("🟢"*30 + "\n")
    
    # Run tests
    await test_analyze_spreadsheet()
    await test_execute_pandas()
    await test_get_info()
    await test_data_transformation()
    
    print("\n\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

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
    test_file = "tests/test_data/sales_data.csv"
    
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
        
        # Display execution metrics if available
        if hasattr(result, 'execution_metrics') and result.execution_metrics:
            _display_metrics(result.execution_metrics, "Query Analysis")
        
        # Display session metrics
        _display_session_metrics(query_agent_instance.get_metrics())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def _display_metrics(metrics: dict, operation_name: str = "Operation"):
    """Display execution metrics in a beautiful format."""
    print("\n" + "━"*70)
    print(f"📊 {operation_name} - EXECUTION METRICS")
    print("━"*70)
    
    # Performance metrics
    print("\n⏱️  PERFORMANCE:")
    print(f"  Total Latency:        {metrics.get('latency_ms', 0):.2f} ms")
    print(f"  Cache Hit:            {'✅ Yes' if metrics.get('cache_hit') else '❌ No'}")
    print(f"  Iterations:           {metrics.get('iterations', 0)}")
    
    # LLM metrics
    if metrics.get('llm_calls', 0) > 0:
        print("\n🤖 LLM STATISTICS:")
        print(f"  API Calls:            {metrics.get('llm_calls', 0)}")
        print(f"  Retries:              {metrics.get('retries', 0)}")
        print(f"  Input Tokens:         {metrics.get('tokens_input', 0):,}")
        print(f"  Output Tokens:        {metrics.get('tokens_output', 0):,}")
        print(f"  Total Tokens:         {(metrics.get('tokens_input', 0) + metrics.get('tokens_output', 0)):,}")
    
    # Resource usage
    if 'memory_used_mb' in metrics:
        print("\n💾 RESOURCE USAGE:")
        print(f"  Memory Used:          {metrics.get('memory_used_mb', 0):.2f} MB")
    
    print("━"*70)


def _display_session_metrics(metrics: dict):
    """Display session-level metrics."""
    print("\n" + "═"*70)
    print("🎯 SESSION-LEVEL METRICS")
    print("═"*70)
    
    # Query statistics
    queries = metrics.get('queries', {})
    print("\n📊 QUERIES:")
    print(f"  Total:                {queries.get('total', 0)}")
    print(f"  Successful:           {queries.get('successful', 0)} ✅")
    print(f"  Failed:               {queries.get('failed', 0)} ❌")
    print(f"  Success Rate:         {metrics.get('success_rate', 0):.1f}%")
    
    # Performance statistics
    perf = metrics.get('performance', {})
    print("\n⚡ PERFORMANCE:")
    print(f"  Avg Latency:          {perf.get('avg_latency_ms', 0):.2f} ms")
    print(f"  Completed:            {perf.get('queries_completed', 0)}")
    
    # LLM statistics
    llm = metrics.get('llm_calls', {})
    print("\n🤖 LLM CALLS:")
    print(f"  Total:                {llm.get('total', 0)}")
    if llm.get('groq', 0) > 0:
        print(f"  Groq:                 {llm.get('groq', 0)}")
    if llm.get('cerebras', 0) > 0:
        print(f"  Cerebras:             {llm.get('cerebras', 0)}")
    if llm.get('nvidia', 0) > 0:
        print(f"  NVIDIA:               {llm.get('nvidia', 0)}")
    if llm.get('google', 0) > 0:
        print(f"  Google:               {llm.get('google', 0)}")
    if llm.get('openai', 0) > 0:
        print(f"  OpenAI:               {llm.get('openai', 0)}")
    if llm.get('anthropic', 0) > 0:
        print(f"  Anthropic:            {llm.get('anthropic', 0)}")
    print(f"  Retries:              {llm.get('retries', 0)}")
    print(f"  Failures:             {llm.get('failures', 0)}")
    
    # Token statistics
    tokens = metrics.get('tokens', {})
    if tokens.get('input_total', 0) > 0:
        print("\n📝 TOKEN USAGE:")
        print(f"  Input Tokens:         {tokens.get('input_total', 0):,}")
        print(f"  Output Tokens:        {tokens.get('output_total', 0):,}")
        print(f"  Total Tokens:         {(tokens.get('input_total', 0) + tokens.get('output_total', 0)):,}")
        print(f"  Estimated Cost:       ${tokens.get('estimated_cost_usd', 0):.4f}")
    
    # Cache statistics
    cache = metrics.get('cache', {})
    print("\n💾 CACHE:")
    print(f"  Hits:                 {cache.get('hits', 0)}")
    print(f"  Misses:               {cache.get('misses', 0)}")
    print(f"  Hit Rate:             {cache.get('hit_rate', 0):.1f}%")
    
    # Retry statistics
    retry = metrics.get('retry', {})
    if retry.get('total_retries', 0) > 0:
        print("\n🔄 RETRY:")
        print(f"  Total Retries:        {retry.get('total_retries', 0)}")
        print(f"  Successful:           {retry.get('successful_retries', 0)}")
        print(f"  Success Rate:         {retry.get('retry_success_rate', 0):.1f}%")
    
    # Resource usage
    resource = metrics.get('resource', {})
    print("\n🖥️  RESOURCES:")
    print(f"  Current Memory:       {resource.get('current_memory_mb', 0):.2f} MB")
    print(f"  Peak Memory:          {resource.get('peak_memory_mb', 0):.2f} MB")
    print(f"  Avg CPU:              {resource.get('avg_cpu_percent', 0):.1f}%")
    
    # Uptime
    print(f"\n⏰ UPTIME:              {metrics.get('uptime_seconds', 0):.2f} seconds")
    print("═"*70)


async def test_execute_pandas():
    """Test 2: Execute pandas operations on spreadsheet."""
    print("\n\n" + "="*60)
    print("🐼 TEST 2: Execute Pandas Operations")
    print("="*60 + "\n")
    
    # Put your test file in: backend/tests/test_data/
    test_file = "tests/test_data/sales_data.csv"
    
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
    test_file = "tests/test_data/sales_data.csv"
    
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
    
    test_file = "tests/test_data/sales_data.csv"
    
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
    print("🚀 SPREADSHEET AGENT MANUAL TESTS WITH METRICS")
    print("🟢"*30 + "\n")
    
    # Run tests
    await test_analyze_spreadsheet()
    await test_execute_pandas()
    await test_get_info()
    await test_data_transformation()
    
    # Final session summary
    print("\n\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    print("\n📊 FINAL SESSION METRICS:")
    _display_session_metrics(query_agent_instance.get_metrics())
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

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
from agents.spreadsheet_agent.utils import load_dataframe
from agents.spreadsheet_agent.session import store_dataframe, get_dataframe

# Create global query agent instance
query_agent_instance = SpreadsheetQueryAgent()


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
    print(f"  Cerebras:             {llm.get('cerebras', 0)}")
    print(f"  Groq:                 {llm.get('groq', 0)}")
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


async def test_compare_two_files():
    """Test: Compare data from two spreadsheet files."""
    print("\n\n" + "="*70)
    print("🔄 TEST: Compare Two Spreadsheet Files")
    print("="*70 + "\n")
    
    # Define two test files
    file1 = "tests/test_data/sales_data.csv"
    file2 = "tests/test_data/Todo.xlsx"  # or any other file
    
    # Check if both files exist
    if not Path(file1).exists():
        print(f"⚠️  First test file not found: {file1}")
        print("📁 Please add test files to: backend/tests/test_data/")
        return
    
    if not Path(file2).exists():
        print(f"⚠️  Second test file not found: {file2}")
        print("📁 Using same file for comparison demo")
        file2 = file1  # Use same file for demo
    
    try:
        # Load both spreadsheets
        df1 = load_dataframe(file1)
        df2 = load_dataframe(file2)
        
        # Create separate thread IDs for each file
        thread_id_1 = "test-file-1"
        thread_id_2 = "test-file-2"
        
        # Store both in session
        store_dataframe(thread_id_1, df1, file1)
        store_dataframe(thread_id_2, df2, file2)
        
        print(f"📄 File 1: {Path(file1).name}")
        print(f"   Shape: {df1.shape}")
        print(f"   Columns: {list(df1.columns)[:5]}...")  # Show first 5 columns
        
        print(f"\n📄 File 2: {Path(file2).name}")
        print(f"   Shape: {df2.shape}")
        print(f"   Columns: {list(df2.columns)[:5]}...")
        
        # Query both files
        print("\n" + "─"*70)
        print("Query File 1:")
        query1 = "What are the key statistics or summary of this data?"
        result1 = await query_agent_instance.query(df1, query1, thread_id=thread_id_1, max_iterations=3)
        print(f"✅ File 1 Result: {result1.answer if hasattr(result1, 'answer') else result1}")
        
        if hasattr(result1, 'execution_metrics') and result1.execution_metrics:
            _display_metrics(result1.execution_metrics, "File 1 Query")
        
        print("\n" + "─"*70)
        print("Query File 2:")
        query2 = "What are the main insights from this spreadsheet?"
        result2 = await query_agent_instance.query(df2, query2, thread_id=thread_id_2, max_iterations=3)
        print(f"✅ File 2 Result: {result2.answer if hasattr(result2, 'answer') else result2}")
        
        if hasattr(result2, 'execution_metrics') and result2.execution_metrics:
            _display_metrics(result2.execution_metrics, "File 2 Query")
        
        # Combined comparison query (using merged data)
        print("\n" + "─"*70)
        print("Combined Analysis:")
        
        # For demo purposes, we'll query file1 with context about both files
        combined_query = f"""I have two datasets:
Dataset 1 ({Path(file1).name}): {df1.shape[0]} rows, {df1.shape[1]} columns
Dataset 2 ({Path(file2).name}): {df2.shape[0]} rows, {df2.shape[1]} columns

Compare the structure and size of these datasets. What are the key differences?"""
        
        result_combined = await query_agent_instance.query(
            df1, 
            combined_query, 
            thread_id=thread_id_1, 
            max_iterations=3
        )
        print(f"✅ Combined Analysis: {result_combined.answer if hasattr(result_combined, 'answer') else result_combined}")
        
        if hasattr(result_combined, 'execution_metrics') and result_combined.execution_metrics:
            _display_metrics(result_combined.execution_metrics, "Combined Query")
        
        # Display overall session metrics
        _display_session_metrics(query_agent_instance.get_metrics())
        
        print("\n✨ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_merge_analysis():
    """Test: Analyze merged data from two files."""
    print("\n\n" + "="*70)
    print("🔗 TEST: Merge and Analyze Two Files")
    print("="*70 + "\n")
    
    file1 = "tests/test_data/sales_data.csv"
    file2 = "tests/test_data/sales_data.csv"  # Use same file for demo
    
    if not Path(file1).exists():
        print(f"⚠️  Test file not found: {file1}")
        return
    
    try:
        df1 = load_dataframe(file1)
        df2 = load_dataframe(file2)
        
        # Merge the dataframes (simple concat for demo)
        df_merged = pd.concat([df1, df2], ignore_index=True)
        thread_id = "test-merged"
        
        store_dataframe(thread_id, df_merged, "merged_data")
        
        print(f"📊 Merged Data Shape: {df_merged.shape}")
        print(f"   Original File 1: {df1.shape[0]} rows")
        print(f"   Original File 2: {df2.shape[0]} rows")
        print(f"   Merged Total: {df_merged.shape[0]} rows")
        
        query = "What insights can you provide from this merged dataset? Look for patterns or trends."
        result = await query_agent_instance.query(df_merged, query, thread_id=thread_id, max_iterations=3)
        
        print(f"\n✅ Merged Analysis Result:")
        print(f"   {result.answer if hasattr(result, 'answer') else result}")
        
        if hasattr(result, 'execution_metrics') and result.execution_metrics:
            _display_metrics(result.execution_metrics, "Merged Data Query")
        
        _display_session_metrics(query_agent_instance.get_metrics())
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all multi-file tests."""
    print("\n" + "🚀"*35)
    print("MULTI-FILE SPREADSHEET TESTS")
    print("🚀"*35)
    
    await test_compare_two_files()
    await test_merge_analysis()
    
    print("\n\n" + "🎉"*35)
    print("ALL TESTS COMPLETED")
    print("🎉"*35 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

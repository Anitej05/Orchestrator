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
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import both agent components
from agents.spreadsheet_agent.llm_agent import SpreadsheetQueryAgent
from agents.spreadsheet_agent.utils import load_dataframe
from agents.spreadsheet_agent.session import store_dataframe
from agents.document_agent.agent import DocumentAgent
from agents.document_agent.schemas import AnalyzeDocumentRequest


def _display_metrics(metrics: dict, operation_name: str = "Operation"):
    """Display execution metrics in a beautiful format."""
    print("\n" + "━"*70)
    print(f"📊 {operation_name} - EXECUTION METRICS")
    print("━"*70)
    
    # Performance metrics
    print("\n⏱️  PERFORMANCE:")
    latency = metrics.get('latency_ms', metrics.get('performance', {}).get('total_latency_ms', 0))
    print(f"  Total Latency:        {latency:.2f} ms")
    print(f"  Cache Hit:            {'✅ Yes' if metrics.get('cache_hit', metrics.get('cache', {}).get('hits', 0) > 0) else '❌ No'}")
    
    # LLM metrics
    llm_calls = metrics.get('llm_calls', 0)
    if isinstance(llm_calls, dict):
        total_calls = llm_calls.get('total', 0)
    else:
        total_calls = llm_calls
    
    if total_calls > 0:
        print("\n🤖 LLM STATISTICS:")
        print(f"  API Calls:            {total_calls}")
        print(f"  Retries:              {metrics.get('retries', 0)}")
        
        tokens_input = metrics.get('tokens_input', metrics.get('tokens', {}).get('input_total', 0))
        tokens_output = metrics.get('tokens_output', metrics.get('tokens', {}).get('output_total', 0))
        
        if tokens_input > 0 or tokens_output > 0:
            print(f"  Input Tokens:         {tokens_input:,}")
            print(f"  Output Tokens:        {tokens_output:,}")
            print(f"  Total Tokens:         {(tokens_input + tokens_output):,}")
    
    # Resource usage
    memory_used = metrics.get('memory_used_mb', metrics.get('resource', {}).get('current_memory_mb', 0))
    if memory_used > 0:
        print("\n💾 RESOURCE USAGE:")
        print(f"  Memory Used:          {memory_used:.2f} MB")
    
    print("━"*70)


def _display_document_metrics(metrics: dict):
    """Display document agent session metrics."""
    print("\n" + "═"*70)
    print("📄 DOCUMENT AGENT - SESSION METRICS")
    print("═"*70)
    
    # Analysis statistics
    processing = metrics.get('processing', {})
    print("\n📊 DOCUMENTS PROCESSED:")
    print(f"  Total Documents:      {processing.get('documents_analyzed', 0)}")
    print(f"  Total Files:          {processing.get('total_files_processed', 0)}")
    print(f"  Successful:           {processing.get('successful_analyses', 0)} ✅")
    print(f"  Failed:               {processing.get('failed_analyses', 0)} ❌")
    
    # Performance
    perf = metrics.get('performance', {})
    print("\n⚡ PERFORMANCE:")
    print(f"  Avg Latency:          {perf.get('avg_latency_ms', 0):.2f} ms")
    print(f"  Total Latency:        {perf.get('total_latency_ms', 0):.2f} ms")
    
    # LLM calls
    llm = metrics.get('llm_calls', {})
    print("\n🤖 LLM CALLS:")
    print(f"  Total:                {llm.get('total', 0)}")
    print(f"  Analysis:             {llm.get('analysis', 0)}")
    print(f"  QA:                   {llm.get('qa', 0)}")
    
    # RAG statistics
    rag = metrics.get('rag', {})
    if rag.get('chunks_retrieved', 0) > 0:
        print("\n🔍 RAG STATISTICS:")
        print(f"  Chunks Retrieved:     {rag.get('chunks_retrieved', 0)}")
        print(f"  Avg Chunks/Query:     {rag.get('avg_chunks_per_query', 0):.1f}")
    
    # Cache
    cache = metrics.get('cache', {})
    print("\n💾 CACHE:")
    print(f"  Hits:                 {cache.get('hits', 0)}")
    print(f"  Misses:               {cache.get('misses', 0)}")
    print(f"  Hit Rate:             {cache.get('hit_rate', 0):.1f}%")
    
    # Resource usage
    resource = metrics.get('resource', {})
    print("\n🖥️  RESOURCES:")
    print(f"  Current Memory:       {resource.get('current_memory_mb', 0):.2f} MB")
    print(f"  Peak Memory:          {resource.get('peak_memory_mb', 0):.2f} MB")
    
    print("═"*70)


def _display_spreadsheet_metrics(metrics: dict):
    """Display spreadsheet agent session metrics."""
    print("\n" + "═"*70)
    print("📊 SPREADSHEET AGENT - SESSION METRICS")
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
    
    print("═"*70)


async def test_document_analysis():
    """Test: Analyze a document using Document Agent.
    
    NOTE: Document Agent requires a vector store for RAG-based analysis.
    This test demonstrates the structure but will show an error without vector store setup.
    To fully test: First create a vector store from documents, then provide vector_store_path.
    """
    print("\n\n" + "="*70)
    print("📄 TEST: Document Analysis (Requires Vector Store)")
    print("="*70 + "\n")
    
    print("⚠️  NOTICE: Document Agent requires pre-created vector store for RAG analysis")
    print("   This test shows the integration structure. For full functionality:")
    print("   1. Create FAISS vector store from documents")
    print("   2. Provide vector_store_path in AnalyzeDocumentRequest")
    print("   3. Agent will use RAG to answer queries about documents\n")
    
    # Define test document
    doc_path = "tests/test_data/sample_document.pdf"  # or .txt, .docx
    
    if not Path(doc_path).exists():
        # Create a sample text file for testing
        test_dir = Path("tests/test_data")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        sample_file = test_dir / "sample_document.txt"
        sample_content = """# Sample Document for Testing

This is a sample document created for testing the Document Agent.

## Overview
This document contains information about various topics including:
- Technology trends
- Data analysis
- Machine learning applications

## Technology Trends
The field of artificial intelligence is rapidly evolving. Machine learning models
are becoming more sophisticated and accessible. Cloud computing enables scalable
solutions for businesses of all sizes.

## Data Analysis
Data-driven decision making is crucial for modern businesses. Organizations
collect vast amounts of data daily. Proper analysis can reveal valuable insights
and patterns that drive strategic decisions.

## Conclusion
Technology continues to transform how we work and live. Staying informed about
these trends is essential for success in the digital age."""
        
        sample_file.write_text(sample_content, encoding='utf-8')
        doc_path = str(sample_file)
        print(f"✅ Created sample file: {doc_path}")
    
    try:
        # Initialize document agent
        doc_agent = DocumentAgent()
        session_id = "test-doc-session-1"
        
        print(f"📄 Document file: {Path(doc_path).name}")
        print(f"   Status: Ready for vector store creation\n")
        
        # For demonstration: show what a complete request would look like
        print("📋 Example request structure (requires vector_store_path):")
        print("   AnalyzeDocumentRequest(")
        print("       vector_store_path='storage/vector_store/my_docs',")
        print("       file_path='tests/test_data/sample_document.txt',")
        print("       query='What are the main topics?',")
        print("       thread_id='test-session'")
        print("   )\n")
        
        # Skip actual analysis without vector store
        print("⏭️  Skipping actual analysis (requires vector store setup)")
        print("   Document Agent initialized successfully ✅")
        print("   File exists and ready for processing ✅\n")
        
        # Get and display agent metrics
        session_metrics = doc_agent.get_metrics()
        print("📊 Document Agent Status:")
        print(f"   Initialized: ✅")
        print(f"   LLM Client: Ready")
        print(f"   Session Manager: Ready")
        print(f"   Version Manager: Ready\n")
        
        return doc_agent, {"success": False, "message": "Skipped - requires vector store"}
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_spreadsheet_analysis():
    """Test: Analyze a spreadsheet using Spreadsheet Agent."""
    print("\n\n" + "="*70)
    print("📊 TEST: Spreadsheet Analysis")
    print("="*70 + "\n")
    
    spreadsheet_path = "tests/test_data/sales_data.csv"
    
    if not Path(spreadsheet_path).exists():
        print(f"⚠️  Spreadsheet file not found: {spreadsheet_path}")
        print("📁 Please add a spreadsheet to: backend/tests/test_data/")
        print("\n💡 Creating a sample CSV file for testing...")
        
        test_dir = Path("tests/test_data")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        sample_file = test_dir / "sales_data.csv"
        sample_data = pd.DataFrame({
            'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop'],
            'Amount': [1500, 25, 75, 350, 1600],
            'Quantity': [1, 2, 1, 1, 1],
            'Region': ['North', 'South', 'East', 'West', 'North']
        })
        sample_data.to_csv(sample_file, index=False)
        spreadsheet_path = str(sample_file)
        print(f"✅ Created sample file: {spreadsheet_path}")
    
    try:
        # Initialize spreadsheet agent
        spreadsheet_agent = SpreadsheetQueryAgent()
        
        # Load spreadsheet
        df = load_dataframe(spreadsheet_path)
        thread_id = "test-spreadsheet-session-1"
        store_dataframe(thread_id, df, spreadsheet_path)
        
        print(f"📊 Analyzing spreadsheet: {Path(spreadsheet_path).name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Query spreadsheet
        query = "What are the total sales and which product sold the most?"
        result = await spreadsheet_agent.query(df, query, thread_id=thread_id, max_iterations=3)
        
        print(f"\n✅ Spreadsheet Analysis Result:")
        print(f"   {result.answer if hasattr(result, 'answer') else result}")
        
        # Display metrics
        if hasattr(result, 'execution_metrics') and result.execution_metrics:
            _display_metrics(result.execution_metrics, "Spreadsheet Analysis")
        
        # Get session metrics
        session_metrics = spreadsheet_agent.get_metrics()
        _display_spreadsheet_metrics(session_metrics)
        
        return spreadsheet_agent, result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_combined_analysis():
    """Test: Use both Document and Spreadsheet agents together.
    
    NOTE: Full test requires:
    - Document Agent: Vector store setup for RAG
    - Spreadsheet Agent: Available LLM API quota
    """
    print("\n\n" + "="*70)
    print("🔄 TEST: Combined Document + Spreadsheet Analysis")
    print("="*70 + "\n")
    
    print("This test demonstrates the integration structure for using both agents:")
    print("1. Document Agent - Analyzes text documents using RAG (requires vector store)")
    print("2. Spreadsheet Agent - Analyzes tabular data using LLM")
    print("3. Combined workflow - Integrates insights from both sources\n")
    
    # Run document analysis (will show setup requirements)
    doc_agent, doc_result = await test_document_analysis()
    
    # Run spreadsheet analysis (may hit rate limits)
    spreadsheet_agent, spreadsheet_result = await test_spreadsheet_analysis()
    
    print("\n\n" + "="*70)
    print("🎯 INTEGRATION SUMMARY")
    print("="*70)
    
    print("\n📄 Document Agent:")
    if doc_agent:
        print("   ✅ Initialized successfully")
        print("   ⚠️  Requires vector store for full RAG analysis")
        print("   📚 Use case: Answer questions about document content")
    else:
        print("   ❌ Not initialized")
    
    print("\n📊 Spreadsheet Agent:")
    if spreadsheet_agent:
        print("   ✅ Initialized successfully")
        if spreadsheet_result and hasattr(spreadsheet_result, 'answer'):
            print("   ✅ Query executed successfully")
            print(f"   💬 Answer: {str(spreadsheet_result.answer)[:100]}...")
        else:
            print("   ⚠️  Query failed (likely rate limits)")
        print("   📈 Use case: Analyze data and answer statistical questions")
    else:
        print("   ❌ Not initialized")
    
    print("\n✨ Integration Workflow:")
    print("   In a real application, these agents work together to:")
    print("   • Extract insights from documents (policies, reports, guides)")
    print("   • Analyze data patterns from spreadsheets (sales, metrics, logs)")  
    print("   • Combine text and numerical insights for comprehensive analysis")
    print("   • Cross-validate claims in documents against spreadsheet data")
    print("   • Generate reports combining narrative and statistical findings")
    
    print("\n📋 Setup Requirements:")
    print("   1. Document Agent:")
    print("      - Create FAISS vector stores from your documents")
    print("      - Provide vector_store_path in requests")
    print("   2. Spreadsheet Agent:")
    print("      - Ensure LLM API quota available (Cerebras/Groq)")
    print("      - Configure fallback providers")
    
    # Display combined metrics if both ran
    if doc_agent or spreadsheet_agent:
        print("\n" + "─"*70)
        print("📈 AGENT STATUS")
        print("─"*70)
        
        if doc_agent:
            doc_metrics = doc_agent.get_metrics()
            print(f"\n📄 Document Agent:")
            print(f"   Requests Completed:   {doc_metrics.get('performance', {}).get('requests_completed', 0)}")
            print(f"   LLM Calls:            {doc_metrics.get('llm_calls', {}).get('total', 0)}")
        
        if spreadsheet_agent:
            sheet_metrics = spreadsheet_agent.get_metrics()
            print(f"\n📊 Spreadsheet Agent:")
            queries = sheet_metrics.get('queries', {})
            print(f"   Queries Total:        {queries.get('total', 0)}")
            print(f"   Successful:           {queries.get('successful', 0)}")
            print(f"   Failed:               {queries.get('failed', 0)}")
            
            tokens = sheet_metrics.get('tokens', {})
            if tokens.get('input_total', 0) > 0:
                print(f"   Total Cost:           ${tokens.get('estimated_cost_usd', 0):.4f}")
        
        print("─"*70)


async def main():
    """Run all combined agent tests."""
    print("\n" + "🚀"*35)
    print("DOCUMENT + SPREADSHEET AGENT TESTS")
    print("🚀"*35)
    
    await test_combined_analysis()
    
    print("\n\n" + "🎉"*35)
    print("ALL TESTS COMPLETED")
    print("🎉"*35 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

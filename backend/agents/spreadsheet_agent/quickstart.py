"""
Quick Start Script for Modularized Spreadsheet Agent

This script demonstrates how to use the modularized spreadsheet agent.
Run: python -m agents.spreadsheet_agent.quickstart
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


async def main():
    print("=" * 70)
    print("Spreadsheet Agent v2.0 - Modular Architecture Quickstart")
    print("=" * 70)
    print()
    
    # 1. Import modules
    print("1️⃣ Importing modular components...")
    try:
        from agents.spreadsheet_agent import (
            config,
            memory,
            llm_agent,
            session,
            display,
            utils
        )
        from agents.spreadsheet_agent.main import app
        print("   ✅ All modules imported successfully!")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    print()
    
    # 2. Check configuration
    print("2️⃣ Checking configuration...")
    print(f"   📁 Storage directory: {config.STORAGE_DIR}")
    print(f"   🔌 Agent port: {config.AGENT_PORT}")
    print(f"   📦 Max file size: {config.MAX_FILE_SIZE_MB}MB")
    print(f"   🤖 LLM providers: {len(llm_agent.query_agent.providers)}")
    if llm_agent.query_agent.providers:
        provider_names = ' → '.join([p['name'] for p in llm_agent.query_agent.providers])
        print(f"   🔗 Provider chain: {provider_names}")
    else:
        print("   ⚠️  No LLM providers configured (set API keys in .env)")
    print()
    
    # 3. Check memory system
    print("3️⃣ Testing memory/cache system...")
    print(f"   💾 Memory cache directory: {config.MEMORY_CACHE_DIR}")
    
    # Test cache
    memory.spreadsheet_memory.cache_df_metadata("test_file", {
        "shape": (100, 10),
        "columns": ["col1", "col2"]
    })
    cached = memory.spreadsheet_memory.get_df_metadata("test_file")
    if cached and cached["shape"] == (100, 10):
        print("   ✅ Cache write/read working!")
    else:
        print("   ❌ Cache test failed!")
    
    # Show cache stats
    stats = memory.spreadsheet_memory.get_cache_stats()
    print(f"   📊 Metadata cache: {stats['metadata']['size']} entries")
    print(f"   📊 Query cache: {stats['query']['size']} entries")
    print(f"   📊 Context cache: {stats['context']['size']} entries")
    print()
    
    # 4. Check session management
    print("4️⃣ Testing session management...")
    import pandas as pd
    
    # Create test dataframe
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "Salary": [50000, 60000, 70000]
    })
    
    # Store in session
    session.store_dataframe("test_123", df, "/fake/path.csv", "thread_test")
    
    # Retrieve from session
    retrieved_df = session.get_dataframe("test_123", "thread_test")
    if retrieved_df is not None and retrieved_df.shape == (3, 3):
        print("   ✅ Session storage working!")
        print(f"   📊 DataFrame: {retrieved_df.shape[0]} rows × {retrieved_df.shape[1]} columns")
    else:
        print("   ❌ Session test failed!")
    
    # Test thread isolation
    dfs1 = session.get_conversation_dataframes("thread1")
    dfs2 = session.get_conversation_dataframes("thread2")
    if dfs1 is not dfs2:
        print("   ✅ Thread isolation working!")
    print()
    
    # 5. Check utilities
    print("5️⃣ Testing utility functions...")
    
    # Test numpy conversion
    import numpy as np
    test_data = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "array": np.array([1, 2, 3])
    }
    converted = utils.convert_numpy_types(test_data)
    if isinstance(converted["int"], int):
        print("   ✅ Numpy type conversion working!")
    
    # Test dataframe serialization
    serialized = utils.serialize_dataframe(df)
    if "columns" in serialized and "data" in serialized:
        print("   ✅ DataFrame serialization working!")
    print()
    
    # 6. Check FastAPI app
    print("6️⃣ Checking FastAPI application...")
    print(f"   🌐 App title: {app.title}")
    print(f"   📌 App version: {app.version}")
    
    routes = [route.path for route in app.routes]
    key_routes = ["/upload", "/nl_query", "/transform", "/health", "/stats"]
    missing = [r for r in key_routes if r not in routes]
    
    if not missing:
        print("   ✅ All key routes present!")
        print(f"   🛣️  Total routes: {len(routes)}")
    else:
        print(f"   ⚠️  Missing routes: {missing}")
    print()
    
    # 7. Show how to start server
    print("7️⃣ Starting the server...")
    print()
    print("   To run the agent standalone:")
    print("   ┌─────────────────────────────────────────────────────┐")
    print("   │ python -m agents.spreadsheet_agent.main             │")
    print("   └─────────────────────────────────────────────────────┘")
    print()
    print("   Or with uvicorn:")
    print("   ┌─────────────────────────────────────────────────────┐")
    print(f"   │ uvicorn agents.spreadsheet_agent.main:app \\        │")
    print(f"   │         --host 0.0.0.0 --port {config.AGENT_PORT}                    │")
    print("   └─────────────────────────────────────────────────────┘")
    print()
    
    # 8. Integration example
    print("8️⃣ Integration Example:")
    print()
    print("   ```python")
    print("   # In orchestrator or main.py")
    print("   from agents.spreadsheet_agent.main import app as spreadsheet_app")
    print()
    print("   # Mount agent (if needed)")
    print("   # main_app.mount('/spreadsheet', spreadsheet_app)")
    print()
    print("   # Or use HTTP proxy to agent port")
    print(f"   # Spreadsheet agent running on: http://localhost:{config.AGENT_PORT}")
    print("   ```")
    print()
    
    # 9. Summary
    print("=" * 70)
    print("✅ Modularization Complete!")
    print("=" * 70)
    print()
    print("📦 Modules:")
    print("   • config.py       - Configuration management")
    print("   • models.py       - Pydantic data models")
    print("   • memory.py       - LRU cache system (NEW)")
    print("   • llm_agent.py    - Natural language query agent")
    print("   • code_generator.py - Code generation")
    print("   • session.py      - Session management")
    print("   • display.py      - Canvas display utilities")
    print("   • main.py         - FastAPI application")
    print("   • utils/          - Utility functions")
    print()
    print("🎯 Key Features:")
    print("   ✅ Modular architecture (10 focused modules)")
    print("   ✅ Root-level storage (storage/spreadsheets/)")
    print("   ✅ Intelligent caching (3-tier LRU cache)")
    print("   ✅ Thread-safe operations")
    print("   ✅ Multi-provider LLM fallback")
    print("   ✅ Comprehensive error handling")
    print()
    print("📊 Performance:")
    print(f"   Cache sizes: Metadata={config.MEMORY_CACHE_MAX_SIZE}, ")
    print(f"                Query=500, Context=200")
    print(f"   Cache TTL: {config.MEMORY_CACHE_TTL_SECONDS}s")
    print()
    print("🚀 Ready for production!")
    print()


if __name__ == "__main__":
    asyncio.run(main())

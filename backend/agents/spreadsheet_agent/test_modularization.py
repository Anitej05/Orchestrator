#!/usr/bin/env python3
"""
Test script for modularized Spreadsheet Agent
Run this to verify all functionality works correctly
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

print(f"Backend dir: {backend_dir}")
print(f"Python path: {sys.path[0]}")

def test_imports():
    """Test all module imports"""
    print("🧪 Testing imports...")
    
    try:
        from agents.spreadsheet_agent import (
            config, models, memory, session, display,
            llm_agent, code_generator
        )
        from agents.spreadsheet_agent.utils import core_utils, data_utils
        from agents.spreadsheet_agent.main import app
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration module"""
    print("\n🧪 Testing config...")
    
    try:
        from agents.spreadsheet_agent.config import (
            STORAGE_DIR, AGENT_PORT, MAX_FILE_SIZE_MB,
            CEREBRAS_API_KEY, GROQ_API_KEY
        )
        
        assert STORAGE_DIR.exists(), "Storage directory doesn't exist"
        assert AGENT_PORT > 0, "Invalid port number"
        assert MAX_FILE_SIZE_MB > 0, "Invalid file size limit"
        
        print(f"  📁 Storage: {STORAGE_DIR}")
        print(f"  🔌 Port: {AGENT_PORT}")
        print(f"  📊 Max file size: {MAX_FILE_SIZE_MB}MB")
        print(f"  🤖 Cerebras API: {'✓' if CEREBRAS_API_KEY else '✗'}")
        print(f"  🤖 Groq API: {'✓' if GROQ_API_KEY else '✗'}")
        print("✅ Config module OK")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_models():
    """Test Pydantic models"""
    print("\n🧪 Testing models...")
    
    try:
        from agents.spreadsheet_agent.models import (
            ApiResponse, CreateSpreadsheetRequest,
            NaturalLanguageQueryRequest
        )
        
        # Test ApiResponse
        response = ApiResponse(success=True, result={"test": "data"})
        assert response.success == True
        
        # Test CreateSpreadsheetRequest
        req = CreateSpreadsheetRequest(
            instruction="Create a test spreadsheet",
            output_filename="test.csv"
        )
        assert req.instruction == "Create a test spreadsheet"
        
        print("✅ Models module OK")
        return True
    except Exception as e:
        print(f"❌ Models test failed: {e}")
        return False


def test_memory():
    """Test memory/cache module"""
    print("\n🧪 Testing memory...")
    
    try:
        from agents.spreadsheet_agent.memory import spreadsheet_memory
        
        # Test cache operations
        spreadsheet_memory.cache_df_metadata("test_file", {"rows": 100, "cols": 5})
        
        # Test cache stats
        stats = spreadsheet_memory.get_cache_stats()
        assert "metadata_cache_size" in stats
        assert "query_cache_size" in stats
        assert "context_cache_size" in stats
        
        print(f"  📊 Cache stats: Metadata={stats['metadata_cache_size']}, Queries={stats['query_cache_size']}")
        print("✅ Memory module OK")
        return True
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility modules"""
    print("\n🧪 Testing utils...")
    
    try:
        from agents.spreadsheet_agent.utils.core_utils import (
            serialize_dataframe, convert_numpy_types
        )
        from agents.spreadsheet_agent.utils.data_utils import (
            validate_file, normalize_column_names
        )
        import pandas as pd
        import numpy as np
        
        # Test serialization
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        serialized = serialize_dataframe(df)
        assert "data" in serialized
        assert "columns" in serialized
        
        # Test numpy conversion
        value = np.int64(42)
        converted = convert_numpy_types(value)
        assert isinstance(converted, int)
        
        # Test column normalization
        df_test = pd.DataFrame({"Name ": [1], " Age": [2], "Email Address": [3]})
        normalized_df = normalize_column_names(df_test)
        assert "name" in normalized_df.columns
        assert "age" in normalized_df.columns
        
        print("✅ Utils modules OK")
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session():
    """Test session management"""
    print("\n🧪 Testing session...")
    
    try:
        from agents.spreadsheet_agent.session import (
            store_dataframe, get_dataframe
        )
        import pandas as pd
        
        # Create test dataframe
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        
        # Store and retrieve
        store_dataframe("test_id", df, "/tmp/test.csv", "test_thread")
        retrieved = get_dataframe("test_id", "test_thread")
        
        assert retrieved is not None
        assert len(retrieved) == 2
        
        print("✅ Session module OK")
        return True
    except Exception as e:
        print(f"❌ Session test failed: {e}")
        return False


def test_display():
    """Test display utilities"""
    print("\n🧪 Testing display...")
    
    try:
        from agents.spreadsheet_agent.display import (
            dataframe_to_canvas, format_dataframe_preview
        )
        import pandas as pd
        
        # Create test dataframe
        df = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})
        
        # Test canvas generation
        canvas = dataframe_to_canvas(
            df=df,
            title="Test",
            filename="test.csv",
            file_id="test_123"
        )
        
        # Canvas returns a dict with canvas_type and canvas_data
        assert isinstance(canvas, dict)
        assert "canvas_type" in canvas or "canvas_data" in canvas
        
        print(f"  Canvas keys: {list(canvas.keys())}")
        
        # Test preview formatting
        preview = format_dataframe_preview(df, max_rows=10)
        assert "data" in preview
        
        print("✅ Display module OK")
        return True
    except Exception as e:
        print(f"❌ Display test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app():
    """Test FastAPI application"""
    print("\n🧪 Testing FastAPI app...")
    
    try:
        from agents.spreadsheet_agent.main import app
        
        # Check routes
        routes = [r for r in app.routes if hasattr(r, 'path')]
        route_paths = [r.path for r in routes]
        
        required_routes = [
            "/upload",
            "/nl_query",
            "/transform",
            "/get_summary",
            "/get_summary_with_canvas",
            "/query",
            "/get_column_stats",
            "/display",
            "/download/{file_id}",
            "/execute_pandas",
            "/create",
            "/files",
            "/files/{file_id}",
            "/cleanup",
            "/health",
            "/stats"
        ]
        
        missing = []
        for route in required_routes:
            if route not in route_paths:
                missing.append(route)
        
        if missing:
            print(f"❌ Missing routes: {missing}")
            return False
        
        print(f"  🛣️  Total routes: {len(route_paths)}")
        print(f"  ✅ All {len(required_routes)} required routes present")
        print("✅ FastAPI app OK")
        return True
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 SPREADSHEET AGENT - MODULARIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_models,
        test_memory,
        test_utils,
        test_session,
        test_display,
        test_fastapi_app
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Modularization complete and verified.")
        print("\n📝 Next steps:")
        print("  1. Start the agent: python -m agents.spreadsheet_agent.main")
        print("  2. Test endpoints manually")
        print("  3. Run integration tests")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

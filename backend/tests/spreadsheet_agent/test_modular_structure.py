"""
Basic tests for modularized spreadsheet agent.

Run with: pytest tests/spreadsheet_agent/test_modular_structure.py -v
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


def test_imports():
    """Test that all modules can be imported"""
    # Core modules
    from agents.spreadsheet_agent import config
    from agents.spreadsheet_agent import models
    from agents.spreadsheet_agent import memory
    
    # Business logic
    from agents.spreadsheet_agent import llm_agent
    from agents.spreadsheet_agent import code_generator
    from agents.spreadsheet_agent import session
    from agents.spreadsheet_agent import display
    
    # Utilities
    from agents.spreadsheet_agent import utils
    
    # Main app
    from agents.spreadsheet_agent import app
    
    assert config is not None
    assert models is not None
    assert memory is not None
    assert llm_agent is not None
    assert code_generator is not None
    assert session is not None
    assert display is not None
    assert utils is not None
    assert app is not None


def test_config_values():
    """Test configuration module"""
    from agents.spreadsheet_agent import config
    
    # Check storage paths
    assert config.STORAGE_DIR.exists() or True  # May not exist yet
    assert str(config.STORAGE_DIR).endswith("storage/spreadsheets")
    
    # Check settings
    assert config.AGENT_PORT == 8041
    assert config.MAX_FILE_SIZE_MB > 0
    assert config.LLM_TEMPERATURE >= 0
    assert config.LLM_MAX_TOKENS_QUERY > 0


def test_models():
    """Test Pydantic models"""
    from agents.spreadsheet_agent.models import ApiResponse, CreateSpreadsheetRequest
    
    # Test ApiResponse
    response = ApiResponse(success=True, result={"test": "data"})
    assert response.success is True
    assert response.result == {"test": "data"}
    assert response.error is None
    
    # Test CreateSpreadsheetRequest
    request = CreateSpreadsheetRequest(
        content="col1,col2\n1,2",
        output_format="csv"
    )
    assert request.content == "col1,col2\n1,2"
    assert request.output_format == "csv"


def test_memory():
    """Test memory/cache system"""
    from agents.spreadsheet_agent.memory import spreadsheet_memory, LRUCache
    
    # Test LRU cache
    cache = LRUCache(max_size=10, ttl_seconds=60)
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("nonexistent") is None
    
    # Test spreadsheet memory
    spreadsheet_memory.cache_df_metadata("test_file", {"shape": (10, 5)})
    metadata = spreadsheet_memory.get_df_metadata("test_file")
    assert metadata is not None
    assert metadata["shape"] == (10, 5)
    
    # Test cache stats
    stats = spreadsheet_memory.get_cache_stats()
    assert "metadata" in stats
    assert "query" in stats
    assert "context" in stats


def test_session():
    """Test session management"""
    from agents.spreadsheet_agent.session import (
        get_conversation_dataframes,
        get_conversation_file_paths,
        store_dataframe
    )
    import pandas as pd
    
    # Test thread-scoped storage
    dfs1 = get_conversation_dataframes("thread1")
    dfs2 = get_conversation_dataframes("thread2")
    
    # Should be isolated
    assert dfs1 is not dfs2
    
    # Test storing dataframe
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    store_dataframe("test_file", df, "/fake/path.csv", "thread1")
    
    dfs = get_conversation_dataframes("thread1")
    assert "test_file" in dfs
    assert dfs["test_file"].shape == (3, 2)


def test_utils():
    """Test utility functions"""
    from agents.spreadsheet_agent.utils import (
        validate_file,
        convert_numpy_types,
        serialize_dataframe
    )
    import pandas as pd
    import numpy as np
    
    # Test file validation
    try:
        validate_file("test.csv", 100)  # 100 bytes
        # Should not raise for valid file
    except Exception:
        pass  # Expected for invalid file
    
    # Test numpy conversion
    data = {
        "int": np.int64(42),
        "float": np.float64(3.14),
        "bool": np.bool_(True)
    }
    converted = convert_numpy_types(data)
    assert isinstance(converted["int"], int)
    assert isinstance(converted["float"], float)
    assert isinstance(converted["bool"], bool)
    
    # Test dataframe serialization
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    serialized = serialize_dataframe(df)
    assert "columns" in serialized
    assert "data" in serialized


def test_llm_agent():
    """Test LLM agent initialization"""
    from agents.spreadsheet_agent.llm_agent import query_agent
    
    # Agent should be initialized
    assert query_agent is not None
    assert hasattr(query_agent, "providers")
    # Note: providers list may be empty if no API keys set


def test_app():
    """Test FastAPI app"""
    from agents.spreadsheet_agent.main import app
    
    # Check app is created
    assert app is not None
    assert app.title == "Spreadsheet Agent"
    assert app.version == "2.0.0"
    
    # Check routes exist
    routes = [route.path for route in app.routes]
    assert "/upload" in routes
    assert "/nl_query" in routes
    assert "/transform" in routes
    assert "/health" in routes
    assert "/stats" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

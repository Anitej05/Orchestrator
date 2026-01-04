# Spreadsheet Agent v2.0 - Modular Architecture

## Overview

The Spreadsheet Agent has been completely modularized from a single 2,100-line file into a clean, maintainable package structure. This refactoring improves testability, maintainability, and adds new features like intelligent caching.

## Architecture

### Core Modules

#### `config.py` (72 lines)
- **Purpose**: Centralized configuration management
- **Key Features**:
  - Root-level storage paths (`storage/spreadsheets/`)
  - LLM provider settings (Cerebras → Groq fallback)
  - Cache and memory settings
  - Agent operational limits
- **Storage Migration**: Uses root `storage/` directory instead of agent-specific subdirectory

#### `models.py` (97 lines)
- **Purpose**: Pydantic data models for requests/responses
- **Models**:
  - `ApiResponse` - Standard response wrapper with numpy serialization
  - Request models: `CreateSpreadsheetRequest`, `NaturalLanguageQueryRequest`
  - Response models: `SummaryResponse`, `QueryResponse`, `StatsResponse`
  - Operation tracking: `QueryPlan`, `QueryResult`

#### `memory.py` (257 lines) ⭐ NEW FEATURE
- **Purpose**: Intelligent caching system for performance
- **Components**:
  - `LRUCache` - Generic LRU cache with TTL and thread-safety
  - `SpreadsheetMemory` - Three-tier caching:
    - **Metadata cache**: 1000 entries, 1h TTL (dataframe info)
    - **Query cache**: 500 entries, 30min TTL (query results)
    - **Context cache**: 200 entries, 1h TTL (conversation context)
- **Features**:
  - Persistent disk storage
  - Cache statistics
  - Thread-safe operations
  - Automatic expiration

### Business Logic Modules

#### `llm_agent.py` (460 lines)
- **Purpose**: LLM-powered natural language query processing
- **Key Class**: `SpreadsheetQueryAgent`
- **Features**:
  - ReAct-style reasoning loop
  - Multi-provider fallback (Cerebras → Groq)
  - Safe pandas code execution
  - Query result caching
  - Context-aware querying
- **Methods**:
  - `query()` - Process natural language questions
  - `_safe_execute_pandas()` - Execute code safely
  - `_get_dataframe_context()` - Generate DF context for LLM

#### `code_generator.py` (175 lines)
- **Purpose**: Generate pandas code from natural language
- **Functions**:
  - `generate_modification_code()` - Generate transformation code
  - `generate_csv_from_instruction()` - Create CSV from description
- **Features**:
  - Pattern-based code templates
  - Multi-provider support
  - Markdown cleanup

#### `session.py` (140 lines)
- **Purpose**: Thread-safe session and dataframe management
- **Key Features**:
  - Thread-local storage for isolation
  - File loading with multiple fallbacks (cache → memory → file_manager)
  - Dataframe state tracking
- **Functions**:
  - `get_conversation_dataframes()` - Thread-scoped DF storage
  - `ensure_file_loaded()` - Smart file loading with fallbacks
  - `store_dataframe()` - Save DF with caching
  - `get_dataframe_state()` - DF metadata extraction

#### `display.py` (88 lines)
- **Purpose**: Canvas display formatting
- **Functions**:
  - `dataframe_to_canvas()` - Convert DF to canvas format
  - `format_dataframe_preview()` - Create preview dict
- **Features**:
  - Fallback to basic format if canvas utils unavailable
  - Row limiting for display

### Utilities

#### `utils/core_utils.py` (149 lines)
- **Purpose**: Error handling and serialization
- **Components**:
  - Custom exceptions (`SpreadsheetError`, `FileLoadError`, etc.)
  - `NumpyEncoder` - JSON encoder for numpy types
  - Error handling functions
- **Functions**:
  - `convert_numpy_types()` - Recursive type conversion
  - `serialize_dataframe()` - DF to JSON-compatible
  - `handle_execution_error()` - User-friendly error messages

#### `utils/data_utils.py` (198 lines)
- **Purpose**: Data validation and conversion
- **Functions**:
  - **Validation**: `validate_file()`, `validate_dataframe()`, `validate_column_names()`
  - **Loading**: `csv_to_dataframe()`, `excel_to_dataframe()`, `load_dataframe()`
  - **Export**: `dataframe_to_csv()`, `dataframe_to_excel()`, `dataframe_to_json()`
  - **Utilities**: `normalize_column_names()`, `is_valid_csv()`

### Main Application

#### `main.py` (500+ lines)
- **Purpose**: FastAPI application with all routes
- **Key Routes**:
  - `POST /upload` - Upload CSV/Excel files
  - `POST /nl_query` - Natural language queries
  - `POST /transform` - Data transformations
  - `GET /health` - Health check with stats
  - `GET /stats` - Agent statistics
- **Features**:
  - Startup/shutdown events
  - Memory cache persistence
  - Thread-safe operations
  - Standardized file management integration

## Installation

The modularized agent is a drop-in replacement. Update your imports:

```python
# OLD
from agents.spreadsheet_agent import app as spreadsheet_app

# NEW
from agents.spreadsheet_agent.main import app as spreadsheet_app
```

## Usage

### Start the Agent

```python
# Standalone
python -m agents.spreadsheet_agent.main

# Or with uvicorn
uvicorn agents.spreadsheet_agent.main:app --host 0.0.0.0 --port 8041
```

### Use Individual Modules

```python
from agents.spreadsheet_agent import (
    config,
    memory,
    llm_agent,
    session,
    display
)

# Use memory cache
memory.spreadsheet_memory.cache_df_metadata("file_123", {...})
cached = memory.spreadsheet_memory.get_df_metadata("file_123")

# Use LLM agent
result = await llm_agent.query_agent.query(df, "Show top 10 rows")

# Manage sessions
session.store_dataframe("file_123", df, "/path/to/file.csv", "thread_456")
df = session.get_dataframe("file_123", "thread_456")
```

## Benefits

### 1. Maintainability
- **Before**: Single 2,100-line file
- **After**: 10 focused modules (72-460 lines each)
- Each module has a single, clear responsibility

### 2. Testability
- Modules can be tested independently
- Easier to mock dependencies
- Clear interfaces between components

### 3. Performance ⭐
- **NEW**: Intelligent caching system
  - Avoid recomputing dataframe metadata
  - Cache query results for repeated questions
  - Maintain conversation context efficiently
- **Memory footprint**: LRU cache with automatic eviction
- **Cache hit rates**: Track with `memory.spreadsheet_memory.get_cache_stats()`

### 4. Debuggability
- Smaller files easier to navigate
- Clear module boundaries
- Structured logging per module

### 5. Extensibility
- Easy to add new transformations
- Simple to integrate new LLM providers
- Clear extension points

## Migration Notes

### Storage Paths
- **Old**: `backend/agents/storage/spreadsheets/`
- **New**: `storage/spreadsheets/` (repository root)

### Imports
All imports updated to use modular structure. No changes needed in other agents.

### Session Management
Thread-scoped storage ensures conversation isolation. Use `thread_id` parameter for multi-user scenarios.

### Memory Cache
- Automatically loaded on startup from `storage/spreadsheet_memory/cache.json`
- Automatically saved on shutdown
- Manual control: `spreadsheet_memory.save_to_disk()` / `load_from_disk()`

## Testing

```bash
# Run all tests
pytest tests/spreadsheet_agent/

# Test individual modules
pytest tests/spreadsheet_agent/test_memory.py
pytest tests/spreadsheet_agent/test_llm_agent.py
pytest tests/spreadsheet_agent/test_session.py

# Run with coverage
pytest --cov=agents.spreadsheet_agent tests/spreadsheet_agent/
```

## Performance Monitoring

Check cache effectiveness:

```python
stats = spreadsheet_memory.get_cache_stats()
print(f"Metadata cache: {stats['metadata']['hits']} hits, {stats['metadata']['misses']} misses")
print(f"Query cache: {stats['query']['hits']} hits, {stats['query']['misses']} misses")
```

## Troubleshooting

### Import Errors
- Ensure `backend/` is in `PYTHONPATH`
- Check that all dependencies are installed: `pandas`, `fastapi`, `openai`, `pydantic`

### Cache Issues
- Clear cache: `spreadsheet_memory.clear_all()`
- Check disk space in `storage/spreadsheet_memory/`

### LLM Provider Failures
- Verify API keys in environment variables
- Check provider chain in logs: `🤖 LLM providers initialized: ...`
- Fallback chain: Cerebras → Groq

## Future Enhancements

- Add more LLM providers (OpenAI, Anthropic)
- Implement distributed caching (Redis)
- Add query optimization hints
- Expand code generation templates
- Add visualization generation

## Version History

- **v2.0.0** (Current): Complete modularization with memory system
- **v1.0.0**: Original monolithic implementation

## License

Part of Orbimesh Agent System

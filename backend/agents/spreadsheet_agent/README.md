# Spreadsheet Agent v2.0

**Purpose**: A specialized agent for spreadsheet operations including upload, natural language querying, transformations, multi-file operations, and safe execution with human-in-the-loop approvals.

## Architecture Overview

The agent is built on a modular architecture with clear separation of concerns:

```
spreadsheet_agent/
├── main.py                    # FastAPI routes & orchestration
├── config.py                  # Configuration & LLM settings
├── models.py                  # Pydantic request/response models
├── actions.py                 # Structured action models & executor
├── planner.py                 # Multi-stage planning workflow
├── simulate.py                # Safe operation preview
├── llm_agent.py              # LLM-powered query processing
├── code_generator.py         # Pandas code generation
├── session.py                # Thread-safe DataFrame registry
├── spreadsheet_session_manager.py  # High-level session handling
├── memory.py                 # LRU caching layer
├── display.py                # Canvas/UI formatting
├── multi_file_ops.py         # Compare & merge operations
└── utils/
    ├── core_utils.py         # Error handling & serialization
    └── data_utils.py         # Validation & data I/O
```

## Core Modules

### 1. main.py - FastAPI Application (~1830 lines)
The main entry point exposing all API routes:
- **File Operations**: `/upload`, `/download`
- **Queries**: `/nl_query` (natural language questions)
- **Transformations**: `/transform`, `/execute_pandas`
- **Planning Workflow**: `/plan_operation` (propose/revise/simulate/execute stages)
- **Multi-File**: `/compare`, `/merge`
- **Utilities**: `/health`, `/stats`, `/get_summary`, `/query`

**Key Features**:
- Decision contract enforcement (blocks unauthorized writes/schema changes)
- Standardized response format (`StandardResponse` from orchestrator)
- Thread-safe operations with async locks
- Metrics tracking middleware

### 2. actions.py - Structured Operations (~526 lines)
Type-safe action models replacing raw pandas code:

**Available Actions**:
- `FilterAction` - Row filtering by condition
- `SortAction` - Sort by column(s)
- `AddColumnAction` - Add calculated columns
- `RenameColumnAction` - Rename columns
- `DropColumnAction` - Remove columns
- `GroupByAction` - Group and aggregate
- `FillNaAction` - Fill missing values
- `DropDuplicatesAction` - Remove duplicate rows
- `AddSerialNumberAction` - Add serial number column
- `AppendSummaryRowAction` - Add summary row with aggregations
- `CompareFilesAction` - Multi-file comparison
- `MergeFilesAction` - Multi-file merge/join

**Benefits**:
- Schema validation before execution
- Clear error messages with suggestions
- Auditable operation history
- Converts to pandas code via `to_pandas_code()`

### 3. planner.py - Multi-Stage Planning (~701 lines)
Implements a safe execution workflow:

**Workflow**: Propose → Revise → Simulate → Execute

**Components**:
- `ExecutionPlan` - Encapsulates actions + reasoning + stage
- `PlanHistory` - Tracks all plans and failure patterns
- `MultiStagePlanner` - Orchestrates the workflow

**Why Multi-Stage?**:
- **Safety**: Preview changes before committing
- **Transparency**: User sees what will happen
- **Reversibility**: Can revise plans based on simulation
- **Learning**: Tracks failures to avoid repeat mistakes

### 4. simulate.py - Safe Preview (~315 lines)
Executes operations on a DataFrame copy:

**Features**:
- Schema validation (catches column name errors early)
- Detects data loss warnings (>50% row reduction)
- Tracks null value increases
- Identifies dtype conversions
- Returns detailed before/after comparison

**Output**: `SimulationResult` with preview_df, warnings, and observation data

### 5. llm_agent.py - Natural Language Processing (~460 lines)
ReAct-style agent for processing NL queries:

**LLM Provider Chain** (with fallback):
1. Cerebras (fast, cost-effective)
2. Groq (reliable fallback)
3. NVIDIA NIM
4. Google Gemini
5. OpenAI GPT-4o-mini
6. Anthropic Claude Haiku

**Features**:
- Safe pandas code execution in sandboxed environment
- Query result caching
- Context-aware conversations
- Automatic error recovery with retries

### 6. session.py - State Management (~140 lines)
Thread-safe DataFrame storage:

**Key Functions**:
- `store_dataframe()` - Save DF to thread-local storage + cache
- `get_dataframe()` - Retrieve DF for thread
- `ensure_file_loaded()` - Smart loading with fallbacks (cache → memory → disk)
- `get_dataframe_state()` - Extract metadata (shape, dtypes, sample)

**Why Thread-Local?**: Isolates conversations—multiple users can work on different files simultaneously without conflicts.

### 7. memory.py - Caching Layer (~257 lines)
LRU caches with TTL and persistence:

**Three-Tier Cache**:
- **Metadata Cache**: 1000 entries, 1h TTL (DataFrame info)
- **Query Cache**: 500 entries, 30min TTL (NL query results)
- **Context Cache**: 200 entries, 1h TTL (conversation state)

**Persistence**: Auto-saves to `storage/spreadsheet_memory/cache.json` on shutdown

### 8. multi_file_ops.py - Multi-File Operations
Compare and merge multiple spreadsheets:

**Compare Modes**:
- `schema_only` - Compare column names and types
- `schema_and_key` - Compare schemas + key-based differences
- `full_diff` - Full row-by-row comparison

**Merge Types**:
- `join` - SQL-style join (inner/left/right/outer)
- `union` - Stack matching columns
- `concat` - Stack all rows/columns

## Key Behaviors & Design Patterns

### 1. Decision Contract Enforcement
The orchestrator sends a `DecisionContract` with each request specifying allowed operations:
```python
contract = {
    "allow_write": False,        # Can modify data?
    "allow_schema_change": False # Can change columns?
}
```
The agent validates instructions against the contract and returns `needs_clarification=True` if blocked.

**Example**: If contract has `allow_write=False` and user says "delete rows", the agent rejects the request rather than guessing intent.

### 2. Multi-Stage Planning Workflow
For complex/destructive operations, use the staged approach:

**Stage 1: Propose**
```bash
POST /plan_operation?stage=propose
Body: {file_id, instruction}
→ Returns: plan_id + canvas summary of proposed actions
```

**Stage 2: Simulate**
```bash
POST /plan_operation?stage=simulate
Body: {file_id, instruction: {"plan_id": "..."}}
→ Returns: preview DataFrame, warnings, before/after comparison
```

**Stage 3: Execute**
```bash
POST /plan_operation?stage=execute
Body: {file_id, instruction: {"plan_id": "...", "force": false}}
→ Returns: modified DataFrame, execution summary
```

**Why?** Gives users visibility and control over destructive operations (drop columns, filter rows, etc.)

### 3. Fast Paths for Simple Queries
For read-only or straightforward operations:
- `/nl_query` - Ask questions (no modification)
- `/transform` - Direct modification with instruction
- `/execute_pandas` - Run pandas code directly

### 4. Multi-File Operations
Compare or merge multiple uploads:

**Compare**:
```python
POST /compare
Body: {
    "file_ids": ["file1", "file2"],
    "comparison_mode": "schema_and_key",
    "key_columns": ["ID"]
}
→ Returns: schema diff, row differences, summary statistics
```

**Merge**:
```python
POST /merge
Body: {
    "file_ids": ["file1", "file2"],
    "merge_type": "join",
    "join_type": "inner",
    "key_columns": ["CustomerID"]
}
→ Returns: merged DataFrame, match statistics
```

### 5. Thread Safety
All operations use:
- `AsyncLock` for pandas operations (prevents race conditions)
- Thread-local storage for DataFrames (isolates conversations)
- Thread-safe LRU caches (concurrent access safe)

## Runtime Configuration

### Storage Paths (config.py)
```python
STORAGE_DIR = "storage/spreadsheets/"           # File uploads
SESSIONS_DIR = "storage/spreadsheet_sessions/"  # Session state
MEMORY_CACHE_DIR = "storage/spreadsheet_memory/" # Cache persistence
```

### File Limits
```python
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
```

### LLM Settings
```python
LLM_TEMPERATURE = 0.1                  # Low temperature for consistency
LLM_MAX_TOKENS_QUERY = 2000            # Max tokens per query
LLM_MAX_TOKENS_CODE_GEN = 2000         # Max tokens for code generation
LLM_TIMEOUT = 60                       # Timeout in seconds
```

### Memory/Cache Settings
```python
MEMORY_CACHE_MAX_SIZE = 1000           # Max cached entries
MEMORY_CACHE_TTL_SECONDS = 3600        # Cache expiration (1 hour)
CONTEXT_MEMORY_MAX_TOKENS = 2000       # Max conversation context
```

## Starting the Agent

```bash
# From repository root
python -m agents.spreadsheet_agent.main

# Or with uvicorn
uvicorn agents.spreadsheet_agent.main:app --host 0.0.0.0 --port 8041
```

## Typical Workflows

### Workflow 1: Upload and Query (Read-Only)
```python
# 1. Upload file
POST /upload
Files: {file: sales_data.csv}
Response: {file_id: "abc123", rows: 1000, columns: 10, ...}

# 2. Ask questions
POST /nl_query
Body: {
    file_id: "abc123",
    instruction: "What are the top 5 products by revenue?"
}
Response: {
    success: true,
    data: {...},
    canvas_display: {...}  # Formatted table for UI
}
```

### Workflow 2: Safe Transformation (With Approval)
```python
# 1. Propose plan
POST /plan_operation?stage=propose
Body: {
    file_id: "abc123",
    instruction: "Add a column for Total = Quantity * Price, then drop TempColumn"
}
Response: {
    plan_id: "plan-xyz",
    canvas_display: {
        actions: [
            {step: 1, action: "Add Column", description: "Create Total"},
            {step: 2, action: "Drop Column", description: "Remove TempColumn"}
        ],
        requires_confirmation: true
    }
}

# 2. Simulate to preview
POST /plan_operation?stage=simulate
Body: {
    file_id: "abc123",
    instruction: '{"plan_id": "plan-xyz"}'
}
Response: {
    simulation: {
        success: true,
        preview_df: {...},
        warnings: ["⚠️ Column 'TempColumn' will be removed"],
        before_shape: [1000, 10],
        after_shape: [1000, 10]  # Same rows, same columns (1 added, 1 removed)
    }
}

# 3. Execute if satisfied
POST /plan_operation?stage=execute
Body: {
    file_id: "abc123",
    instruction: '{"plan_id": "plan-xyz", "force": false}'
}
Response: {
    success: true,
    execution: {...},
    shape: [1000, 10],
    message: "Plan executed successfully"
}
```

### Workflow 3: Quick Transformation (Direct)
```python
# For simple, non-destructive changes
POST /transform
Body: {
    file_id: "abc123",
    instruction: "Sort by Date descending"
}
Response: {
    success: true,
    data: {...},
    message: "Transformation applied"
}
```

### Workflow 4: Multi-File Comparison
```python
# Upload multiple files first, then:
POST /compare
Body: {
    file_ids: ["sales_jan.csv", "sales_feb.csv"],
    comparison_mode: "schema_and_key",
    key_columns: ["OrderID"]
}
Response: {
    schema_comparison: {
        common_columns: ["OrderID", "Amount", "Date"],
        only_in_file1: ["TempCol"],
        only_in_file2: ["NewField"]
    },
    row_differences: {
        only_in_file1: 50,
        only_in_file2: 30,
        modified: 5
    },
    samples: {...}
}
```

### Workflow 5: Multi-File Merge
```python
POST /merge
Body: {
    file_ids: ["customers.csv", "orders.csv"],
    merge_type: "join",
    join_type: "left",
    key_columns: ["CustomerID"]
}
Response: {
    success: true,
    result_file_id: "merged_123",
    rows: 1500,
    columns: 15,
    merge_stats: {
        left_rows: 1000,
        right_rows: 800,
        matched: 750,
        unmatched_left: 250
    }
}
```

## API Endpoints Reference

### File Management
- `POST /upload` - Upload CSV/Excel file
- `GET /download/{file_id}?format=csv|json|xlsx` - Download current state

### Queries & Analysis (Read-Only)
- `POST /nl_query` - Natural language questions
- `POST /query` - Direct pandas query/filter
- `GET /get_summary` - DataFrame summary statistics

### Transformations
- `POST /transform` - Direct transformation with NL instruction
- `POST /execute_pandas` - Execute pandas code (or generate from instruction)
- `POST /simulate_operation` - Preview operation without commit

### Multi-Stage Planning
- `POST /plan_operation?stage=propose` - Generate execution plan
- `POST /plan_operation?stage=revise` - Revise plan with feedback
- `POST /plan_operation?stage=simulate` - Preview plan execution
- `POST /plan_operation?stage=execute` - Execute approved plan

### Multi-File Operations
- `POST /compare` - Compare multiple files
- `POST /merge` - Merge/join multiple files

### Monitoring
- `GET /health` - Health check with cache stats
- `GET /stats` - Detailed metrics (API calls, timing, LLM usage)

## Troubleshooting

### Common Issues

**1. Column Not Found Errors**
```
Error: Column 'Totla' not found. Did you mean: Total?
```
→ The agent provides fuzzy matching suggestions. Check column names with `/get_summary`.

**2. LLM Provider Failures**
```
Warning: Cerebras failed, falling back to Groq...
```
→ Normal behavior. Check API keys if all providers fail.

**3. Cache Issues**
```python
# Clear all caches
from agents.spreadsheet_agent.memory import spreadsheet_memory
spreadsheet_memory.clear_all()
```

**4. Thread ID Confusion**
- Always provide the same `thread_id` for a conversation
- Different `thread_id` = different isolated session
- Missing `thread_id` defaults to `"default"`

**5. Decision Contract Violations**
```json
{
  "success": false,
  "needs_clarification": true,
  "message": "Contract forbids write operations"
}
```
→ Orchestrator blocked the operation. Request permission or reclassify the task.

## Development Tips

### Adding a New Action
1. Create action class in `actions.py` inheriting `SpreadsheetAction`
2. Implement `validate_against_df()` and `to_pandas_code()`
3. Add to `ActionParser.ACTION_MAP`
4. Update planner prompt in `planner.py`

### Adding a New LLM Provider
1. Add API key to `config.py`
2. Add provider to `llm_agent.py` provider chain
3. Test fallback behavior

### Debugging Execution
- Check logs for `🤖`, `📋`, `🧪` emoji tags
- Use `/simulate_operation` to preview without committing
- Enable verbose logging: `LOG_LEVEL=DEBUG`

## Performance Tuning

### Cache Hit Rates
```python
stats = spreadsheet_memory.get_cache_stats()
print(f"Metadata: {stats['metadata']['hit_rate']:.1%}")
print(f"Query: {stats['query']['hit_rate']:.1%}")
```

Target hit rates:
- Metadata cache: >70% (high reuse)
- Query cache: >40% (moderate reuse)
- Context cache: >60% (conversation continuity)

### Memory Usage
- Each cached DataFrame consumes ~MB of RAM
- LRU eviction kicks in at max size
- Consider reducing `MEMORY_CACHE_MAX_SIZE` if memory-constrained

## Testing

```bash
# Unit tests (fast, isolated)
pytest tests/spreadsheet_agent/test_actions.py
pytest tests/spreadsheet_agent/test_memory.py

# Integration tests (slower, requires API keys)
pytest tests/spreadsheet_agent/test_llm_agent.py

# Manual end-to-end test
python backend/tests/spreadsheet_agent/test_spreadsheet_manual.py --dataset retail --difficulty medium
```

## Best Practices

1. **Always use `thread_id`** for multi-user scenarios
2. **Preview before execute** - Use simulate/propose stages for destructive ops
3. **Check decision contracts** - Agent enforces orchestrator policies
4. **Monitor cache stats** - Optimize for cache hit rates
5. **Handle errors gracefully** - Agent provides user-friendly error messages with suggestions
6. **Log everything** - Structured logging helps debug issues
7. **Use StandardResponse format** - Maintains consistency with orchestrator

## Future Roadmap

See [temp/plan.md](../../temp/plan.md) for detailed implementation plan.

### Phase 1: Pause/Resume Protocol
- Add `/resume` endpoint for continuing paused operations
- Add `/status/{transaction_id}` for checking operation state
- Implement transaction rollback/recovery

### Phase 2: Unified Execution
- Consolidate `/nl_query`, `/plan_operation`, `/transform` into single `/execute` endpoint
- Stream progress updates to frontend
- Support partial commits with checkpoints

### Phase 3: Advanced Analytics
- Add `/correlations` endpoint for correlation matrix + insights
- Add `/profile` endpoint for data profiling (nulls, distributions, anomalies)
- Integrate profiling into planning workflow

### Phase 4: Context Compression
- Adaptive context pruning (keep only relevant columns/rows)
- Intelligent sampling for large files
- Remove hard token limits, use dynamic compression

---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**License**: Part of Orbimesh Agent System

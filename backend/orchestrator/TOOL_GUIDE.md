# Orchestrator Tool Integration Guide

## Overview
The orchestrator now has awareness of direct function tools that can handle simple, stateless operations without needing to route to separate agent services.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER REQUEST                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               PARSE PROMPT NODE                          │
│  - LLM sees both TOOLS and AGENTS                       │
│  - Creates tasks with tool-friendly names               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             EXECUTE BATCH NODE                           │
│                                                          │
│  For each task:                                         │
│    ┌──────────────────────────────────┐                │
│    │  1. Is tool capable?             │                │
│    │     ├─ YES → Execute tool ✓      │                │
│    │     └─ NO  → Look up agent       │                │
│    └──────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

## Tool Categories and Capabilities

### Finance Tools (yfinance-based)
**Category**: `finance`
**Capabilities**:
- `get stock quote` - Current price, day high/low, volume
- `get stock price` - Alias for get stock quote
- `check stock price` - Alias for get stock quote
- `current stock price` - Alias for get stock quote
- `get stock history` - Historical OHLCV data
- `stock price history` - Alias for get stock history
- `historical stock data` - Alias for get stock history
- `ohlc data` - Alias for get stock history
- `get company info` - Company details, sector, industry
- `company information` - Alias for get company info
- `company details` - Alias for get company info
- `business summary` - Alias for get company info
- `get key statistics` - PE ratio, market cap, dividend yield
- `financial metrics` - Alias for get key statistics
- `pe ratio` - Alias for get key statistics
- `market cap` - Alias for get key statistics

**Example Parameters**:
```json
{
    "ticker": "AAPL",
    "period": "1mo"  // for history only
}
```

### News Tools (NewsAPI-based)
**Category**: `news`
**Capabilities**:
- `search news` - Search news by keyword
- `find news` - Alias for search news
- `news articles` - Alias for search news
- `get company news headlines` - Alias for search news
- `get top headlines` - Top business headlines by country
- `top news` - Alias for get top headlines
- `latest headlines` - Alias for get top headlines
- `breaking news` - Alias for get top headlines

**Example Parameters**:
```json
{
    "query": "Apple",
    "page_size": 10,
    "country": "us"  // for headlines only
}
```

### Wikipedia Tools
**Category**: `wiki`
**Capabilities**:
- `search wikipedia` - Search for Wikipedia pages
- `wikipedia search` - Alias for search wikipedia
- `find wiki page` - Alias for search wikipedia
- `get wikipedia summary` - Get full page summary
- `wikipedia page` - Alias for get wikipedia summary
- `wiki summary` - Alias for get wikipedia summary
- `get wikipedia section` - Get specific section content
- `wiki section` - Alias for get wikipedia section

**Example Parameters**:
```json
{
    "query": "Artificial Intelligence",  // for search
    "title": "Python (programming language)",  // for summary/section
    "section": "History"  // for section only
}
```

### Search Tools (Groq compound model)
**Category**: `search`
**Capabilities**:
- `web search` - Real-time web search with AI summary
- `search web` - Alias for web search
- `internet search` - Alias for web search
- `search and summarize` - Alias for web search
- `research topic` - Alias for web search

**Example Parameters**:
```json
{
    "query": "Latest developments in AI",
    "include_domains": ["arxiv.org"],  // optional
    "exclude_domains": ["social-media.com"]  // optional
}
```

### Image Tools (Groq vision model)
**Category**: `image`
**Capabilities**:
- `analyze image` - Analyze image and answer questions
- `image analysis` - Alias for analyze image
- `describe image` - Alias for analyze image
- `vision` - Alias for analyze image

**Example Parameters**:
```json
{
    "image_path": "/path/to/image.jpg",
    "query": "What objects are in this image?"
}
```

## How to Use in LLM Prompts

The `parse_prompt` node now includes tool descriptions in the LLM context:

```python
prompt = f'''
    ...
    
    **AVAILABLE DIRECT TOOLS (Fast, stateless operations):**
    These tools can handle simple queries without needing full agent services. Prefer these for straightforward data retrieval:
    {tool_descriptions}
    
    **ENDPOINT-AWARE TASK CREATION:**
    - For simple data queries (stock quotes, news, Wikipedia), use the exact tool capability names shown above
    - For complex operations (document editing, spreadsheet analysis), use agent endpoints
    ...
'''
```

## Execution Priority

When `execute_batch` encounters a task:

1. **PRIORITY 1**: Check tool registry
   - If tool can handle it → Execute tool (0.3-0.5s)
   - Emit task_started/task_completed events
   - Return result immediately

2. **PRIORITY 2**: Agent lookup (fallback)
   - Search agent directory
   - Build agent payload
   - Execute agent API call (2-3s)
   - Handle retries/fallbacks

## Example Task Flow

### User Request: "Get AAPL stock price and latest news"

**1. Parse Prompt Output**:
```json
{
    "tasks": [
        {
            "task_name": "get stock quote",
            "task_description": "Get the current stock price for AAPL",
            "parameters": {"ticker": "AAPL"}
        },
        {
            "task_name": "search news",
            "task_description": "Find latest news articles about AAPL",
            "parameters": {"query": "AAPL", "page_size": 5}
        }
    ]
}
```

**2. Execute Batch**:
- Task 1: `get stock quote` → Tool handles it ✓ (0.4s)
- Task 2: `search news` → Tool handles it ✓ (0.5s)
- **Total**: 0.9s vs 4-6s with agents

### User Request: "Edit this document to add a summary"

**1. Parse Prompt Output**:
```json
{
    "tasks": [
        {
            "task_name": "edit document",
            "task_description": "Edit the document to add a professional summary",
            "parameters": {"file_path": "doc.docx", "instruction": "add summary"}
        }
    ]
}
```

**2. Execute Batch**:
- Task 1: `edit document` → No tool for this ✗
- Falls back to agent lookup → Document Agent found ✓
- Agent executes edit (2-3s)

## Code Integration Points

### 1. Lazy Loading (Automatic - No Startup Code Needed!)
```python
# Tools initialize automatically on FIRST use
# No startup code required in main.py

# When orchestrator first calls:
from orchestrator.tool_registry import is_tool_capable

# This triggers lazy initialization (one-time)
if is_tool_capable("get stock quote"):
    # Tools are now loaded and cached
    pass
```

### 2. Tool Checking (graph.py - execute_batch)
```python
async def try_task_with_fallbacks(planned_task: PlannedTask):
    # PRIORITY 1: Check tools
    from orchestrator.tool_registry import is_tool_capable, execute_tool
    
    if is_tool_capable(planned_task.task_name):
        tool_result = await execute_tool(
            planned_task.task_name,
            planned_task.parameters or {}
        )
        if tool_result.get('success'):
            return format_tool_result(tool_result)
    
    # PRIORITY 2: Agent fallback
    # ... existing agent lookup code ...
```

### 3. LLM Context (graph.py - parse_prompt)
```python
def parse_prompt(state: State):
    # Get tool descriptions
    from orchestrator.tool_registry import get_tool_descriptions
    tool_descriptions = get_tool_descriptions()
    
    # Include in LLM prompt
    prompt = f'''
        ...
        **AVAILABLE DIRECT TOOLS:**
        {tool_descriptions}
        ...
    '''
```

## Performance Monitoring

### Metrics to Track:
- Tool execution time vs agent execution time
- Tool success rate
- Tool usage frequency by category
- Fallback rate (tool fails → agent succeeds)

### Log Patterns:
```
✅ Tool execution successful for 'get stock quote' in 0.4s
⚠️ Tool execution failed for 'get stock quote': API rate limit - falling back to agent
📊 Tool stats: 15 executions, 14 successes, 1 fallback
```

## Adding New Tools

1. Create tool function in `backend/tools/`:
```python
from langchain_core.tools import tool

@tool
def my_new_tool(param1: str, param2: int) -> Dict:
    """Description of what the tool does."""
    return {"result": "data"}
```

2. Register in `tool_registry.py`:
```python
def initialize_tools():
    from tools.my_tools import my_new_tool
    
    register_tool(my_new_tool, [
        "capability name 1",
        "capability name 2"
    ], "category")
```

3. That's it! Tools lazy-load automatically on first use - no restart needed!

## Limitations

Tools are best for:
- ✅ Stateless operations
- ✅ Single API calls
- ✅ Read-only queries
- ✅ Simple parameter mapping

Agents are needed for:
- ❌ Stateful operations (sessions)
- ❌ Complex multi-step workflows
- ❌ File manipulation
- ❌ Authentication/authorization
- ❌ ReAct loops / iterative processing

## Troubleshooting

### Tool not found:
```python
# Check if registered
from orchestrator.tool_registry import get_all_tool_capabilities
print(get_all_tool_capabilities())
```

### Tool failing:
```python
# Test directly
from orchestrator.tool_registry import execute_tool
result = await execute_tool("get stock quote", {"ticker": "AAPL"})
print(result)
```

### LLM not using tools:
- Check if tool descriptions are in parse_prompt
- Verify capability names match exactly
- Check LLM prompt includes tool context

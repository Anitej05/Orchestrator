# Browser Agent - Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Component Details](#component-details)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Data Flows](#data-flows)
9. [Security & Best Practices](#security--best-practices)
10. [Performance & Scalability](#performance--scalability)
11. [Reliability & Error Handling](#reliability--error-handling)
12. [Troubleshooting](#troubleshooting)
13. [Testing](#testing)

---

## Overview

The Browser Agent is a sophisticated, stateful browser automation system designed for complex web tasks. It combines:
- **Playwright** for reliable browser control and DOM interaction
- **LLMs** (NVIDIA/Cerebras/Groq) for intelligent action planning
- **Vision models** for precise element selection via Set-of-Mark overlays
- **State management** for multi-step task memory and replanning
- **Robust error handling** with smart retries and fallbacks

**Key Features:**
- ✅ Stateful: Maintains memory of actions, observations, and extracted data across steps
- ✅ Intelligent: Uses LLMs to reason about page state and plan next actions
- ✅ Resilient: Smart retries, fallback mechanisms, and graceful error handling
- ✅ Observable: Comprehensive logging and screenshot capture for debugging
- ✅ Flexible: Supports text-based and vision-based action planning

---

## Quick Start

### Prerequisites
```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r backend/requirements.txt

# Required environment variables (.env file)
NVIDIA_API_KEY=your_nvidia_key
CEREBRAS_API_KEY=your_cerebras_key
GROQ_API_KEY=your_groq_key
OLLAMA_API_KEY=your_ollama_key  # Optional, for vision
```

### Launch the Agent Server
```bash
# Development mode
cd backend/agents/browser_agent
python -m uvicorn __init__:app --reload --port 8080

# Production mode
uvicorn __init__:app --host 0.0.0.0 --port 8080 --workers 4
```

### Run a Simple Task
```bash
# Terminal 1: Start server
python -m uvicorn backend.agents.browser_agent:app --port 8080

# Terminal 2: Make a request
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Navigate to Google, search for weather in New York, and report the temperature",
    "headless": true,
    "thread_id": "task-001"
  }'
```

### Expected Response
```json
{
  "success": true,
  "task_summary": "Completed web task",
  "actions_taken": [
    {"action": "navigate", "result": "✓ Navigated to https://google.com"},
    {"action": "type", "result": "✓ Typed search query"},
    {"action": "click", "result": "✓ Clicked search button"},
    {"action": "extract", "result": "✓ Extracted weather data"}
  ],
  "extracted_data": {
    "temperature": "32°F",
    "condition": "Clear"
  }
}
```

---

## Architecture

### System Diagram
```
┌─────────────────────────────────────────────────────┐
│  FastAPI Server (__init__.py)                      │
│  • GET /health  (status, metrics)                  │
│  • POST /task   (accepts BrowserTask)              │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼─────┐          ┌───────▼──┐
    │ Planner │          │  Agent   │ (main orchestrator)
    │ (plan   │          │ (state,  │
    │ tasks)  │          │ control) │
    └─────────┘          └────┬─────┘
                              │
        ┌─────────┬───────────┼──────────┬────────┐
        │         │           │          │        │
    ┌───▼──┐  ┌──▼───┐   ┌───▼─┐  ┌────▼──┐  ┌──▼───┐
    │ LLM  │  │ DOM  │   │ Vis │  │Actins │  │Memry │
    │Client│  │Extrc │   │ ion │  │Exectr │  │(state│
    └──────┘  └──┬───┘   └─────┘  └───┬───┘  └──────┘
                 │                      │
             ┌───▼──────────────────────▼──┐
             │   Browser (Playwright)      │
             │   • Page control            │
             │   • Event handling          │
             │   • Tab management          │
             └───────────────────────────┘
```

### Component Overview

| Component | Responsibility | Key Methods |
|-----------|-----------------|-------------|
| **agent.py** | Orchestration and lifecycle | `launch()`, `run()`, `execute_step()` |
| **browser.py** | Playwright wrapper; navigation | `launch()`, `navigate()`, `get_active_page()` |
| **actions.py** | Execute atomic actions | `execute()`, `_execute_single()` |
| **dom.py** | Extract page state | `get_page_content()`, `_get_interactive_elements()` |
| **llm.py** | LLM-based planning | `plan_action()`, `call_llm_direct()` |
| **vision.py** | Vision-based action planning | `_add_som_overlay()`, vision LLM calls |
| **planner.py** | Task decomposition | `create_initial_plan()`, `update_plan()` |
| **state.py** | Memory management | `AgentMemory`, `Subtask`, state updates |
| **schemas.py** | Data contracts | Pydantic models for all data structures |

---

## Component Details

### 1. agent.py — BrowserAgent (Orchestrator)
**Purpose:** Coordinates all components and manages the task lifecycle.

**Key Attributes:**
```python
BrowserAgent(
    task: str,                           # The task to complete
    headless: bool = False,              # Run in headless mode
    thread_id: Optional[str] = None,     # Session identifier
    backend_url: str = "http://localhost:8000"
)
```

**Main Methods:**
- `async launch()` → Starts browser, initializes components
- `async run()` → Executes the full task (plan → execute → repeat until done)
- `async execute_step()` → Executes one iteration (DOM extract → LLM plan → action execute)
- `async _handle_download()` → Manages file downloads

### 2. browser.py — Playwright Wrapper
**Purpose:** Reliable browser control with tab handling.

**Key Methods:**
```python
async launch(headless: bool, on_download=None) → bool
async navigate(url: str, timeout: int = 60000) → bool
async _handle_new_page(page: Page) → None
def get_active_page() → Optional[Page]
```

### 3. actions.py — Action Executor
**Purpose:** Execute browser actions with smart retry logic.

**Supported Actions:**
- `navigate` → Go to URL
- `click` → Click element by xpath or text
- `type` → Type text into input
- `scroll` → Scroll by pixels or to element
- `hover` → Hover over element
- `press` → Press keyboard keys
- `select` → Select option from dropdown
- `wait` → Wait for timeout
- `extract` → Extract data from page
- `done` → Mark task complete
- `screenshot` → Capture screenshot

### 4. dom.py — DOM Extraction
**Purpose:** Build rich page context for LLM.

**Extracted Data:**
- URL, title, viewport info
- Visible body text (up to 200k chars, configurable)
- Interactive elements with XPath and coordinates
- Accessibility tree (semantic structure)
- Scroll position and metrics

### 5. llm.py — LLM Client
**Purpose:** Plan actions using language models.

**Provider Fallback Chain:**
1. **NVIDIA** (minimaxai/minimax-m2) — Best reasoning
2. **Cerebras** (gpt-oss-120b) — Fast fallback
3. **Groq** (openai/gpt-oss-120b) — Last resort

### 6. vision.py — Vision Client
**Purpose:** Enable vision-based action planning with Set-of-Mark overlays.

**Features:**
- Draws numbered boxes on screenshot for each interactive element
- Maps marks to element metadata (xpath, role, name)
- Sends overlaid image to vision LLM for precise action planning

### 7. planner.py — Task Planner
**Purpose:** Decompose main task into subtasks.

### 8. state.py — State Management
**Purpose:** Track plan, history, and observations.

**Core Classes:**
```python
Subtask
  id: int
  description: str
  status: "pending" | "active" | "completed" | "failed"

AgentMemory
  task: str
  plan: List[Subtask]
  history: List[Dict]
  observations: Dict
  extracted_data: Dict
```

---

## Configuration

### Environment Variables
```bash
# LLM API Keys
NVIDIA_API_KEY=nvidia_xxxxxxxxxxxx
CEREBRAS_API_KEY=csk_xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx
OLLAMA_API_KEY=ollama_xxxxxxxxxxxx

# Browser Settings (optional)
BROWSER_HEADLESS=true
BROWSER_TIMEOUT=60000
BROWSER_VIEWPORT_WIDTH=1280
BROWSER_VIEWPORT_HEIGHT=800

# Agent Settings (optional)
AGENT_MAX_STEPS=50
AGENT_STEP_TIMEOUT=30
```

### File Storage
Auto-created directories:
- `storage/browser_downloads/` — Downloaded files (72-hour TTL)
- `storage/browser_screenshots/` — Screenshots (72-hour TTL)

---

## API Reference

### GET /health
Check agent status and basic metrics.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "2.0.0"
}
```

### POST /task
Execute a browser automation task.

**Request:**
```json
{
  "task": "Navigate to example.com and extract the h1 heading",
  "headless": true,
  "thread_id": "task-001"
}
```

**Response:**
```json
{
  "success": true,
  "task_summary": "Completed extraction task",
  "actions_taken": [
    {
      "action": "navigate",
      "message": "✓ Navigated to https://example.com",
      "success": true
    }
  ],
  "extracted_data": {
    "heading": "Welcome to Example.com"
  }
}
```

---

## Usage Examples

### Example 1: Simple Data Extraction
```python
import asyncio
from backend.agents.browser_agent.agent import BrowserAgent

async def extract_data():
    agent = BrowserAgent(
        task="Go to example.com and extract the main heading",
        headless=False,
        thread_id="task-001"
    )
    await agent.launch(headless=False)
    result = await agent.run()
    print(f"Success: {result.success}")
    print(f"Data: {result.extracted_data}")
    await agent.cleanup()

asyncio.run(extract_data())
```

### Example 2: Form Filling
```bash
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Go to example.com/form, fill in email and password, submit the form",
    "headless": true
  }'
```

### Example 3: Multi-Step Navigation
```bash
curl -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Search for laptops on Amazon, filter by price $800-$1200 and rating >= 4 stars, extract top 3 products",
    "headless": true
  }'
```

---

## Data Flows

### High-Level Flow

```
User Request (task string)
      ↓
[Planner] → Breaks into subtasks (5-10 steps)
      ↓
[Loop while tasks remain]:
  ├─ [DOM Extractor] → Get page state
  ├─ [LLM Client] → Plan next action(s)
  ├─ [Action Executor] → Execute plan
  ├─ [Memory] → Update observations
  └─ [Repeat]
      ↓
[Final Result] → Success/failure with extracted data
```

### Data Transformations

1. **DOM Extraction** → Page content dict
2. **LLM Planning** → ActionPlan (actions + reasoning)
3. **Action Execution** → ActionResult (success/failure)
4. **Memory Update** → Observations + extracted_data

---

## Security & Best Practices

### 1. API Key Management
```bash
# ✅ GOOD: Use environment variables
export NVIDIA_API_KEY="..."

# ❌ BAD: Hardcoded in code
API_KEY = "secret_key_123"
```

### 2. URL Validation
- Enforce `http(s)` schemes only
- Reject dangerous schemes (`file://`, `javascript:`, `data:`)
- Add domain allowlist for production

### 3. PII Redaction
- Redact emails, phone numbers in page text before LLM sends
- Avoid screenshots when entering secrets
- Mask sensitive fields in logs

### 4. Logging Best Practices
- Never log API keys or credentials
- Redact typed passwords and sensitive input
- Log action summaries, not raw content

---

## Performance & Scalability

### Optimizations

**DOM Extraction:**
- Limit `body_text` to 50k chars (currently 200k)
- Focus on visible viewport only
- Filter to interactive elements only

**LLM Requests:**
- Add request timeout (30s)
- Cache repeated prompts
- Use smaller models for simple tasks

**Screenshot Handling:**
- Lazy capture only on failure
- Set TTL (72 hours default)
- Auto-cleanup old files

### Resource Usage
| Resource | Per Task | Notes |
|----------|----------|-------|
| Memory | 100-500MB | Depends on page size |
| Browser | ~400MB | Reuse contexts when possible |
| Storage | Variable | 72-hour auto-cleanup |

---

## Reliability & Error Handling

### Retry Strategy

**Click Failures:**
1. Direct click
2. Scroll + wait + retry
3. Text-based fallback
4. Extended wait + original retry

**Navigation Failures:**
1. goto with domcontentloaded
2. Wait on failure, retry (up to 3x)
3. Exponential backoff: 2^n seconds

**LLM Call Failures:**
1. NVIDIA
2. Cerebras
3. Groq
4. Backoff: 1s, 3s, 5s between retries

### Timeout Strategy
- Browser Navigation: 60 seconds
- LLM Call: 30 seconds
- Vision API: 30 seconds
- Per Action: 15 seconds

---

## Troubleshooting

### Browser won't launch
```bash
# Install Playwright browsers
playwright install chromium

# Check disk space and resources
df -h
```

### Actions failing (element not found)
1. Check DOM extraction in logs: `🔍 DOM Extracted X elements`
2. Use vision mode for visual debugging
3. Inspect page manually: `await page.screenshot(path="debug.png")`

### LLM returning invalid actions
1. Verify API key is set: `echo $NVIDIA_API_KEY`
2. Check provider status
3. Provide more context in system prompt

### Out of memory
1. Reduce page text: `BODY_TEXT_MAX = 30000`
2. Clear screenshots more frequently
3. Limit element count
4. Use headless mode

### Screenshots not saving
```bash
mkdir -p storage/browser_screenshots
mkdir -p storage/browser_downloads
ls -la storage/
```

---

## Testing

### Test Categories

**Unit Tests:**
- Agent initialization
- DOM extraction
- State management
- Action execution

**Integration Tests:**
- End-to-end task workflow
- Provider failover
- Error recovery

**High-Priority Tests:**
- ✅ Browser launches and closes gracefully
- ✅ DOM extraction returns valid elements
- ✅ LLM planning returns valid ActionPlan
- ✅ Smart click retry succeeds
- ✅ State marks subtasks correctly
- ✅ Provider fallover works
- ✅ Screenshots save correctly

### Running Tests
```bash
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test
pytest tests/test_agent.py::test_agent_initializes -v

# With coverage
pytest tests/ --cov=backend.agents.browser_agent
```

---

## Implementation Roadmap

### Current (v2.0)
- ✅ Core browser automation
- ✅ Multi-step planning and memory
- ✅ LLM-based action planning
- ✅ Smart retries and error handling
- ✅ Vision-based overlays (Set-of-Mark)

### Planned (v2.1)
- 🔲 Request timeout / backoff standardization
- 🔲 PII redaction pipeline
- 🔲 Sensitive data flag for typing actions
- 🔲 Domain allowlist validation
- 🔲 Comprehensive test suite

### Future (v3.0)
- 🔲 Persistent session management
- 🔲 PDF/document extraction
- 🔲 Custom CSS selector learning
- 🔲 Proxy support
- 🔲 Rate limiting and quota management

---

## Support & References

**Key Files:**
- Configuration: `backend/agents/browser_agent/`
- Storage: `storage/browser_downloads/`, `storage/browser_screenshots/`
- Logs: `storage/browser_agent/logs/`

**External Resources:**
- Playwright docs: https://playwright.dev/python/
- Pydantic docs: https://docs.pydantic.dev/
- NVIDIA API: https://build.nvidia.com/
- FastAPI docs: https://fastapi.tiangolo.com/

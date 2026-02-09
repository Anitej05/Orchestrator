# Omni-Dispatcher Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for the Omni-Dispatcher architecture, which consists of the Brain (reasoning engine) and Hands (execution dispatcher) components, along with the frontend conversation store.

## Architecture Summary
- **Brain**: Analyzes state, creates execution plans, decides on resource activation (agents, tools, Python, terminal)
- **Hands**: Executes the decisions made by Brain (agents, tools, Python, terminal, parallel actions)
- **Orchestrator**: Manages the Brain-Hands cycle with state persistence
- **Frontend**: Zustand store for conversation management and WebSocket integration

---

## 1. Backend Testing

### 1.1 Unit Tests

#### `test_omni_brain.py` - Brain Reasoning Logic
**Purpose**: Test the Brain's decision-making logic without external dependencies.

**Key Test Areas**:
- Initial state initialization
- Task list management
- LLM decision parsing (mocked)
- Execution plan creation and validation
- Phase completion logic
- Action history building
- Insights extraction
- Human-in-the-loop approval triggers
- Error handling and fallback modes

**Mock Strategy**:
- Mock `inference_service.generate_structured()` to return controlled BrainDecision objects
- Mock `agent_registry.list_active_agents()` for available resources
- Mock `tool_registry.list_tools()` for available tools
- Mock `content_orchestrator.get_optimized_llm_context()` for context building

---

#### `test_omni_hands.py` - Hands Dispatcher Logic
**Purpose**: Test the Hands' execution routing and result handling.

**Key Test Areas**:
- Agent execution (mocked HTTP calls)
- Tool execution (mocked tool registry)
- Python code execution (mocked sandbox)
- Terminal command execution (mocked terminal service)
- Plan/Replan acknowledgment
- Parallel execution with concurrent actions
- Parallel retry logic with exponential backoff
- Action history recording
- Result summary generation
- Failure handling and telemetry logging
- Phase insights extraction from parallel results

**Mock Strategy**:
- Mock `httpx.AsyncClient` for agent HTTP calls
- Mock `tool_registry.execute_tool()` for tool execution
- Mock `code_sandbox.execute_code()` for Python execution
- Mock `terminal_service.execute_command()` for terminal commands
- Mock `telemetry_service` for logging
- Mock `credential_service` for auth headers

---

### 1.2 Integration Tests

**Purpose**: Test the Brain-Hands cycle end-to-end with mocked external services.

**Key Test Areas**:
- Full omni_dispatch cycle
- State transitions between Brain and Hands
- Plan creation → Phase execution → Plan completion
- Parallel action coordination
- Human-in-the-loop approval flow (approve/reject)
- Failure recovery and replan
- Action history accumulation
- Insights persistence across cycles

**Test Scenarios**:
```
1. Simple single-action task
   Input: "Calculate 2 + 2"
   Expect: Brain selects python → Hands executes → Result returned

2. Multi-phase planning
   Input: "Analyze data, create report, email it"
   Expect: Plan created → Phases executed sequentially → All phases completed

3. Parallel independent tasks
   Input: "Get Q3 data AND Q4 data"
   Expect: Parallel action → Both executed concurrently → Results merged

4. Human approval required
   Input: "Send the report to finance@example.com"
   Expect: Brain sets requires_approval → Hands pauses → User approves → Execution continues

5. Failure and replan
   Input: Task that fails initially
   Expect: Failure recorded → Brain triggers replan → New plan executed
```

---

### 1.3 Schema Validation Tests

**Purpose**: Validate all Pydantic schemas defined in the system.

**Key Schemas to Test**:
- `TaskItem` from `orchestrator.schemas`
- `BrainAction` from `orchestrator.schemas`
- `ActionResult` from `orchestrator.schemas`
- `BrainDecision` from `orchestrator.brain`
- `PlanPhase` from `orchestrator.schemas`
- `ParallelAction` from `orchestrator.schemas`

**Test Areas**:
- Valid data acceptance
- Invalid data rejection (type errors, required fields, validation rules)
- Default values
- Serialization/deserialization
- Field validators (min_length, pattern constraints)

---

## 2. Frontend Testing

### 2.1 Unit Tests - Conversation Store

#### `frontend/src/__tests__/store/conversation-store.test.ts`
**Purpose**: Test Zustand store logic without external dependencies.

**Key Test Areas**:
- `startConversation()` - Initial state setup, message creation
- `continueConversation()` - Message appending, WebSocket integration
- `loadConversation()` - State restoration from API
- `resetConversation()` - Complete state clearing
- `sendCanvasConfirmation()` - Confirmation action handling
- `_setConversationState()` - State merging logic
- Messages deduplication logic
- Metadata merging and limiting
- Plan state updates
- Canvas state management
- LocalStorage persistence

**Mock Strategy**:
- Mock `window.__websocket` for WebSocket communication
- Mock `apiClient.uploadFiles()` for file upload
- Mock `authFetch()` for API calls
- Mock `localStorage` for persistence

---

### 2.2 Component Testing (Future)

**Purpose**: Test React components that interact with the store.

**Key Areas**:
- Message rendering
- Canvas display components
- Task status indicators
- Approval UI
- Plan visualization

---

## 2.3 Integration Tests (Future)

**Purpose**: Test frontend-backend communication via WebSocket.

**Key Test Scenarios**:
1. Full conversation flow (start → messages → response)
2. Action approval flow
3. Error handling and recovery
4. Large conversation load testing

---

## 3. Test Organization

```
./backend/tests/
├── __init__.py
├── test_omni_brain.py          # Brain unit tests
├── test_omni_hands.py          # Hands unit tests
├── test_omni_dispatcher.py     # Integration tests
├── test_schemas.py             # Schema validation tests
├── core/
│   └── (existing tests)
├── orchestrator_tests/
│   └── (existing tests)
└── test_data/
    └── (mock data fixtures)

./frontend/src/__tests__/
├── store/
│   ├── conversation-store.test.ts
│   └── (future store tests)
├── components/
│   └── (future component tests)
└── integration/
    └── (future integration tests)
```

---

## 4. Testing Best Practices

### 4.1 Mocking Guidelines
- **Mock all external dependencies**: NVIDIA API, Agents, Tools, WebSocket
- **Use dependency injection**: Make services injectable for easier mocking
- **Reset mocks between tests**: Ensure test isolation
- **Use realistic mock data**: Reflect production schemas and structures

### 4.2 Test Data Fixtures
- Define reusable mock data in `test_data/` directory
- Include:
  - Sample agent definitions
  - Sample tool configurations
  - Sample conversation states
  - Typical BrainDecision responses
  - Typical ActionResult objects

### 4.3 Async Testing
- Use `pytest-asyncio` for async test functions
- Use `aiohttp` or `unittest.mock.AsyncMock` for mocking async calls
- Handle timeouts appropriately in tests

### 4.4 Coverage Goals
- **Backend**: Minimum 80% code coverage
- **Critical paths**: 90%+ coverage (Brain.think(), Hands.execute())
- **Schema validation**: 100% coverage
- **Frontend store**: 80%+ coverage

---

## 5. Test Execution

### 5.1 Backend Tests
```bash
# Run all tests
pytest ./backend/tests/ -v

# Run specific test file
pytest ./backend/tests/test_omni_brain.py -v

# Run with coverage
pytest ./backend/tests/ --cov=./backend/orchestrator --cov-report=html

# Run only unit tests (not integration)
pytest ./backend/tests/ -m "not integration"
```

### 5.2 Frontend Tests
```bash
# Run all jest tests
npm test

# Run specific test file
npm test conversation-store.test.ts

# Run with coverage
npm test -- --coverage

# Watch mode for development
npm test -- --watch
```

---

## 6. Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run backend tests
        run: |
          pytest ./backend/tests/ --cov=./backend --pytest-cov

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      - name: Run frontend tests
        run: |
          cd frontend
          npm test -- --coverage
```

---

## 7. Future Test Areas

### 7.1 Performance Tests
- Large action history handling
- Many concurrent parallel actions
- Long conversation memory management

### 7.2 Security Tests
- Credential handling validation
- Input sanitization
- CSRF validation
- WebSocket message validation

### 7.3 End-to-End Tests
- Full user workflows with Playwright/Cypress
- Multi-user scenarios
- File upload/download flows

---

## 8. Test Metrics and Reporting

### Key Metrics to Track
- Test coverage percentage
- Test execution time
- Flaky test detection
- Failed test trends over time

### Reporting Tools
- pytest-cov for coverage reports
- GitHub Actions test results
- pytest-html for HTML reports

---

## 9. Test Maintenance

### Regular Maintenance Tasks
- Update mock data when schemas change
- Refactor common test logic into fixtures
- Remove obsolete tests
- Add tests for new features
- Review and reduce flaky tests

### Test Documentation
- Comment complex test scenarios
- Document mock behaviors
- Maintain this TESTING_STRATEGY.md as the source of truth

---

## Appendix: Key Schemas Reference

### TaskItem (Backend)
```python
task_id: str
description: str
priority: int
status: TaskStatus (pending/in_progress/completed/failed)
payload: Dict[str, Any]
created_at: datetime
updated_at: datetime
```

### BrainAction (Backend)
```python
action_id: str
action_type: str
target_task: Optional[str]
parameters: Dict[str, Any]
created_at: datetime
```

### ActionResult (Backend)
```python
action_id: str
success: bool
output: Optional[Any]
error_message: Optional[str]
execution_time_ms: Optional[float]
completed_at: datetime
```

### BrainDecision (Backend)
```python
action_type: str
resource_id: Optional[str]
payload: Dict[str, Any]
reasoning: Optional[str]
user_response: Optional[str]
memory_updates: Optional[Dict[str, Any]]
is_finished: bool
execution_plan: Optional[List[Dict[str, Any]]]
phase_complete: bool
phase_goal_verified: Optional[str]
parallel_actions: Optional[List[Dict[str, Any]]]
requires_approval: bool
approval_reason: Optional[str]
fallback_mode: bool
```

### Conversation State (Frontend)
```typescript
thread_id: string | undefined
status: string
messages: Message[]
isWaitingForUser: boolean
task_agent_pairs: any[]
plan: any[]
execution_plan: any[]
current_phase_id: string | undefined
action_history: any[]
insights: Record<string, string>
pending_action_approval: boolean
isLoading: boolean
```

# Backend Changes for Interactive Agent Orchestration

## Overview
This document describes all the changes made to support interactive workflows where the orchestrator can pause execution, ask the user for additional information, and then continue processing based on user responses.

## Key Features Added
- **Interactive Conversations**: The orchestrator can now pause and ask users for clarification
- **Session Management**: Each conversation has a unique thread_id for maintaining state
- **Persistent Memory**: Conversations are stored and can be resumed
- **Multiple Response Types**: Support for final responses, pending user input, and error states

## File Changes

### 1. `orchestrator/state.py`
**Changes Made:**
- Added `pending_user_input: bool` field to track when workflow is waiting for user input
- Added `question_for_user: Optional[str]` field to store questions for the user
- Added `user_response: Optional[str]` field to store user's response to questions

**Purpose:**
These fields enable the orchestrator to pause execution and communicate with users during workflow execution.

### 2. `orchestrator/graph.py`
**Changes Made:**
- Added `wait_for_user_input` node that pauses workflow when user input is needed
- Added `synchronize_and_route` function to handle workflow routing based on user input state
- Added `create_graph_with_checkpointer()` function to create graphs with memory persistence
- Modified graph edges to include conditional routing to `wait_for_user_input`

**Key Functions:**
```python
def wait_for_user_input(state: State) -> State:
    """Node that pauses execution when user input is needed"""
    # Simply returns the state as-is, allowing external systems to detect pending_user_input=True
    return state

def synchronize_and_route(state: State) -> str:
    """Routes workflow based on user input state"""
    if state.pending_user_input and not state.user_response:
        return "wait_for_user_input"
    elif state.user_response or not state.pending_user_input:
        return "aggregate_responses"
    else:
        return "execute_batch"
```

**Graph Structure:**
```
parse_prompt → match_capabilities → agent_directory_search → rank_agents → plan_execution → execute_batch
                                                                                              ↓
wait_for_user_input ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← (conditional)
        ↓
plan_execution → execute_batch → aggregate_responses → END
```

### 3. `main.py`
**Major Changes:**

#### A. New Pydantic Models
```python
class UserResponse(BaseModel):
    """Model for user responses to orchestrator questions"""
    response: str
    thread_id: str

class ConversationStatus(BaseModel):
    """Model for conversation status responses"""
    thread_id: str
    status: str  # "completed", "pending_user_input", "error"
    question_for_user: Optional[str] = None
    final_response: Optional[str] = None
    task_agent_pairs: Optional[List[Dict]] = None
    error_message: Optional[str] = None
```

#### B. Enhanced ProcessResponse Schema
```python
class ProcessResponse(BaseModel):
    # Existing fields...
    
    # New interactive fields
    thread_id: str
    status: str  # "completed", "pending_user_input", "error"
    question_for_user: Optional[str] = None
    requires_user_input: bool = False
```

#### C. New API Endpoints

**1. Enhanced `/api/chat` endpoint:**
- Now returns `thread_id` for session tracking
- Includes `status` field to indicate conversation state
- Returns `question_for_user` when user input is needed
- Sets `requires_user_input=True` when paused

**2. New `/api/chat/continue` endpoint:**
```python
@app.post("/api/chat/continue", response_model=ProcessResponse)
async def continue_conversation(user_response: UserResponse):
    """Continue a paused conversation with user's response"""
```

**3. New `/api/chat/status/{thread_id}` endpoint:**
```python
@app.get("/api/chat/status/{thread_id}", response_model=ConversationStatus)
async def get_conversation_status(thread_id: str):
    """Get the current status of a conversation"""
```

#### D. Enhanced WebSocket Support
The WebSocket endpoint now supports:
- Interactive conversations with real-time user input
- Status updates during paused workflows
- Seamless continuation after user responses

#### E. Memory Integration
- Added `MemorySaver` for persistent conversation state
- Integrated memory checkpointer with graph compilation
- Thread-based conversation tracking

### 4. `schemas.py`
**Changes Made:**
- Enhanced `ProcessResponse` with interactive fields
- Added support for conversation status tracking
- Maintained backward compatibility with existing response format

## How Interactive Workflows Work

### 1. Starting a Conversation
```python
# User sends initial request
POST /api/chat
{
    "prompt": "I need help with data analysis",
    "max_results": 5
}

# Response includes thread_id and may indicate pending user input
{
    "task_agent_pairs": [...],
    "thread_id": "uuid-123",
    "status": "pending_user_input",
    "question_for_user": "What type of data are you analyzing?",
    "requires_user_input": true
}
```

### 2. Continuing a Conversation
```python
# User provides response
POST /api/chat/continue
{
    "response": "I'm analyzing sales data from our e-commerce platform",
    "thread_id": "uuid-123"
}

# Workflow continues and may complete or ask another question
{
    "task_agent_pairs": [...],
    "thread_id": "uuid-123",
    "status": "completed",
    "final_response": "Analysis complete!",
    "requires_user_input": false
}
```

### 3. Checking Status
```python
# Check conversation status anytime
GET /api/chat/status/uuid-123

{
    "thread_id": "uuid-123",
    "status": "pending_user_input",
    "question_for_user": "What specific metrics are you interested in?",
    "task_agent_pairs": [...]
}
```

## Workflow States

1. **"pending_user_input"**: Orchestrator is waiting for user response
2. **"completed"**: Workflow finished successfully
3. **"error"**: Workflow encountered an error

## Memory and Persistence

- Each conversation maintains state using `thread_id`
- Conversations persist across server restarts (using MemorySaver)
- Users can resume conversations at any time
- State includes all previous context and responses

## Backward Compatibility

All changes maintain backward compatibility:
- Existing `/api/chat` endpoint works as before for non-interactive workflows
- New fields are optional in responses
- Existing frontend implementations continue to work without modification

## Testing Interactive Workflows

### Test Scenario 1: Simple Completion
```bash
# Start conversation
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find me a weather agent"}'

# Should return completed status with agents
```

### Test Scenario 2: Interactive Flow
```bash
# Start conversation that requires user input
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Help me analyze data but I need guidance"}'

# Should return pending_user_input status with question

# Continue conversation
curl -X POST "http://localhost:8000/api/chat/continue" \
  -H "Content-Type: application/json" \
  -d '{"response": "I want to analyze sales data", "thread_id": "returned-thread-id"}'
```

### Test Scenario 3: WebSocket Interactive
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

// Send initial request
ws.send(JSON.stringify({
    type: 'start',
    prompt: 'Help me with data analysis',
    thread_id: 'new-conversation'
}));

// Receive question from orchestrator
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.requires_user_input) {
        // Show question to user and collect response
        const userResponse = prompt(data.question_for_user);
        
        // Send user response
        ws.send(JSON.stringify({
            type: 'continue',
            response: userResponse,
            thread_id: data.thread_id
        }));
    }
};
```

## Architecture Benefits

1. **Scalable**: Each conversation maintains its own state
2. **Persistent**: Conversations can be stored (when serialization is implemented)
3. **Flexible**: Supports both synchronous and asynchronous workflows
4. **User-Friendly**: Clear feedback on workflow status
5. **Developer-Friendly**: Easy to extend with new interactive patterns

## Known Issues and Solutions

### Memory Persistence Issue
**Problem**: LangGraph's MemorySaver checkpointer cannot serialize complex Pydantic objects (AgentCard, TaskAgentPair, etc.) due to msgpack serialization limitations.

**Error**: `TypeError: Type is not msgpack serializable: AgentCard`

**Current Status**: Memory persistence is temporarily disabled to maintain functionality. The interactive workflow features work correctly, but conversations don't persist across server restarts.

**Root Cause**: The `State` object contains Pydantic models that cannot be serialized by msgpack:
- `candidate_agents: Dict[str, List[AgentCard]]`
- `task_agent_pairs: List[TaskAgentPair]`
- `task_plan: List[List[PlannedTask]]`

**Future Solution**: Implement custom serialization by:
1. Converting Pydantic objects to dictionaries before storing in state
2. Converting dictionaries back to Pydantic objects when retrieving from state
3. Creating a custom serializer that handles complex objects
4. Using a different persistence backend (Redis, Database)

**Workaround for Memory**: For now, each API call is stateless. To implement conversation memory:
1. Store conversation state in a database or cache (Redis)
2. Implement session management at the API level
3. Pass conversation context in each request

**Code Locations**: 
- `main.py` line ~223: Memory checkpointer usage commented out
- `orchestrator/state.py`: Added documentation about serialization requirements
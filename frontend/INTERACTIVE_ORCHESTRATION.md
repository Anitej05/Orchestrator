# Interactive Agent Orchestration Implementation

This document describes the implementation of interactive agent orchestration features in the frontend, following the requirements outlined in the implementation guide.

## Overview

The frontend has been enhanced with three different modes for agent orchestration:

1. **Interactive Mode** - Conversational interface with clarifying questions
2. **Real-time Mode** - WebSocket-based live updates and bidirectional communication  
3. **Classic Mode** - Traditional single-request workflow execution

## Implementation Details

### 1. Enhanced API Client (`lib/api-client.ts`)

#### New Type Definitions
```typescript
export interface ProcessResponse {
  task_agent_pairs: TaskAgentPair[];
  message: string;
  thread_id: string;
  status: 'pending_user_input' | 'completed' | 'error';
  question_for_user?: string;
  requires_user_input: boolean;
  final_response?: string;
  error_message?: string;
}

export interface ConversationState {
  thread_id: string;
  status: 'pending_user_input' | 'completed' | 'error';
  messages: Message[];
  isWaitingForUser: boolean;
  currentQuestion?: string;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    task_agent_pairs?: TaskAgentPair[];
    progress?: number;
  };
}
```

#### New API Endpoints
```typescript
// Start a new conversation
export async function startConversation(prompt: string): Promise<ProcessResponse>

// Continue an existing conversation
export async function continueConversation(response: string, threadId: string): Promise<ProcessResponse>

// Get conversation status
export async function getConversationStatus(threadId: string): Promise<ConversationStatus>
```

### 2. Conversation Management Hook (`hooks/use-conversation.ts`)

The `useConversation` hook manages conversation state and handles the flow between user input and system responses:

#### Key Features
- **State Management**: Tracks conversation thread, messages, and waiting states
- **Message Handling**: Manages user, system, and assistant message types
- **Error Handling**: Graceful error recovery and user notification
- **Conversation Flow**: Handles both single-turn and multi-turn conversations

#### Usage Example
```typescript
const { state, isLoading, startConversation, continueConversation } = useConversation({
  onComplete: (result) => {
    // Handle completed workflow
  },
  onError: (error) => {
    // Handle errors
  }
});
```

### 3. Interactive Chat Interface (`components/interactive-chat-interface.tsx`)

A conversational UI component that supports:

#### Features
- **Message Display**: Shows conversation history with different message types
- **Dynamic Input**: Switches between initial prompt and response modes
- **Status Indicators**: Visual feedback for conversation state
- **Task Agent Pairs**: Displays suggested workflow components
- **Reset Functionality**: Clear conversation and start over

#### Visual Elements
- User messages (blue, right-aligned)
- System questions (yellow background)
- Assistant responses (gray background)
- Task-agent pair cards with pricing information

### 4. WebSocket Real-time Communication (`hooks/use-websocket-conversation.ts`)

Provides real-time bidirectional communication for live workflow updates:

#### Features
- **Auto-reconnection**: Handles connection drops gracefully
- **Progress Updates**: Real-time progress percentage tracking
- **Message Streaming**: Live message updates during workflow execution
- **Error Recovery**: Automatic reconnection and error handling

#### WebSocket Message Types
```typescript
interface WebSocketMessage {
  type: 'progress' | 'completion' | 'error' | 'user_input_required';
  progress_percentage?: number;
  requires_user_input?: boolean;
  question_for_user?: string;
  final_response?: string;
  task_agent_pairs?: any[];
  message?: string;
  thread_id?: string;
  error?: string;
}
```

### 5. Enhanced Task Builder (`components/task-builder.tsx`)

The main orchestration interface now includes three tabs:

#### Interactive Mode
- Conversational interface using `InteractiveChatInterface`
- Step-by-step clarification process
- Natural language interaction

#### Real-time Mode  
- WebSocket-based communication using `WebSocketWorkflow`
- Live progress updates
- Bidirectional real-time messaging

#### Classic Mode
- Traditional single-request workflow
- Immediate task submission
- Dry run capabilities

### 6. WebSocket Workflow Component (`components/websocket-workflow.tsx`)

Dedicated component for real-time workflow orchestration:

#### Features
- **Connection Management**: Connect/disconnect controls
- **Live Progress**: Real-time progress bar
- **Interactive Questions**: Handle user input requests during execution
- **Message History**: Complete conversation tracking
- **Status Indicators**: Connection and workflow status

## Backend Integration Points

### Expected API Endpoints

1. **POST /api/chat** - Start conversation
   ```json
   {
     "prompt": "string",
     "max_results": 5
   }
   ```

2. **POST /api/chat/continue** - Continue conversation
   ```json
   {
     "response": "string",
     "thread_id": "string"
   }
   ```

3. **GET /api/chat/status/:threadId** - Get conversation status

4. **WebSocket /ws/chat** - Real-time communication
   - Supports both conversation start and continuation
   - Provides progress updates and completion notifications

### Response Format
```json
{
  "task_agent_pairs": [...],
  "message": "string",
  "thread_id": "string",
  "status": "pending_user_input" | "completed" | "error",
  "question_for_user": "string?",
  "requires_user_input": boolean,
  "final_response": "string?",
  "error_message": "string?"
}
```

## State Management Strategy

### Conversation Persistence
- Local state management for active conversations
- Thread ID tracking for continuation
- Message history maintenance
- Optional localStorage persistence for session recovery

### Error Handling
- Network error recovery
- Invalid thread ID handling
- WebSocket connection failures
- Graceful degradation to HTTP endpoints

### UI State Management
- Tab switching between modes
- Loading states during API calls
- Form validation and submission
- Results display and agent rating

## Features Implemented

✅ **Conversation State Management**
- Thread ID tracking
- Message history
- Status management (pending/completed/error)

✅ **Enhanced API Client**
- New interactive endpoints
- Type-safe response handling
- Error management

✅ **Interactive Chat Interface**
- Conversational UI
- Dynamic input modes
- Message type handling

✅ **WebSocket Integration**
- Real-time communication
- Progress updates
- Bidirectional messaging

✅ **Tabbed Interface**
- Three orchestration modes
- Shared result display
- Mode-specific features

✅ **Backward Compatibility**
- Existing API endpoints continue to work
- Legacy processPrompt function maintained
- Gradual migration support

## Testing Recommendations

### Interactive Mode Testing
1. Start conversation with ambiguous request
2. Respond to system clarification questions
3. Verify workflow completion and agent suggestions

### Real-time Mode Testing
1. Connect to WebSocket endpoint
2. Send workflow request
3. Monitor progress updates
4. Handle interactive questions during execution

### Error Scenarios
1. Network disconnection during conversation
2. Invalid thread ID usage
3. WebSocket connection failures
4. Malformed API responses

## Future Enhancements

### Planned Features
- **Conversation Persistence**: Save/restore conversations across sessions
- **Advanced WebSocket Features**: File uploads, streaming responses
- **UI Improvements**: Message reactions, conversation branching
- **Performance Optimization**: Message virtualization, connection pooling

### Integration Opportunities
- **Voice Interface**: Speech-to-text for conversation input
- **Visual Workflow Builder**: Drag-and-drop task configuration
- **Analytics Dashboard**: Conversation success metrics
- **Agent Marketplace**: Browse and discover new agents during conversation

## Configuration

### Environment Variables
```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# WebSocket URL
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Feature flags
NEXT_PUBLIC_ENABLE_WEBSOCKET=true
NEXT_PUBLIC_ENABLE_INTERACTIVE_MODE=true
```

### Component Props
Most components accept optional `onComplete`, `onError`, and `className` props for customization and integration.

## Migration Guide

For existing implementations:

1. **Update imports**: Add new hooks and components
2. **Enhance API calls**: Use new conversation endpoints
3. **Add UI components**: Integrate interactive interfaces
4. **Configure WebSocket**: Set up real-time communication
5. **Test thoroughly**: Verify all modes work correctly

The implementation maintains full backward compatibility while providing powerful new interactive features for enhanced user experience.

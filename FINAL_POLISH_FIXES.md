# Final Polish Fixes

## Issue 1: Premature Timeout Message

### Problem
A "Request timed out. Please try again." message would appear briefly during answer generation, then immediately disappear when the actual answer arrived.

### Root Cause
The frontend had a 2-minute timeout that would trigger even while the backend was still processing. The timeout message would be added to the chat, but then immediately replaced when the `__end__` event arrived with the actual response.

### Solution
Removed the client-side timeout logic. The WebSocket connection itself handles timeouts properly, and the backend will always send either an `__end__` or `__error__` event to complete the request.

**Files Changed:**
- `frontend/lib/conversation-store.ts`: Removed setTimeout blocks in both `startConversation` and `continueConversation`

## Issue 2: Sidebar Metadata Not Loading

### Problem
When loading a previous conversation, the messages would load correctly but the sidebar tabs (Metadata, Plan, Attachments, Canvas) would be empty.

### Root Cause
The `_setConversationState` function was trying to be "smart" by merging/accumulating data from the new state with existing state. This worked fine for ongoing conversations, but when **loading** a conversation, it would merge the loaded data with empty state, causing issues.

For example:
- Loading conversation with `task_agent_pairs: [{task1}]`
- Current state has `task_agent_pairs: []`
- Merge logic would try to deduplicate, but since current is empty, it would just use the loaded data
- However, the logic was complex and had edge cases

### Solution
Added logic to detect when we're **loading** a conversation vs **updating** an existing one:

```typescript
const isLoadingConversation = newState.thread_id && newState.thread_id !== state.thread_id;

if (isLoadingConversation) {
  // Replace data when loading a different conversation
  updatedTaskAgentPairs = newState.task_agent_pairs;
} else {
  // Merge data when updating current conversation
  // ... merge logic ...
}
```

This ensures that:
- **Loading a conversation**: All data is replaced with the loaded data
- **Updating current conversation**: Data is merged/accumulated properly

**Files Changed:**
- `frontend/lib/conversation-store.ts`: Updated `_setConversationState` to handle loading vs updating differently

## Testing

### Test Timeout Fix
1. Send a message that takes a while to process (e.g., "Explain quantum computing in detail")
2. **Expected**: No timeout message appears
3. **Expected**: Only the actual response appears when ready

### Test Sidebar Loading
1. Have a conversation with multiple messages
2. Reload the page
3. **Expected**: Messages load correctly
4. **Expected**: Sidebar tabs show correct data:
   - **Metadata**: Shows original prompt, completed tasks, etc.
   - **Plan**: Shows execution plan if any
   - **Attachments**: Shows uploaded files if any
   - **Canvas**: Shows canvas content if any

## Summary

Both issues were caused by overly complex logic trying to be "smart":
1. **Timeout**: Trying to handle timeouts on both client and server
2. **Sidebar**: Trying to merge data in all cases instead of replacing when loading

The fixes simplify the logic:
1. **Timeout**: Trust the WebSocket to handle timeouts
2. **Sidebar**: Detect loading vs updating and handle appropriately

**Result**: Cleaner, more predictable behavior.

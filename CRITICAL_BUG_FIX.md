# Critical Bug Fix: Missing User Messages

## The Problem

When sending a second message in a conversation, the user message would disappear after the AI responded. Only the first user message and all AI responses were visible.

## Root Cause

The `continueConversation` function was sending the wrong field to the backend:

```typescript
// WRONG - This is for answering system questions only
const messageData = {
  thread_id: thread_id,
  user_response: input,  // ❌ WRONG FIELD
  files: [...]
};
```

### What Happened

1. User sends second message "What can you do?"
2. Frontend calls `continueConversation`
3. Frontend sends `{thread_id: "abc", user_response: "What can you do?"}`
4. Backend receives `user_response` field
5. Backend treats this as answering a system question (not a new user message)
6. Backend doesn't add the user message to the conversation history
7. Backend only adds the AI response
8. Result: User message is missing!

### Backend Logic

The backend has two different flows:

**Flow 1: New/Continuing Conversation** (uses `prompt` field)
```python
if prompt and current_conversation:
    # Add user message to history
    messages = current_conversation.get("messages", []) + [HumanMessage(content=prompt)]
```

**Flow 2: Answering System Question** (uses `user_response` field)
```python
if user_response:
    # Update state with user's answer, but don't add as a message
    initial_state["user_response"] = user_response
```

The frontend was using Flow 2 when it should use Flow 1!

## The Fix

Changed `continueConversation` to send `prompt` instead of `user_response`:

```typescript
// CORRECT - This continues the conversation
const messageData = {
  thread_id: thread_id,
  prompt: input,  // ✅ CORRECT FIELD
  files: [...]
};
```

## File Changed

- `frontend/lib/conversation-store.ts`: Line 212

## Testing

After this fix:

1. **Send first message "Hi!"**
   - Should see: User "Hi!" + AI response

2. **Send second message "What can you do?"**
   - Should see: User "Hi!" + AI response + User "What can you do?" + AI response
   - **User message should NOT disappear!**

3. **Reload page**
   - Should see all 4 messages
   - No messages missing

4. **Send third message**
   - Should see all previous messages + new exchange
   - Total 6 messages

## Why This Bug Was Hard to Find

1. The field name `user_response` sounds like it should work for user messages
2. The backend accepted it without error (it's a valid field)
3. The backend still generated a response (so it seemed to work)
4. Only the message history was affected (subtle bug)

## Lesson Learned

**Always check the API contract!**

The WebSocket API has two distinct fields:
- `prompt`: For new messages in a conversation
- `user_response`: For answering system questions only

Using the wrong field causes silent data loss.

# Gmail Agent - Quick Start Guide

## Prerequisites

1. **Environment Variables** - Ensure these are set in your `.env`:
   ```bash
   COMPOSIO_API_KEY=your_composio_key
   GROQ_API_KEY=your_groq_key  # Or CEREBRAS_API_KEY or NVIDIA_API_KEY
   ```

2. **User Gmail Connection** - User must connect Gmail via OAuth:
   ```bash
   # Connect Gmail for user
   curl -X POST "http://localhost:8000/api/integrations/auth/start/{user_id}?apps=gmail"
   
   # Verify connection
   curl "http://localhost:8000/api/integrations/status/{user_id}"
   ```

## Start the Agent

### Option 1: Standalone
```bash
cd backend/agents/gmail_agent
python agent.py
```
Agent runs on: `http://localhost:8003`

### Option 2: Via Orchestrator
The orchestrator will automatically discover gmail_agent from `agent_entries/gmail_agent.json`

## Test Endpoints

### 1. Health Check
```bash
curl http://localhost:8003/health
```
Expected: `{"status":"healthy","agent":"gmail_agent"}`

### 2. Search Emails
```bash
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "query": "unread emails",
    "max_results": 5
  }'
```

### 3. Get Email
```bash
curl "http://localhost:8003/message/your_user_id/message_id_here"
```

### 4. Summarize Emails
```bash
curl -X POST http://localhost:8003/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "message_ids": ["msg1", "msg2", "msg3"]
  }'
```

### 5. Draft Smart Reply
```bash
curl -X POST http://localhost:8003/draft-reply \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "message_id": "msg_id_here",
    "user_instructions": "Thank them for the update"
  }'
```

### 6. Send Email
```bash
curl -X POST http://localhost:8003/send \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "to": "recipient@example.com",
    "subject": "Test Email",
    "body": "Hello from Gmail Agent!",
    "is_html": false
  }'
```

### 7. List Drafts
```bash
curl "http://localhost:8003/drafts/your_user_id?max_results=10"
```

### 8. Extract Action Items
```bash
curl -X POST http://localhost:8003/extract-actions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "message_ids": ["msg1", "msg2"]
  }'
```

### 9. List Labels
```bash
curl "http://localhost:8003/labels/your_user_id"
```

### 10. Download Attachments
```bash
curl -X POST http://localhost:8003/attachments/download \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "message_id": "msg_id_here"
  }'
```

## Via Orchestrator

Once the agent is running, the orchestrator can route to it:

```bash
curl -X POST http://localhost:8000/api/orchestrator/process \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "your_user_id",
    "request": "Search for unread emails from my boss"
  }'
```

The orchestrator will:
1. Detect this is an email task
2. Route to `gmail_agent`
3. Execute search with LLM optimization
4. Return results

## Interactive Testing with Swagger UI

Open browser: `http://localhost:8003/docs`

This provides:
- Interactive API documentation
- Try-it-now functionality
- Schema validation
- Example requests/responses

## Common Workflows

### Workflow 1: Search → Summarize
```bash
# Step 1: Search
SEARCH_RESULT=$(curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","query":"project updates","max_results":5}')

# Step 2: Extract message IDs (using jq)
MESSAGE_IDS=$(echo $SEARCH_RESULT | jq -r '.messages[].id' | jq -R -s -c 'split("\n")[:-1]')

# Step 3: Summarize
curl -X POST http://localhost:8003/summarize \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"user123\",\"message_ids\":$MESSAGE_IDS}"
```

### Workflow 2: Search → Draft Reply → Send
```bash
# Step 1: Search for specific email
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","query":"Demo Request","max_results":1}'

# Step 2: Draft smart reply
curl -X POST http://localhost:8003/draft-reply \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","message_id":"msg_id","user_instructions":"Confirm meeting"}'

# Step 3: Send draft (get draft_id from step 2)
curl -X POST "http://localhost:8003/draft/user123/draft_id/send"
```

### Workflow 3: Extract Actions → Create Tasks
```bash
# Step 1: Search emails
curl -X POST http://localhost:8003/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","query":"deadline OR meeting","max_results":10}'

# Step 2: Extract action items
curl -X POST http://localhost:8003/extract-actions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","message_ids":["msg1","msg2","msg3"]}'

# Result includes deadlines, meetings, tasks with priorities
```

## Troubleshooting

### Issue: "User not connected to Gmail"
**Solution:**
```bash
# Connect Gmail
curl -X POST "http://localhost:8000/api/integrations/auth/start/your_user_id?apps=gmail"
# Follow OAuth flow in browser
```

### Issue: "LLM providers unavailable"
**Solution:**
```bash
# Check API keys in .env
echo $GROQ_API_KEY
echo $CEREBRAS_API_KEY
echo $NVIDIA_API_KEY

# At least one must be set
```

### Issue: Port 8003 already in use
**Solution:**
```bash
# Find and kill process
netstat -ano | findstr :8003
taskkill /PID <PID> /F

# Or change port in agent.py:
# uvicorn.run(app, host="0.0.0.0", port=8004)
```

### Issue: Connection errors
**Solution:**
```bash
# Refresh OAuth token
curl -X POST "http://localhost:8000/api/integrations/refresh/your_user_id/gmail"

# Check connection status
curl "http://localhost:8000/api/integrations/status/your_user_id"
```

## Monitoring

### Logs
```bash
# View real-time logs
tail -f logs/gmail_agent.log

# Search for errors
grep -i "error" logs/gmail_agent.log
```

### Metrics
The agent logs key metrics:
- Tool execution times
- Success/failure rates
- LLM provider performance
- Memory usage

## Performance Tips

1. **Concurrent Fetching** - The agent fetches up to 5 emails concurrently by default
2. **Result Caching** - Search results cached in memory for follow-up actions
3. **Service Reuse** - User service instances cached (don't create per request)
4. **Batch Operations** - Use batch endpoints when possible (e.g., download all attachments)

## Next Steps

1. ✅ Test basic operations (search, send, get)
2. ✅ Test LLM features (summarize, draft reply, extract actions)
3. ✅ Test draft management (create, list, send)
4. ✅ Test advanced features (labels, attachments, contacts)
5. ✅ Monitor logs for errors
6. ✅ Collect user feedback
7. ✅ Optimize based on usage patterns

## Documentation

- **README.md** - Complete feature documentation
- **IMPLEMENTATION_PLAN.md** - Architecture and design
- **MIGRATION_COMPLETE.md** - Migration status and checklist
- **Swagger UI** - http://localhost:8003/docs

## Support

For issues or questions:
1. Check logs: `logs/gmail_agent.log`
2. Review documentation: `README.md`
3. Test with Swagger UI: `http://localhost:8003/docs`
4. Contact development team

---

**Status:** ✅ Ready for Testing  
**Version:** 1.0.0  
**Last Updated:** February 7, 2026

# Mail Agent Canvas Integration

## Summary

Successfully integrated CanvasService with the Mail Agent to enable email preview displays in the orchestrator canvas.

## Changes Made

### 1. `backend/agents/mail_agent/agent.py`

**Added Canvas Preview Generation for Send Action:**

```python
elif "send" in step_action:
    # Build email preview canvas for user confirmation
    canvas = CanvasService.build_email_preview(
        to=step_params.get("to", []),
        subject=step_params.get("subject", ""),
        body=step_params.get("body", ""),
        cc=step_params.get("cc", []),
        requires_confirmation=True,
        confirmation_message=f"Confirm: Send email to {', '.join(step_params.get('to', []))}?",
    )
    
    res = await gmail_client.send_email_with_attachments(...)
    result = res.get("data")
    
    # Add canvas_display to result for extraction later
    if isinstance(result, dict):
        result["canvas_display"] = canvas.model_dump()
```

**Updated AgentResponse to Include Canvas:**

```python
return AgentResponse(
    status=AgentResponseStatus.COMPLETE,
    result={"results": results},
    standard_response=StandardAgentResponse(
        status="success",
        summary="Email composition complete. Ready to send.",
        data=results,
        canvas_display=last_canvas,  # Include canvas from send action
    ),
)
```

### 2. `backend/orchestrator/hands.py`

**Enhanced Canvas Extraction Logic:**

- Handles both dict and Pydantic object formats
- Extracts canvas_display from StandardAgentResponse
- Supports multiple canvas types: html, plan_graph, email_preview, spreadsheet

```python
# Handle both dict and Pydantic object responses
output = result.output
if isinstance(output, dict) and "standard_response" in output:
    std_response = output.get("standard_response")
elif hasattr(output, "standard_response") and output.standard_response:
    std_response = output.standard_response

# Handle both formats for canvas_display
if isinstance(std_response, dict) and "canvas_display" in std_response:
    canvas = std_response["canvas_display"]
elif hasattr(std_response, "canvas_display") and std_response.canvas_display:
    canvas = std_response.canvas_display
```

## Canvas Flow

```
User Request → Orchestrator → Mail Agent
                              ↓
                    CanvasService.build_email_preview()
                              ↓
                    StandardAgentResponse with canvas_display
                              ↓
                    Hands._update_state_with_result()
                              ↓
                    Orchestrator State (has_canvas=True)
                              ↓
                    Frontend Canvas Display
```

## Canvas Types Supported

| Canvas Type | Description | Current View |
|-------------|-------------|-------------|
| email_preview | Email preview with confirmation | browser |
| spreadsheet | Data table display | spreadsheet |
| html | Raw HTML content | browser |
| plan_graph | Execution plan visualization | plan |

## Testing

Run the test script:

```bash
source venv/bin/activate
python /home/clawuser/.openclaw/workspace/Orchestrator/test_mail_canvas_fixed.py
```

Expected output:
```
✅ CanvasService.build_email_preview() ✅ Working
✅ StandardAgentResponse includes canvas_display ✅ Working
✅ AgentResponse includes canvas in result ✅ Working
✅ Hands._update_state_with_result extracts canvas ✅ Working
```

## API Response Format

When Mail Agent returns a response with canvas:

```json
{
  "status": "complete",
  "result": {"results": [...]},
  "standard_response": {
    "status": "success",
    "summary": "Email composition complete. Ready to send.",
    "canvas_display": {
      "canvas_type": "email_preview",
      "canvas_data": {
        "to": ["a.anitej@gmail.com"],
        "subject": "Test Subject",
        "body": "Test body"
      },
      "requires_confirmation": true,
      "confirmation_message": "Send this email?"
    }
  }
}
```

## Notes

1. **Credentials Required**: Real email sending requires COMPOSIO_API_KEY
2. **Canvas Preview**: Even without credentials, canvas preview is generated
3. **Confirmation Flow**: Email previews have `requires_confirmation=true` for user approval

## Future Improvements

1. Add confirmation flow to actually send/reject emails
2. Support draft saving before sending
3. Add attachment preview in canvas
4. Test with real Gmail credentials


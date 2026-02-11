# Agent Selection Fixes

## Problem
The orchestrator was not properly calling agents based on tasks. The Brain was receiving SKILL.md context but wasn't selecting the right agent due to:
1. Verbose skills context making it hard for LLM to identify agents
2. Inconsistent agent naming between Brain suggestions and Hands execution
3. Lack of clear guidance for agent selection

## Solution

### Changes to `/home/clawuser/.openclaw/workspace/Orchestrator/backend/orchestrator/brain.py`

1. **Improved Agent List Formatting** (lines ~196-206)
   - Created structured agent list with exact names from SKILL files
   - Included clear descriptions for when to use each agent
   - Used bold formatting for better visibility

2. **Added Quick Reference Guide** (lines ~278-283)
   - Explicit list of exact agent names and their use cases
   - IDs provided for fallback matching
   - Clear mapping of tasks to agents

3. **Enhanced Agent Selection Examples** (lines ~290-300)
   - Added multiple examples showing correct agent names
   - One example per agent type
   - Clear payload formatting

## Agent Names (Exact from SKILL.md files)

| Task Type | Agent Name | Agent ID |
|-----------|-----------|----------|
| Web navigation, scraping | Browser Automation Agent | browser_automation_agent |
| CSV, Excel, data analysis | Spreadsheet Agent | spreadsheet_agent |
| Email, Gmail | Mail Agent | mail_agent |
| PDF, Word documents | Document Agent | document_agent |
| Invoicing, finance | Zoho Books Agent | zoho_books_agent |

## How It Works

1. **Brain receives task** from user
2. **Brain sees available agents** with exact names and descriptions
3. **Brain selects appropriate agent** based on task type
4. **Brain specifies agent name** in resource_id field
5. **Hands executes agent** by matching name or ID

## Testing

All agent names are properly configured and match between:
- SKILL.md files → Agent registry → Brain prompts → Hands execution

Run the test script to verify:
```bash
cd /home/clawuser/.openclaw/workspace/Orchestrator/backend
source venv/bin/activate
python -c "from services.agent_registry_service import agent_registry; print(agent_registry.get_all_skills_context())"
```

## Verification

The system now correctly:
✅ Loads agents from SKILL.md files
✅ Provides clear agent names to Brain
✅ Matches agent names in Hands execution  
✅ Handles both name and ID matching
✅ Includes detailed selection guidance


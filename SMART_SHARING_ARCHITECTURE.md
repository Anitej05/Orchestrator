# Smart File Sharing Architecture

## Who Knows About What?

### Knowledge Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │     ORCHESTRATOR (Brain)            │
                    │                                     │
                    │  Knows about BOTH workspaces:       │
                    │  ✓ Thread workspace (private)       │
                    │  ✓ Shared workspace (persistent)    │
                    │  ✓ Agent workspaces                 │
                    │                                     │
                    │  Can DECIDE when to share files     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │     HANDS (Execution)               │
                    │                                     │
                    │  Executes Python/Terminal           │
                    │  Saves files to thread workspace    │
                    │  Scans for new files                │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐       ┌────────▼───────┐    ┌────────▼──────┐
    │   AGENTS    │       │   SHARED WS    │    │   THREAD WS   │
    │             │       │                │    │               │
    │ Don't know  │       │ Persistent     │    │ Temporary     │
    │ about shared│       │ Cross-session  │    │ Per-session   │
    │ storage     │       │                │    │               │
    │             │       │ Only Brain     │    │ Auto-created  │
    │ Receive     │       │ can write here │    │ Auto-tracked  │
    │ files from  │       │                │    │               │
    │ Brain only  │       │                │    │               │
    └─────────────┘       └────────────────┘    └───────────────┘
```

## The Smart Sharing Flow

### Scenario 1: User Creates a Template
```
User: "Create an email template for invoice reminders"

Brain Decision:
├─ User is creating a "template"
├─ Templates should be REUSABLE
└─ Decision: SHARE this file

Execution:
1. Python creates: thread_workspace/invoice_template.txt
2. Brain detects sharing intent
3. Python copies to: shared_workspace/invoice_template.txt
4. Brain tells user: "I've saved this template for future use"

Result:
├─ Thread workspace: invoice_template.txt (temporary copy)
└─ Shared workspace: invoice_template.txt (permanent copy)
```

### Scenario 2: One-Time Analysis
```
User: "Analyze Tesla stock and create a chart"

Brain Decision:
├─ User wants analysis (temporary)
├─ No indication to save/persist
└─ Decision: DON'T SHARE

Execution:
1. Python creates: thread_workspace/tesla_chart.png
2. Brain keeps it private
3. Brain returns analysis with chart

Result:
├─ Thread workspace: tesla_chart.png (temporary)
└─ Shared workspace: (empty - not shared)
```

### Scenario 3: User Explicitly Saves
```
User: "Create a sales report... and save it for next quarter"

Brain Decision:
├─ User explicitly says "save it"
├─ Indicates persistence intent
└─ Decision: SHARE this file

Execution:
1. Python creates: thread_workspace/q4_sales_report.xlsx
2. Brain detects "save" keyword
3. Python copies to: shared_workspace/q4_sales_report.xlsx
4. Brain tells user: "Report saved to your persistent storage"

Result:
├─ Thread workspace: q4_sales_report.xlsx
└─ Shared workspace: q4_sales_report.xlsx (available next quarter!)
```

### Scenario 4: Agent Needs Shared File
```
User: "Email the report I created last week"

Brain Decision:
├─ User references "report from last week"
├─ Check shared workspace...
├─ Found: shared_workspace/weekly_report.pdf
├─ Need to share with Mail Agent
└─ Decision: Copy to agent workspace

Execution:
1. Brain finds file in shared workspace
2. Brain copies to: mail_agent_workspace/weekly_report.pdf
3. Brain tells Mail Agent: "Send this file to..."
4. Mail Agent sends email with attachment

Result:
├─ Shared workspace: weekly_report.pdf (permanent)
├─ Mail Agent workspace: weekly_report.pdf (temporary)
└─ Email sent successfully!
```

## Why This Architecture?

### 1. Orchestrator is the Gatekeeper
**Only the Brain decides when to share.**

Benefits:
- ✓ Users don't need to learn commands
- ✓ Context-aware decisions
- ✓ Natural language interface
- ✓ Intelligent defaults

### 2. Agents are Isolated
**Agents never access shared storage directly.**

Benefits:
- ✓ Security (agents can't leak files)
- ✓ Simplicity (agents don't need to know)
- ✓ Control (Brain decides what agents see)
- ✓ Clean boundaries

### 3. Persistent vs Temporary is Clear
**Two workspaces = clear mental model**

Benefits:
- ✓ Users understand "this conversation" vs "all conversations"
- ✓ No accidental pollution of persistent storage
- ✓ Easy to find important files
- ✓ Temporary files auto-cleaned eventually

## Smart Sharing Detection

The Brain uses context to decide:

```python
# Sharing triggers:
SHARE = True when:
- "save this" / "keep this" / "remember this"
- "create a template"
- "important" / "permanent" / "don't delete"
- "reuse" / "use again" / "next time"
- File name contains "template", "config", "settings"
- File type is .docx, .xlsx, .pdf (documents)

DON'T SHARE (default):
- Temporary analysis
- Charts and visualizations
- Downloaded files
- Scratch work
- File type .tmp, .cache, .log
```

## Examples of Smart Decisions

| User Request | Brain Decision | Why |
|--------------|----------------|-----|
| "Create a chart" | PRIVATE | Temporary visualization |
| "Create a template" | SHARED | Reusable resource |
| "Analyze data" | PRIVATE | One-time analysis |
| "Save this analysis" | SHARED | User explicitly wants to keep |
| "Download a file" | PRIVATE | Temporary download |
| "Create company logo" | SHARED | Important asset |
| "Generate random numbers" | PRIVATE | Temporary data |
| "Create email template" | SHARED | Reusable template |

## How Users Interact

### Natural Language (No Commands!)
```
❌ Old way: "/save_to_shared file.pdf"

✅ New way:
User: "Create a report and save it"
Orchestrator: "I've created the report and saved it to your persistent storage."

✅ Even simpler:
User: "Create a template for monthly reports"
Orchestrator: "I've created the template and saved it for future use."
```

### The Orchestrator Understands Context
```
User: "Create a quick chart of sales"
→ Brain: "Quick" = temporary → PRIVATE

User: "Create an important chart of sales for the board meeting"
→ Brain: "Important" + "board meeting" = keep → SHARED

User: "Analyze this and keep the results"
→ Brain: "Keep the results" = explicit save → SHARED
```

## Implementation Details

### File: `file_sharing.py`
Provides:
- `SharingIntentDetector` - Analyzes user prompts
- `should_share_file()` - Quick decision function
- `AgentWorkspaceInterface` - Shares files with agents

### File: `brain.py`
Updated prompt includes:
- When to share vs keep private
- How to copy files between workspaces
- Examples of sharing decisions

### No User Commands Needed!
The system is **intelligent** - it understands natural language and makes decisions.

## Benefits

1. **Zero Learning Curve**
   - Users just talk naturally
   - No commands to remember
   - No special syntax

2. **Context-Aware**
   - Understands "this is important"
   - Recognizes templates
   - Detects persistence intent

3. **Secure by Design**
   - Agents can't access shared storage
   - Brain controls all sharing
   - User isolation maintained

4. **Clean Architecture**
   - Clear separation of concerns
   - Simple mental model
   - Predictable behavior

## Summary

**The orchestrator is the only component that knows about both workspaces.**

- ✓ Brain decides when to share (intelligent)
- ✓ Agents are isolated (secure)
- ✓ Users use natural language (simple)
- ✓ Files persist when needed (useful)

This makes the orchestrator truly **smart** - it manages the complexity so users don't have to think about it!

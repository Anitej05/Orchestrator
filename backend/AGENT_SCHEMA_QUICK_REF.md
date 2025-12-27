# Agent Schema Quick Reference

## Overview

This is a quick reference guide for the new standardized agent schema. For complete details, see [AGENT_SCHEMA_SPEC.md](AGENT_SCHEMA_SPEC.md).

## Quick Start

### Creating a New Agent

1. **Choose the right template:**
   - `Agent_entries/templates/http_rest_agent.template.json` - For REST API agents
   - `Agent_entries/templates/mcp_http_agent.template.json` - For MCP servers
   - `Agent_entries/templates/tool_agent.template.json` - For direct function calls

2. **Copy and customize:**
   ```bash
   cp Agent_entries/templates/http_rest_agent.template.json Agent_entries/my_new_agent.json
   ```

3. **Fill in required fields:**
   - `id` - Unique identifier (kebab-case or snake_case)
   - `name` - Human-readable name
   - `description` - What the agent does
   - `agent_type` - `http_rest`, `mcp_http`, or `tool`
   - `version` - Semantic version (e.g., `1.0.0`)
   - `capabilities` - Structured capabilities (see below)

4. **Validate your agent:**
   ```bash
   python manage.py validate my_new_agent
   ```

5. **Sync to database:**
   ```bash
   python manage.py sync
   ```

## Agent Types

| Type | Use Case | Requires | Example |
|------|----------|----------|---------|
| `http_rest` | REST API integration | `connection_config`, `endpoints` | Mail Agent, Browser Agent |
| `mcp_http` | Model Context Protocol servers | `connection_config`, `endpoints` | Future: GitHub MCP, Notion MCP |
| `tool` | Direct Python functions | `tool_functions`, `tool_registry` | News, Wiki, Finance tools |

## Capability Structure

**Old (Flat):**
```json
"capabilities": [
  "read emails",
  "send emails",
  "search emails"
]
```

**New (Structured):**
```json
"capabilities": {
  "categories": [
    {
      "name": "Email Management",
      "description": "Core email operations",
      "priority": "high",
      "capabilities": [
        {
          "id": "read-emails",
          "name": "Read Emails",
          "description": "Fetch and display emails from inbox",
          "keywords": ["read emails", "fetch emails", "get inbox"],
          "requires_permission": true,
          "examples": [
            "Read my emails",
            "Show my inbox"
          ],
          "related_endpoints": ["/fetch_emails"]
        }
      ]
    }
  ],
  "all_keywords": ["read emails", "send emails", "search emails"]
}
```

## Management Commands

### Validation Commands

```bash
# Validate a single agent
python manage.py validate mail_agent

# Validate all agents
python manage.py validate-all

# Sync agents to database (with automatic validation)
python manage.py sync

# Sync quietly (less output)
python manage.py sync --quiet
```

### Migration Commands

```bash
# Migrate a single agent to new schema
python migrate_agent_schema.py Agent_entries/my_agent.json

# Migrate all agents
python migrate_agent_schema.py --all
```

## Common Patterns

### HTTP Agent Pattern

```json
{
  "id": "my-http-agent",
  "agent_type": "http_rest",
  "connection_config": {
    "base_url": "http://localhost:8080",
    "timeout": 30
  },
  "endpoints": [
    {
      "endpoint": "/action",
      "http_method": "POST",
      "description": "Perform action",
      "parameters": [
        {
          "name": "input",
          "param_type": "string",
          "required": true,
          "description": "Input parameter"
        }
      ]
    }
  ]
}
```

### TOOL Agent Pattern

```json
{
  "id": "my-tool-agent",
  "agent_type": "tool",
  "connection_config": null,
  "endpoints": null,
  "tool_functions": ["my_function", "another_function"],
  "tool_registry": {
    "module": "tools.my_tools",
    "functions": [
      {
        "name": "my_function",
        "description": "What this function does",
        "async": false,
        "parameters_schema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search query"
            }
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

### Agent with Credentials

```json
{
  "requires_credentials": true,
  "credential_fields": [
    {
      "name": "api_key",
      "label": "API Key",
      "type": "password",
      "required": true,
      "description": "Your API key",
      "placeholder": "sk_...",
      "validation": {
        "pattern": "^sk_[a-zA-Z0-9]{32,}$"
      }
    }
  ]
}
```

## Validation Rules

### ✅ Valid IDs
- `mail-agent` (kebab-case)
- `mail_agent` (snake_case)
- `document-analysis-agent`

### ❌ Invalid IDs
- `Mail_Agent` (PascalCase)
- `mailAgent` (camelCase)
- `mail agent` (spaces)

### ✅ Valid Versions
- `1.0.0`
- `2.1.3`
- `10.0.0`

### ❌ Invalid Versions
- `1.0` (missing patch)
- `v1.0.0` (no prefix)
- `1.0.0-beta` (no suffix)

## Error Messages

### "Agent ID must be kebab-case or snake_case"
**Fix:** Change ID to use only lowercase, numbers, hyphens, or underscores
```json
// ❌ Wrong
"id": "Mail_Agent"

// ✅ Correct
"id": "mail-agent"
```

### "HTTP_REST agents require connection_config"
**Fix:** Add connection_config with base_url
```json
// ❌ Wrong
"connection_config": null

// ✅ Correct
"connection_config": {
  "base_url": "http://localhost:8080"
}
```

### "TOOL agents cannot have endpoints"
**Fix:** Set endpoints to null for TOOL agents
```json
// ❌ Wrong
"agent_type": "tool",
"endpoints": [...]

// ✅ Correct
"agent_type": "tool",
"endpoints": null
```

### "Capability must have at least 2 keywords"
**Fix:** Add more keywords to capabilities
```json
// ❌ Wrong
"keywords": ["read"]

// ✅ Correct
"keywords": ["read", "fetch", "get emails"]
```

## Best Practices

### 1. Use Clear, Descriptive Names
```json
// ❌ Poor
"name": "Agent"

// ✅ Good
"name": "Gmail Integration Agent"
```

### 2. Group Related Capabilities
```json
{
  "categories": [
    {
      "name": "Email Reading",
      "capabilities": [...]
    },
    {
      "name": "Email Sending",
      "capabilities": [...]
    }
  ]
}
```

### 3. Provide Comprehensive Examples
```json
"examples": [
  "Read my unread emails",
  "Show me emails from john@example.com",
  "What emails did I get today?"
]
```

### 4. Use Semantic Versioning
- **1.0.0** → **1.0.1** - Bug fixes
- **1.0.0** → **1.1.0** - New features
- **1.0.0** → **2.0.0** - Breaking changes

### 5. Document Prerequisites
```json
"description": "Gmail agent using Composio API. Requires Composio account and Gmail connection."
```

## File Structure

```
backend/
├── AGENT_SCHEMA_SPEC.md          # Complete specification
├── AGENT_SCHEMA_QUICK_REF.md     # This file
├── agent_schemas.py               # Pydantic validation models
├── migrate_agent_schema.py        # Migration script
├── manage.py                      # Management commands
└── Agent_entries/
    ├── templates/
    │   ├── http_rest_agent.template.json
    │   ├── mcp_http_agent.template.json
    │   └── tool_agent.template.json
    ├── mail_agent.json
    ├── document_analysis_agent.json
    ├── spreadsheet_agent.json
    ├── browser_automation_agent.json
    └── zoho_books_agent.json
```

## Resources

- **Full Specification:** [AGENT_SCHEMA_SPEC.md](AGENT_SCHEMA_SPEC.md)
- **Templates:** [Agent_entries/templates/](Agent_entries/templates/)
- **Examples:** All agents in [Agent_entries/](Agent_entries/)
- **Validation Module:** [agent_schemas.py](agent_schemas.py)
- **Migration Tool:** [migrate_agent_schema.py](migrate_agent_schema.py)

## Troubleshooting

### Agent fails validation
1. Run validation to see specific errors:
   ```bash
   python manage.py validate agent_id
   ```
2. Check error message and fix accordingly
3. Re-validate until it passes
4. Sync to database

### Agent doesn't appear in UI
1. Ensure agent passes validation
2. Run sync command:
   ```bash
   python manage.py sync
   ```
3. Check for errors in sync output
4. Restart backend if needed

### Capabilities not matching user queries
1. Add more keywords to capabilities
2. Use examples that match actual user language
3. Ensure all_keywords includes relevant terms
4. Re-sync agent

## Summary

**Before creating/editing an agent:**
1. Choose appropriate template
2. Follow naming conventions (kebab-case/snake_case)
3. Structure capabilities properly
4. Add comprehensive keywords and examples
5. Validate before syncing

**Always validate before syncing:**
```bash
python manage.py validate my_agent && python manage.py sync
```

**When in doubt, check existing agents:**
```bash
cat Agent_entries/mail_agent.json
```

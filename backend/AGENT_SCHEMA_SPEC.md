# Agent Schema Specification

## Overview

This document defines the **universal agent schema** for Orbimesh agents. All agents (HTTP, MCP, TOOL) must follow this standardized structure to ensure consistency, maintainability, and ease of integration.

## Design Principles

1. **Type Safety**: All fields have clear types and validation rules
2. **Extensibility**: Schema can accommodate new agent types without breaking changes
3. **Self-Documenting**: Schema includes enough metadata to understand agent capabilities
4. **Validation First**: All agents are validated before being added to the system
5. **Structured Capabilities**: Capabilities are organized by category with proper metadata

---

## Agent Types

### Supported Agent Types

```python
class AgentType(str, enum.Enum):
    HTTP_REST = "http_rest"    # Legacy REST API agents using OpenAPI
    MCP_HTTP = "mcp_http"      # Model Context Protocol servers over HTTP
    TOOL = "tool"              # Direct Python function calls via LangChain @tool
```

### Type-Specific Requirements

| Agent Type | Requires Endpoints | Requires Connection Config | Requires Tool Functions |
|-----------|-------------------|---------------------------|------------------------|
| `HTTP_REST` | ✓ Yes | ✓ Yes (base_url) | ✗ No |
| `MCP_HTTP` | ✓ Yes | ✓ Yes (base_url, transport) | ✗ No |
| `TOOL` | ✗ No | ✗ No | ✓ Yes (function registry) |

---

## Core Schema Structure

### Top-Level Fields

All agent definitions must include these top-level fields:

```typescript
{
  // REQUIRED: Core Identity
  "id": string,                    // Unique identifier (kebab-case)
  "owner_id": string,              // Owner identifier (e.g., "orbimesh-vendor")
  "name": string,                  // Human-readable name
  "description": string,           // Detailed description of agent purpose
  "agent_type": AgentType,         // "http_rest" | "mcp_http" | "tool"
  "version": string,               // Semantic versioning (e.g., "1.0.0")
  
  // REQUIRED: Capabilities
  "capabilities": CapabilityStructure,  // Structured capabilities (see below)
  
  // REQUIRED: Status & Pricing
  "status": string,                // "active" | "inactive" | "deprecated"
  "price_per_call_usd": number,    // Cost per agent call
  
  // OPTIONAL: Authentication
  "requires_credentials": boolean, // Does this agent need credentials?
  "credential_fields": CredentialField[], // Array of credential definitions
  
  // TYPE-SPECIFIC: Connection Configuration
  "connection_config": ConnectionConfig | null,  // For HTTP/MCP agents
  
  // TYPE-SPECIFIC: Endpoints
  "endpoints": Endpoint[] | null,  // For HTTP/MCP agents
  
  // TYPE-SPECIFIC: Tool Registry
  "tool_functions": string[] | null,  // For TOOL agents (function names)
  
  // OPTIONAL: Metadata
  "tags": string[],                // Searchable tags
  "category": string,              // Primary category (e.g., "automation", "analysis")
  "icon": string,                  // Icon identifier or URL
  "documentation_url": string,     // Link to full documentation
  
  // INTERNAL: Signatures
  "public_key_pem": string         // Public key for verification
}
```

---

## Capability Structure

### Problem: Current State
**Old**: Flat array of strings
```json
"capabilities": [
  "read emails",
  "send emails",
  "search emails",
  ...
]
```

**Issues**:
- No categorization
- No priority/importance
- No permission requirements
- Hard to search/filter
- No semantic structure

### Solution: Structured Capabilities

```typescript
interface CapabilityStructure {
  categories: CapabilityCategory[];
  all_keywords: string[];  // Flat list for backward compatibility
}

interface CapabilityCategory {
  name: string;              // Category name (e.g., "Email Management")
  description: string;       // What this category does
  priority: "high" | "medium" | "low";
  capabilities: Capability[];
}

interface Capability {
  id: string;                // Unique capability ID (kebab-case)
  name: string;              // Human-readable name
  description: string;       // What this capability does
  keywords: string[];        // Search keywords
  requires_permission: boolean;  // Does this need special permissions?
  examples: string[];        // Example queries that trigger this
  related_endpoints?: string[];  // For HTTP/MCP: which endpoints provide this
  related_functions?: string[];  // For TOOL: which functions provide this
}
```

### Example: Mail Agent Capabilities

```json
{
  "capabilities": {
    "categories": [
      {
        "name": "Email Reading",
        "description": "Read and fetch emails from inbox",
        "priority": "high",
        "capabilities": [
          {
            "id": "read-emails",
            "name": "Read Emails",
            "description": "Fetch and display emails from Gmail inbox",
            "keywords": ["read emails", "fetch emails", "get inbox", "check email"],
            "requires_permission": true,
            "examples": [
              "Read my emails",
              "Show me my inbox",
              "What emails do I have?"
            ],
            "related_endpoints": ["/fetch_emails"]
          },
          {
            "id": "search-emails",
            "name": "Search Emails",
            "description": "Search emails using Gmail query syntax",
            "keywords": ["search emails", "find emails", "filter inbox"],
            "requires_permission": true,
            "examples": [
              "Find emails from john@example.com",
              "Search for emails about 'project update'"
            ],
            "related_endpoints": ["/fetch_emails"]
          }
        ]
      },
      {
        "name": "Email Sending",
        "description": "Compose and send emails",
        "priority": "high",
        "capabilities": [
          {
            "id": "send-email",
            "name": "Send Email",
            "description": "Send new emails with optional attachments",
            "keywords": ["send email", "compose email", "write email"],
            "requires_permission": true,
            "examples": [
              "Send an email to jane@example.com",
              "Compose a message about the meeting"
            ],
            "related_endpoints": ["/send_email"]
          }
        ]
      },
      {
        "name": "Attachment Management",
        "description": "Handle email attachments",
        "priority": "medium",
        "capabilities": [
          {
            "id": "download-attachments",
            "name": "Download Attachments",
            "description": "Download and save email attachments",
            "keywords": ["download attachments", "get files", "save attachments"],
            "requires_permission": true,
            "examples": [
              "Download the attachments from that email",
              "Save the PDF attachment"
            ],
            "related_endpoints": ["/get_attachments"]
          }
        ]
      }
    ],
    "all_keywords": [
      "read emails", "send emails", "search emails", "fetch emails",
      "compose emails", "download attachments", "get attachments"
    ]
  }
}
```

---

## Connection Configuration

Type-specific configuration for HTTP and MCP agents.

### HTTP_REST Configuration

```typescript
interface HttpRestConfig {
  base_url: string;           // Base URL for all endpoints
  timeout?: number;           // Request timeout in seconds (default: 30)
  headers?: Record<string, string>;  // Default headers
  retry_config?: {
    max_retries: number;
    backoff_factor: number;
  }
}
```

Example:
```json
{
  "connection_config": {
    "base_url": "http://localhost:8040",
    "timeout": 30,
    "headers": {
      "Content-Type": "application/json"
    },
    "retry_config": {
      "max_retries": 3,
      "backoff_factor": 2
    }
  }
}
```

### MCP_HTTP Configuration

```typescript
interface McpHttpConfig {
  base_url: string;           // Base URL for MCP server
  transport: "sse" | "http";  // Transport method
  timeout?: number;
  headers?: Record<string, string>;
}
```

Example:
```json
{
  "connection_config": {
    "base_url": "http://localhost:3000",
    "transport": "sse",
    "timeout": 60
  }
}
```

### TOOL Configuration

TOOL agents don't need connection config (they're local functions).

```json
{
  "connection_config": null
}
```

---

## Endpoint Structure

For HTTP_REST and MCP_HTTP agents only.

```typescript
interface Endpoint {
  endpoint: string;           // Endpoint path (e.g., "/send_email")
  http_method: string;        // HTTP method: GET, POST, PUT, DELETE
  description: string;        // What this endpoint does
  request_format?: string;    // Optional: "json" | "form-data"
  parameters: Parameter[];    // Array of parameters
  response_schema?: object;   // Optional: JSON schema for response
}

interface Parameter {
  name: string;               // Parameter name
  param_type: string;         // "string" | "integer" | "boolean" | "array" | "object"
  required: boolean;          // Is this parameter required?
  description: string;        // What this parameter does
  default_value?: any;        // Optional default value
  validation?: {
    min?: number;             // For numbers/arrays
    max?: number;
    pattern?: string;         // Regex pattern for strings
    enum?: string[];          // Allowed values
  }
}
```

### Example: Mail Agent Endpoint

```json
{
  "endpoints": [
    {
      "endpoint": "/send_email",
      "http_method": "POST",
      "description": "Send a new email via Gmail with optional attachments and HTML formatting",
      "request_format": "json",
      "parameters": [
        {
          "name": "to",
          "param_type": "array",
          "required": true,
          "description": "Array of recipient email addresses",
          "validation": {
            "min": 1
          }
        },
        {
          "name": "subject",
          "param_type": "string",
          "required": true,
          "description": "Email subject line"
        },
        {
          "name": "body",
          "param_type": "string",
          "required": true,
          "description": "Email body content (plain text or HTML)"
        },
        {
          "name": "is_html",
          "param_type": "boolean",
          "required": false,
          "description": "Whether body is HTML formatted",
          "default_value": false
        }
      ],
      "response_schema": {
        "type": "object",
        "properties": {
          "success": {"type": "boolean"},
          "message_id": {"type": "string"},
          "thread_id": {"type": "string"}
        }
      }
    }
  ]
}
```

---

## Credential Configuration

For agents that require authentication.

```typescript
interface CredentialField {
  name: string;               // Internal field name (snake_case)
  label: string;              // Display label for users
  type: "text" | "password" | "url" | "select";
  required: boolean;
  description: string;        // Help text for users
  placeholder?: string;       // Example value
  validation?: {
    pattern?: string;         // Regex validation
    enum?: string[];          // For select fields
  }
}
```

### Example: Mail Agent Credentials

```json
{
  "requires_credentials": true,
  "credential_fields": [
    {
      "name": "composio_api_key",
      "label": "Composio API Key",
      "type": "password",
      "required": true,
      "description": "Your Composio API key from https://app.composio.dev/settings/api-keys",
      "placeholder": "ak_...",
      "validation": {
        "pattern": "^ak_[a-zA-Z0-9]{32,}$"
      }
    },
    {
      "name": "gmail_connection_id",
      "label": "Gmail Connection ID",
      "type": "text",
      "required": true,
      "description": "Your Gmail connection ID from Composio dashboard",
      "placeholder": "Gmail-xxxxx",
      "validation": {
        "pattern": "^Gmail-[a-zA-Z0-9]+$"
      }
    }
  ]
}
```

---

## Tool Function Registry

For TOOL type agents only.

```typescript
interface ToolRegistry {
  module: string;             // Python module path (e.g., "tools.news_tools")
  functions: ToolFunction[];
}

interface ToolFunction {
  name: string;               // Function name
  description: string;        // What this function does
  async: boolean;             // Is this an async function?
  parameters_schema: object;  // JSON schema for parameters
}
```

### Example: News Agent (TOOL Type)

```json
{
  "agent_type": "tool",
  "tool_functions": ["search_news", "get_top_headlines"],
  "tool_registry": {
    "module": "tools.news_tools",
    "functions": [
      {
        "name": "search_news",
        "description": "Search for news articles using NewsAPI",
        "async": false,
        "parameters_schema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search query for news articles"
            },
            "language": {
              "type": "string",
              "description": "Language code (e.g., 'en', 'es')",
              "default": "en"
            },
            "page_size": {
              "type": "integer",
              "description": "Number of articles to return (max 100)",
              "default": 10
            }
          },
          "required": ["query"]
        }
      },
      {
        "name": "get_top_headlines",
        "description": "Get top headlines from NewsAPI",
        "async": false,
        "parameters_schema": {
          "type": "object",
          "properties": {
            "country": {
              "type": "string",
              "description": "Country code (e.g., 'us', 'gb')",
              "default": "us"
            },
            "page_size": {
              "type": "integer",
              "description": "Number of headlines to return (max 100)",
              "default": 10
            }
          },
          "required": []
        }
      }
    ]
  }
}
```

---

## Validation Rules

### General Rules

1. **ID Format**: Must be kebab-case (lowercase with hyphens)
   - ✓ Valid: `mail-agent`, `document-analysis-agent`
   - ✗ Invalid: `Mail_Agent`, `documentAgent`

2. **Version Format**: Must follow semantic versioning (MAJOR.MINOR.PATCH)
   - ✓ Valid: `1.0.0`, `2.1.3`
   - ✗ Invalid: `v1.0`, `1.0`

3. **Status Values**: Must be one of: `active`, `inactive`, `deprecated`

4. **Agent Type**: Must be one of: `http_rest`, `mcp_http`, `tool`

### Type-Specific Rules

#### HTTP_REST Agents
- ✓ Must have `connection_config` with `base_url`
- ✓ Must have at least one endpoint in `endpoints` array
- ✓ `tool_functions` must be `null` or omitted
- ✓ All endpoints must have valid `http_method`: GET, POST, PUT, DELETE, PATCH

#### MCP_HTTP Agents
- ✓ Must have `connection_config` with `base_url` and `transport`
- ✓ Must have at least one endpoint in `endpoints` array
- ✓ `tool_functions` must be `null` or omitted
- ✓ `transport` must be `sse` or `http`

#### TOOL Agents
- ✓ Must have `tool_functions` array with at least one function name
- ✓ `connection_config` must be `null` or omitted
- ✓ `endpoints` must be `null` or omitted
- ✓ All listed functions must exist in `tools/` directory

### Capability Rules

1. **Category Names**: Must be unique within an agent
2. **Capability IDs**: Must be unique within an agent, kebab-case
3. **Keywords**: At least 2 keywords per capability
4. **Examples**: At least 1 example per capability
5. **Related References**: Must reference existing endpoints or functions

### Credential Rules

1. **Field Names**: Must be snake_case
2. **Field Types**: Must be one of: `text`, `password`, `url`, `select`
3. **Required Select Fields**: Must have `validation.enum` array
4. **Password Fields**: Should use secure storage (encrypted)

---

## Schema Migration Guide

### Migrating Old Agents to New Schema

#### Step 1: Add Version and Agent Type
```json
{
  "id": "mail-agent",
  "version": "1.0.0",        // ADD THIS
  "agent_type": "http_rest"  // ADD THIS (or mcp_http/tool)
}
```

#### Step 2: Restructure Capabilities

**Before:**
```json
{
  "capabilities": [
    "read emails",
    "send emails",
    "search emails"
  ]
}
```

**After:**
```json
{
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
            "description": "Fetch and display emails",
            "keywords": ["read emails", "fetch emails", "get inbox"],
            "requires_permission": true,
            "examples": ["Read my emails", "Show inbox"],
            "related_endpoints": ["/fetch_emails"]
          }
        ]
      }
    ],
    "all_keywords": ["read emails", "send emails", "search emails"]
  }
}
```

#### Step 3: Add Metadata
```json
{
  "tags": ["email", "gmail", "communication"],
  "category": "communication",
  "documentation_url": "https://docs.orbimesh.com/agents/mail"
}
```

#### Step 4: Validate Against Schema
Run the validation script:
```bash
python manage.py validate-agent-schema agent_id
```

---

## Template Files

### Template: HTTP_REST Agent

See [templates/http_rest_agent.template.json](templates/http_rest_agent.template.json)

### Template: MCP_HTTP Agent

See [templates/mcp_http_agent.template.json](templates/mcp_http_agent.template.json)

### Template: TOOL Agent

See [templates/tool_agent.template.json](templates/tool_agent.template.json)

---

## Validation Implementation

### Pydantic Models

All agent schemas are validated using Pydantic models defined in `backend/agent_schemas.py`:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class AgentType(str, Enum):
    HTTP_REST = "http_rest"
    MCP_HTTP = "mcp_http"
    TOOL = "tool"

class Capability(BaseModel):
    id: str = Field(..., pattern="^[a-z0-9-]+$")
    name: str
    description: str
    keywords: List[str] = Field(..., min_items=2)
    requires_permission: bool
    examples: List[str] = Field(..., min_items=1)
    related_endpoints: Optional[List[str]] = None
    related_functions: Optional[List[str]] = None

class CapabilityCategory(BaseModel):
    name: str
    description: str
    priority: str = Field(..., pattern="^(high|medium|low)$")
    capabilities: List[Capability] = Field(..., min_items=1)

class CapabilityStructure(BaseModel):
    categories: List[CapabilityCategory] = Field(..., min_items=1)
    all_keywords: List[str] = Field(..., min_items=1)

class AgentSchema(BaseModel):
    id: str = Field(..., pattern="^[a-z0-9-]+$")
    owner_id: str
    name: str
    description: str
    agent_type: AgentType
    version: str = Field(..., pattern="^\\d+\\.\\d+\\.\\d+$")
    capabilities: CapabilityStructure
    status: str = Field(..., pattern="^(active|inactive|deprecated)$")
    price_per_call_usd: float = Field(..., ge=0)
    # ... more fields
```

### Validation Commands

```bash
# Validate a single agent
python manage.py validate-agent mail-agent

# Validate all agents
python manage.py validate-all-agents

# Migrate agent to new schema
python manage.py migrate-agent-schema mail-agent
```

---

## Best Practices

### 1. Capability Organization
- Group related capabilities together
- Use clear, user-friendly category names
- Prioritize categories appropriately (high/medium/low)
- Include comprehensive keywords for matching

### 2. Endpoint Design
- Keep endpoints focused on single responsibilities
- Use clear, RESTful naming conventions
- Document all parameters thoroughly
- Include response schemas for better error handling

### 3. Credential Management
- Use descriptive labels and helpful placeholder text
- Add validation patterns to catch common errors early
- Link to credential setup instructions in descriptions
- Never store credentials in agent definitions

### 4. Documentation
- Provide at least 2-3 examples per capability
- Link to external documentation when available
- Keep descriptions concise but informative
- Update documentation when capabilities change

### 5. Versioning
- Follow semantic versioning strictly
- Increment MAJOR version for breaking changes
- Increment MINOR version for new features
- Increment PATCH version for bug fixes

---

## Common Patterns

### Pattern 1: Multi-Modal Agent
Agents that handle multiple data types (text, images, files)

```json
{
  "capabilities": {
    "categories": [
      {
        "name": "Text Processing",
        "capabilities": [...]
      },
      {
        "name": "Image Processing",
        "capabilities": [...]
      },
      {
        "name": "File Management",
        "capabilities": [...]
      }
    ]
  }
}
```

### Pattern 2: Agent with Optional Features
Some capabilities require credentials, others don't

```json
{
  "capabilities": {
    "categories": [
      {
        "name": "Public Features",
        "capabilities": [
          {
            "id": "search-wikipedia",
            "requires_permission": false,
            ...
          }
        ]
      },
      {
        "name": "Premium Features",
        "capabilities": [
          {
            "id": "api-search",
            "requires_permission": true,
            ...
          }
        ]
      }
    ]
  }
}
```

### Pattern 3: Tool Agent with Related Functions
Multiple functions that work together

```json
{
  "agent_type": "tool",
  "tool_functions": ["search_news", "get_top_headlines", "get_sources"],
  "capabilities": {
    "categories": [
      {
        "name": "News Discovery",
        "capabilities": [
          {
            "id": "search-news",
            "related_functions": ["search_news", "get_sources"]
          },
          {
            "id": "get-headlines",
            "related_functions": ["get_top_headlines"]
          }
        ]
      }
    ]
  }
}
```

---

## Troubleshooting

### Common Validation Errors

**Error**: `Agent ID must be kebab-case`
- **Fix**: Change `Mail_Agent` to `mail-agent`

**Error**: `Agent type 'http_rest' requires connection_config`
- **Fix**: Add `connection_config` with `base_url` field

**Error**: `Capability ID 'Read Emails' must be kebab-case`
- **Fix**: Change to `read-emails`

**Error**: `Tool agent cannot have endpoints`
- **Fix**: Remove `endpoints` field or set to `null`

**Error**: `Capability must have at least 2 keywords`
- **Fix**: Add more keywords: `["read", "fetch", "get emails"]`

**Error**: `Version '1.0' does not match semantic versioning`
- **Fix**: Change to `1.0.0`

---

## Future Extensions

### Planned Features

1. **Dependency Management**: Specify agent dependencies
2. **Rate Limiting**: Define rate limits per endpoint
3. **Health Checks**: Built-in health check endpoints
4. **Monitoring**: Performance metrics and logging configuration
5. **Access Control**: Role-based access control per capability
6. **Localization**: Multi-language support for names/descriptions

---

## Summary

This schema provides a robust, extensible foundation for all Orbimesh agents:

✅ **Type-Safe**: Clear types and validation rules
✅ **Flexible**: Supports HTTP, MCP, and TOOL agents
✅ **Structured**: Organized capabilities with categories
✅ **Self-Documenting**: Rich metadata and examples
✅ **Validated**: Pydantic models ensure correctness
✅ **Extensible**: Easy to add new agent types and features

**Rule of Thumb**: If you're creating a new agent, start with the appropriate template, fill in all required fields, validate against the schema, and sync to the database. The schema will guide you through the process and catch errors early.

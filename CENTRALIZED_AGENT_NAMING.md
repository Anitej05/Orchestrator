# Centralized Agent Naming - Standardized Solution

## Problem
Previous approach used if/else statements scattered across Brain and Hands to identify agent names, leading to:
- Inconsistent naming between components
- Maintenance nightmare (changes needed in multiple places)
- No single source of truth for agent aliases

## Solution: Centralized Naming Registry

### 1. Centralized Alias Mapping (agent_registry_service.py)

AGENT_ALIASES Dictionary with 23 standardized aliases for all agents.

### 2. Standardized Functions

- normalize_agent_name(): Converts any alias to canonical name
- find_agent(): Centralized agent lookup used by Brain and Hands
- get_canonical_name(): Get canonical name from any input

### 3. Updated Components

- Brain: Uses centralized agent list building
- Hands: Uses agent_registry.find_agent() for all lookups

## Benefits

1. Single Source of Truth for all agent aliases
2. Consistent naming across Brain and Hands
3. Easy to add/remove aliases in one place
4. No if/else statements in Brain or Hands

## Agent Names Reference

| Canonical Name | Agent ID | Aliases |
|----------------|----------|---------|
| Browser Automation Agent | browser_automation_agent | browser, web_agent, web_browser |
| Spreadsheet Agent | spreadsheet_agent | spreadsheet, excel, csv, data_agent |
| Mail Agent | mail_agent | mail, email, gmail |
| Document Agent | document_agent | document, pdf, word, docx |
| Zoho Books Agent | zoho_books_agent | zoho, accounting, invoice |


---
id: coding_agent
name: coding_agent
port: 8080
version: 1.0.0
description: Priority agent for coding, software engineering, multi-file editing, and test execution tasks.
---

# Coding Agent

**PRIORITY AGENT** for ALL coding, development, and software engineering tasks.

Powered by [OpenCode](https://opencode.ai) headless server — a full-featured AI coding agent
with codebase awareness, LSP integration, multi-file editing, and test execution.
Auto-installs the OpenCode CLI (`npm i -g opencode-ai@latest`) on first use.

## Capabilities

### Code Modification
- **code_task** — Write, edit, refactor, debug code across multiple files. Returns diffs for user approval.
- **debug** — Analyze tracebacks, identify root cause, optionally apply fixes automatically.

### Code Analysis (Read-Only)
- **review_code** — Code review, security analysis, performance audit. Does NOT modify files.
- **explain_code** — Explain files, functions, classes, patterns, or concepts. Rich markdown output.
- **search_codebase** — Search for patterns, functions, classes, or files across the project.

### Documentation & Preview
- **generate_docs** — Generate README, API docs, docstrings. Supports markdown and live HTML preview.
- **generate_preview** — Generate HTML/React/CSS pages rendered as live iframe in the canvas.

### DevOps
- **run_tests** — Execute project test suite, report pass/fail with terminal output.
- **git_operations** — Git status, diff, log, branch info. Read-only version control inspection.

## Canvas Integration

This agent has full access to the Orbimesh Canvas system. Output is automatically
routed to the best visual format:

| Output Type | Canvas Rendering |
|---|---|
| File diffs | Multi-file diff viewer with syntax highlighting + apply/reject buttons |
| Markdown analysis | Rich markdown document |
| HTML/React generated | **Live iframe preview** (self-contained HTML) |
| JSON data | Collapsible JSON tree viewer |
| Test results | Terminal output with pass/fail indicators |
| Git output | Syntax-highlighted code viewer |

The agent can also use `build_dynamic_canvas()` to generate any registered canvas template
on the fly (charts, spreadsheets, images, etc.).

## When to Use

Use this agent when the user:
- Asks to write, edit, or create code files
- Wants to fix bugs or debug errors/tracebacks
- Requests code refactoring or improvement
- Needs tests written or executed
- Asks for code review, explanation, or documentation
- Wants multi-file changes (e.g., "add authentication to the app")
- Wants a live HTML/UI preview in the canvas
- Needs git status, diff, or log information
- Wants to search the codebase for patterns or functions
- Mentions programming languages, frameworks, or development tools

## NOT For

- Quick one-off calculations → use Python sandbox
- Data analysis of CSV/Excel files → use Spreadsheet Agent
- Web browsing or scraping → use Browser Agent
- Email operations → use Mail Agent
- Document reading/creation → use Document Agent

## Action Routing

| Action | Aliases | Capability |
|---|---|---|
| `code_task` | *(default)* | Write/edit code |
| `review_code` | `review` | Analyze code |
| `run_tests` | `test` | Run tests |
| `debug` | `fix` | Debug errors |
| `explain_code` | `explain` | Explain code |
| `generate_docs` | `docs` | Generate docs |
| `git_operations` | `git` | Git info |
| `search_codebase` | `search` | Search code |
| `generate_preview` | `preview` | HTML preview |

## Example Prompts

- "Fix the authentication bug in the login endpoint"
- "Create a REST API for user management with CRUD operations"
- "Refactor the database module to use connection pooling"
- "Write unit tests for the payment service"
- "Review the security of auth/middleware.py"
- "Explain how the orchestrator's Brain works"
- "Generate API documentation for the agents module"
- "Show me a live preview of a login form with dark theme"
- "Show git log for the last 10 commits"
- "Search for all usages of CanvasDisplay in the codebase"
- "Debug this traceback: ModuleNotFoundError: No module named 'xyz'"

---
id: coding_agent
name: Coding Agent
port: 8080
version: 1.0.0
description: >
  Priority agent for coding, software engineering, multi-file editing,
  test execution, and code review. Powered by OpenCode headless server.
model: ollama/minimax-m2.5:cloud
context_strategy: standard
requires_auth: false
triggers:
  - code
  - programming
  - debug
  - fix bug
  - refactor
  - write function
  - create api
  - unit test
  - code review
  - git
  - html preview
  - software
  - developer
  - traceback
  - error
capabilities:
  - code_task
  - review_code
  - run_tests
  - debug
  - explain_code
  - generate_docs
  - git_operations
  - search_codebase
  - generate_preview
not_for:
  - quick calculations
  - data analysis of CSV/Excel
  - web browsing
  - email operations
  - document reading
---

# Coding Agent

**PRIORITY AGENT** for ALL coding, development, and software engineering tasks.

Powered by [OpenCode](https://opencode.ai) headless server — a full-featured AI coding agent
with codebase awareness, LSP integration, multi-file editing, and test execution.

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

## When to Use

Use this agent when the user:
- Asks to write, edit, or create code files
- Wants to fix bugs or debug errors/tracebacks
- Requests code refactoring or improvement
- Needs tests written or executed
- Asks for code review, explanation, or documentation
- Wants multi-file changes
- Wants a live HTML/UI preview in the canvas
- Needs git status, diff, or log information
- Wants to search the codebase for patterns or functions

## NOT For

- Quick one-off calculations → use Python sandbox
- Data analysis of CSV/Excel files → use Spreadsheet Agent
- Web browsing or scraping → use Browser Agent
- Email operations → use Mail Agent
- Document reading/creation → use Document Agent

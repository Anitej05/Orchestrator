---
id: browser_automation_agent
name: Browser Automation Agent
port: 8090
version: 1.0.0
---

# Browser Automation Agent

LLM-driven web browser automation for navigation and data extraction.

## Capabilities

- Navigate to any website URL
- Click elements, fill forms, submit data
- Extract structured data from web pages
- Take screenshots of browser state
- Multi-step web automation workflows
- Vision capabilities for understanding page layout

## When to Use

Use this agent when the user:
- Wants to navigate to a website
- Needs to interact with web pages (click, type, scroll)
- Asks to extract data from websites
- Mentions web scraping or automation
- Wants to fill out online forms
- Needs screenshots of web content

## NOT For

- Local files (CSV, Excel) → use Spreadsheet Agent
- PDF/Word documents → use Document Agent
- Emails → use Mail Agent
- Running Python code → use Python Sandbox

## Example Prompts

- "Go to leetcode.com and find upcoming contests"
- "Navigate to amazon.com and search for 'laptop stand'"
- "Fill out the contact form on example.com"
- "Extract all product prices from this shopping page"
- "Take a screenshot of the current page"

## Notes

- Uses Playwright for browser control
- Supports multi-provider LLM fallback (Cerebras → Groq → NVIDIA)
- Vision analysis via Ollama or NVIDIA
- Cannot access local filesystem or documents

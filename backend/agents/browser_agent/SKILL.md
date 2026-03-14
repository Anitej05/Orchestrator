---
id: browser_agent
name: Browser Agent
port: 8090
version: 1.0.0
description: >
  LLM-driven web browser automation for navigation, interaction, 
  data extraction, and screenshots using Playwright.
model: ollama/kimi-k2.5:cloud
context_strategy: minimal
requires_auth: false
triggers:
  - website
  - browse
  - navigate
  - web page
  - click
  - fill form
  - scrape
  - screenshot
  - web search
  - url
  - http
capabilities:
  - navigate_to_url
  - click_element
  - fill_form
  - extract_data
  - take_screenshot
  - multi_step_automation
not_for:
  - local files
  - CSV or Excel
  - PDF documents
  - emails
  - running Python code
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

## Notes

- Uses Playwright for browser control
- Uses centralized Inference Service with multi-provider LLM fallback
- Vision analysis via Inference Service (multi-provider)
- Cannot access local filesystem or documents

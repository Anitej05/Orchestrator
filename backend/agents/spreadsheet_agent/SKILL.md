---
id: spreadsheet_agent
name: Spreadsheet Agent
port: 9000
version: 3.0.0
---

# Spreadsheet Agent

**PRIORITY AGENT** for ALL CSV and Excel files (.csv, .xlsx, .xls).

## Capabilities

- Load and analyze spreadsheets from file paths or uploads
- Natural language queries ("what is the average revenue by region?")
- Data transformations: filter, sort, aggregate, pivot
- Column operations: add, drop, rename, transform
- Export to different formats
- Multi-turn dialogues for clarification

## When to Use

Use this agent when the user:
- Mentions CSV, Excel, spreadsheet, or tabular data
- Uploads files with extensions: `.csv`, `.xlsx`, `.xls`
- Asks about rows, columns, cells, or data aggregations
- Wants to filter, sort, or transform table data
- Requests charts or visualizations of tabular data

## NOT For

- PDF documents → use Document Agent
- Word documents (.docx) → use Document Agent  
- Web scraping → use Browser Agent
- Emails → use Mail Agent

## Example Prompts

- "Show me the top 10 customers by revenue"
- "Filter rows where status is 'active'"
- "Add a new column 'profit' = revenue - cost"
- "What's the average sales per month?"

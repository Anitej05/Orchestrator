---
id: zoho_books_agent
name: Zoho Books Agent
port: 8060
version: 1.0.0
---

# Zoho Books Agent

Integration with Zoho Books for accounting and financial operations.

## Capabilities

- Query invoices, bills, and transactions
- Create and update invoices
- Manage contacts and customers
- Generate financial reports
- Track expenses and payments
- Account reconciliation queries

## When to Use

Use this agent when the user:
- Mentions Zoho Books, invoices, or billing
- Wants to create or view invoices
- Asks about financial records
- Needs accounting reports
- Manages customer accounts

## NOT For

- Generic spreadsheets → use Spreadsheet Agent
- Documents → use Document Agent
- Emails → use Mail Agent

## Example Prompts

- "Show me all unpaid invoices"
- "Create an invoice for customer ABC Corp"
- "What's the total revenue this month?"
- "List all overdue bills"

## Notes

- Requires Zoho Books API credentials
- OAuth2 authentication required

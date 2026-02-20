---
id: zoho_books
name: Zoho Books Agent
description: Zoho Books accounting integration agent for managing invoices, contacts, expenses, payments, and financial reports via the Zoho Books API.
version: 1.0.0
port: 8060
type: agent
category: accounting

capabilities:
  - name: list_invoices
    description: List and filter invoices from Zoho Books

  - name: create_invoice
    description: Create a new invoice in Zoho Books

  - name: list_contacts
    description: List contacts (customers and vendors) in Zoho Books

  - name: get_financial_report
    description: Get financial reports (profit & loss, balance sheet, etc.)

  - name: list_expenses
    description: List and filter expenses

  - name: record_payment
    description: Record a payment against an invoice

parameters:
  - name: prompt
    type: string
    required: true
    description: The accounting task to execute

examples:
  - prompt: "List all unpaid invoices"
  - prompt: "Create an invoice for client ABC, $500 for consulting"
  - prompt: "Show me last month's expenses"
  - prompt: "Get the profit and loss report for Q4"
  - prompt: "Record a payment of $1000 for invoice INV-001"

use_when: |
  - Managing invoices (create, list, filter)
  - Looking up contacts or customers
  - Tracking expenses
  - Recording payments
  - Generating financial reports (P&L, balance sheet)
  - Any Zoho Books / accounting related task

not_for: |
  - General data analysis (use Spreadsheet Agent)
  - Sending emails (use Mail Agent)
  - Web browsing (use Browser Agent)
  - Document creation (use Document Agent)
---

# Zoho Books Agent

Accounting integration agent that connects to the Zoho Books API for:
1. **Invoice Management**: Create, list, and filter invoices
2. **Contact Management**: Look up customers and vendors
3. **Expense Tracking**: List and categorize expenses
4. **Payment Recording**: Record payments against invoices
5. **Financial Reports**: Generate P&L, balance sheet, and other reports

## Note
This agent requires Zoho API credentials to be configured. It is a scaffold implementation — full API integration depends on proper OAuth setup.

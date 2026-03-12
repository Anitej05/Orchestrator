---
id: zoho_books
name: Zoho Books Agent
port: 8110
version: 1.0.0
description: >
  Zoho Books accounting integration for invoices, contacts, 
  expenses, payments, and financial reports.
model: cerebras/llama-3.3-70b
context_strategy: minimal
requires_auth: true
composio_app_slug: zohobooks
triggers:
  - invoice
  - accounting
  - zoho
  - expense
  - payment
  - financial report
  - balance sheet
  - profit and loss
  - P&L
  - vendor
  - customer
capabilities:
  - manage_invoices
  - manage_contacts
  - track_expenses
  - record_payments
  - generate_reports
not_for:
  - email
  - spreadsheet analysis
  - web browsing
  - document editing
  - code
---

# Zoho Books Agent

Accounting integration agent that connects to the Zoho Books API for:
1. **Invoice Management**: Create, list, and filter invoices
2. **Contact Management**: Look up customers and vendors
3. **Expense Tracking**: List and categorize expenses
4. **Payment Recording**: Record payments against invoices
5. **Financial Reports**: Generate P&L, balance sheet, and other reports

## Note
This agent requires Zoho API credentials to be configured. 
It is a scaffold implementation — full API integration depends on proper OAuth setup.

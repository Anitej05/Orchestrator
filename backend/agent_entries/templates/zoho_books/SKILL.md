---
id: zoho_books
name: Zoho Books Agent
port: N/A (direct tool integration)
version: 1.0.0
---

# Zoho Books Agent

Comprehensive Zoho Books accounting integration for invoice, contact, item, and transaction management.

## Capabilities

- **Invoice Management**: Create, list, read, update, delete, and void invoices
- **Contact Management**: Create, read, update, and delete customer/vendor contacts
- **Item Management**: Manage inventory items and services with pricing
- **Bank Transactions**: Handle bank transaction recording and reconciliation
- **Approval Workflow**: Built-in human-in-the-loop approval for destructive operations
- **Multi-user Support**: Per-user authentication via Composio OAuth

## When to Use

Use these tools when the user:
- Mentions invoicing, billing, or accounting
- Wants to create or send invoices
- Needs to manage customers, vendors, or contacts
- Asks about payments or transactions
- Wants to update pricing or inventory items
- Mentions Zoho Books explicitly
- Needs to track business finances or accounting

## NOT For

- Email communication → use Gmail Agent
- Document processing → use Document Agent
- Spreadsheet analysis → use Spreadsheet Agent
- General web research → use Browser Agent
- Calendar/scheduling → use Calendar Agent (future)

## Example Prompts

- "Create an invoice for Acme Corp for $5000"
- "List all unpaid invoices"
- "Show me invoices from last month"
- "Add a new customer contact for John Smith"
- "Update the price of Product X to $99"
- "Void invoice #INV-001"
- "Delete the test invoice I created yesterday"
- "Show me all contacts in Zoho Books"
- "Record a bank deposit of $10,000"

## Authentication Requirements

- **Requires**: Active Zoho Books connection via Composio OAuth
- **Connection Setup**: User must visit `/connections` page and link Zoho Books account
- **Per-user**: Each user authenticates their own Zoho Books account
- **Permissions**: Requires full Zoho Books API access

## Available Tools

### 1. create_zoho_books_invoice
Creates a new invoice in Zoho Books.

**Input:**
```python
{
  "user_id": "user_123",
  "customer_id": "123456",
  "line_items": [{"item_id": "789", "quantity": 2, "rate": 100}],
  "notes": "Optional invoice notes"
}
```

**Output:** Invoice ID and status

---

### 2. list_zoho_books_invoices
Lists invoices with optional filters.

**Input:**
```python
{
  "user_id": "user_123",
  "status": "unpaid",  # Optional: all, paid, unpaid, overdue
  "customer_id": "123",  # Optional
  "date_from": "2026-01-01",  # Optional
  "date_to": "2026-12-31"  # Optional
}
```

**Output:** List of invoices with details

---

### 3. get_zoho_books_invoice
Retrieves a specific invoice by ID.

**Input:**
```python
{
  "user_id": "user_123",
  "invoice_id": "INV-001"
}
```

**Output:** Full invoice details

---

### 4. update_zoho_books_invoice
Updates an existing invoice. **Requires approval.**

**Input:**
```python
{
  "user_id": "user_123",
  "invoice_id": "INV-001",
  "updates": {"notes": "Updated notes", "due_date": "2026-02-15"}
}
```

**Output:** Updated invoice details

---

### 5. delete_zoho_books_invoice
Permanently deletes an invoice. **Requires approval.**

**Input:**
```python
{
  "user_id": "user_123",
  "invoice_id": "INV-001"
}
```

**Output:** Confirmation message

---

### 6. void_zoho_books_invoice
Voids (cancels) an invoice. **Requires approval.**

**Input:**
```python
{
  "user_id": "user_123",
  "invoice_id": "INV-001"
}
```

**Output:** Voided invoice status

---

### 7. manage_zoho_books_contacts
Create, read, update, or delete customer/vendor contacts.

**Input:**
```python
{
  "user_id": "user_123",
  "action": "create|read|update|delete",
  "contact_id": "123",  # For read/update/delete
  "contact_data": {  # For create/update
    "contact_name": "Acme Corp",
    "email": "billing@acme.com",
    "contact_type": "customer"
  }
}
```

**Output:** Contact details or confirmation

---

### 8. manage_zoho_books_items
Manage inventory items and service catalog.

**Input:**
```python
{
  "user_id": "user_123",
  "action": "create|read|update|delete|list",
  "item_id": "456",  # For read/update/delete
  "item_data": {  # For create/update
    "name": "Product X",
    "rate": 99.99,
    "description": "Premium product"
  }
}
```

**Output:** Item details or list

---

### 9. manage_zoho_books_bank_transactions
Record and manage bank transactions.

**Input:**
```python
{
  "user_id": "user_123",
  "action": "create|read|update|delete|list",
  "transaction_id": "789",  # For read/update/delete
  "transaction_data": {  # For create/update
    "amount": 1000.00,
    "date": "2026-02-10",
    "description": "Customer payment",
    "transaction_type": "deposit"
  }
}
```

**Output:** Transaction details or list

---

## Approval Workflow

Destructive operations require user approval before execution:

- **Delete operations**: Invoice deletion, contact deletion, item deletion
- **Void operations**: Invoice voiding
- **Update operations**: Invoice updates, contact updates, item updates

When approval is needed:
1. Tool returns `AgentResponse(status=NEEDS_INPUT)` with approval message
2. Orchestrator pauses execution and prompts user
3. User approves or rejects
4. If approved, tool re-executes and completes

---

## Technical Details

- **Integration**: Uses Composio SDK for Zoho Books API access
- **Auth**: Per-user OAuth via `ComposioAuthManager`
- **Tool Registration**: Uses `@tool` decorator from LangChain
- **Error Handling**: Returns structured errors with actionable messages
- **Rate Limits**: Respects Zoho Books API rate limits

---

## Configuration

Required environment variables:
```bash
COMPOSIO_API_KEY=your_api_key
COMPOSIO_AUTH_CONFIG_ZOHOBOOKS=your_zoho_auth_config_id
```

---

## Connection Management

Users can manage their Zoho Books connection via these helper functions:

- `check_zoho_books_connection(user_id)` - Check connection status
- `get_zoho_books_connect_url(user_id)` - Get OAuth URL to connect
- `disconnect_zoho_books(user_id)` - Disconnect Zoho Books

---

## Integration Status

- ✅ Composio SDK configured
- ✅ Per-user authentication working
- ✅ 9 tool functions implemented
- ✅ Approval workflow implemented
- ⚠️ Tools not yet registered in orchestrator (need registry entry or SKILL.md-based discovery)
- ⚠️ Approval flow needs wiring to orchestrator's state mechanism

---

## Future Enhancements

- Add bills (vendor invoices) management
- Implement expense tracking
- Add journal entries support
- Implement bank reconciliation automation
- Add tax calculation and reporting
- Support multiple Zoho Books organizations per user
- Add bulk operations (bulk invoice creation, etc.)
- Implement recurring invoice templates

---

## Notes

- Invoice numbering is automatic based on Zoho Books settings
- Line items must reference existing items in Zoho Books catalog
- Contacts must be created before creating invoices
- Voided invoices cannot be un-voided
- Deleted invoices are permanently removed (use void instead if unsure)
- All monetary amounts are in the account's default currency

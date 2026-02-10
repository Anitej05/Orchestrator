"""
Zoho Books Integration Helpers

Provides convenience functions for Zoho Books operations via Composio.
Uses the unified integrations system for proper multi-user support.

Quick Start:
    from tools.zoho_books_helpers import get_zoho_books_tools
    
    # Get all Zoho Books tools for a user
    tools = get_zoho_books_tools(user_id)
    
    # Or get specific actions only
    tools = get_zoho_books_tools(
        user_id, 
        actions=["ZOHOBOOKS_CREATE_INVOICE", "ZOHOBOOKS_LIST_INVOICES"]
    )
    
    # Use with agent
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    llm = ChatOpenAI(model="gpt-4o")
    agent = create_react_agent(llm, tools)
    result = agent.invoke({
        "messages": [HumanMessage(content="Create an invoice for customer X")]
    })
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool, BaseTool
from schemas import AgentResponse, AgentResponseStatus

logger = logging.getLogger("zoho_books_helpers")

# Define operations that require user approval before execution
REQUIRES_APPROVAL = {
    "ZOHOBOOKS_DELETE_INVOICE": "⚠️ WARNING: This will permanently delete invoice. This action cannot be undone.",
    "ZOHOBOOKS_VOID_INVOICE": "This will void the invoice and mark it as cancelled.",
    "ZOHOBOOKS_UPDATE_INVOICE": "This will modify the existing invoice.",
    "ZOHOBOOKS_DELETE_CONTACT": "⚠️ WARNING: This will permanently delete the contact and all associated data.",
    "ZOHOBOOKS_UPDATE_CONTACT": "This will update the contact information.",
    "ZOHOBOOKS_DELETE_ITEM": "⚠️ WARNING: This may affect existing invoices that reference this item.",
    "ZOHOBOOKS_UPDATE_ITEM": "This will update item pricing/details and affect future invoices.",
    "ZOHOBOOKS_DELETE_BANK_TRANSACTION": "⚠️ WARNING: This may affect bank reconciliation.",
    "ZOHOBOOKS_DELETE_VENDOR_BILL": "⚠️ WARNING: This will permanently delete the bill.",
    "ZOHOBOOKS_UPDATE_VENDOR_BILL": "This will modify the existing bill.",
}


def get_zoho_books_tools(
    user_id: str,
    actions: Optional[List[str]] = None
) -> List[BaseTool]:
    """
    Get Zoho Books tools for a specific user via Composio.
    
    This is the recommended way to access Zoho Books functionality.
    Composio provides 50+ Zoho Books actions including:
    
    Invoice Operations:
        - ZOHOBOOKS_CREATE_INVOICE
        - ZOHOBOOKS_LIST_INVOICES
        - ZOHOBOOKS_GET_INVOICE
        - ZOHOBOOKS_UPDATE_INVOICE
        - ZOHOBOOKS_DELETE_INVOICE
        - ZOHOBOOKS_VOID_INVOICE
        - ZOHOBOOKS_EMAIL_INVOICE
        - ZOHOBOOKS_MARK_INVOICE_AS_SENT
        - ZOHOBOOKS_BULK_EXPORT_INVOICES_PDF
        - ZOHOBOOKS_BULK_PRINT_INVOICES
        - ZOHOBOOKS_GET_INVOICE_PDF
        - ZOHOBOOKS_SEND_PAYMENT_REMINDER
        
    Bill Operations:
        - ZOHOBOOKS_CREATE_VENDOR_BILL
        - ZOHOBOOKS_UPDATE_VENDOR_BILL
        - ZOHOBOOKS_GET_BILL
        - ZOHOBOOKS_LIST_BILLS
        
    Contact Operations:
        - ZOHOBOOKS_CREATE_CONTACT
        - ZOHOBOOKS_UPDATE_CONTACT
        - ZOHOBOOKS_DELETE_CONTACT
        - ZOHOBOOKS_GET_CONTACT
        - ZOHOBOOKS_LIST_CONTACTS
        - ZOHOBOOKS_MARK_CONTACT_AS_ACTIVE
        
    Item Operations:
        - ZOHOBOOKS_CREATE_ITEM
        - ZOHOBOOKS_UPDATE_ITEM
        - ZOHOBOOKS_DELETE_ITEM
        - ZOHOBOOKS_GET_ITEM
        - ZOHOBOOKS_LIST_ITEMS
        
    Sales Order Operations:
        - ZOHOBOOKS_CREATE_SALES_ORDER
        - ZOHOBOOKS_UPDATE_SALES_ORDER
        - ZOHOBOOKS_GET_SALES_ORDER
        - ZOHOBOOKS_LIST_SALES_ORDERS
        - ZOHOBOOKS_OPEN_SALES_ORDER
        
    Estimate Operations:
        - ZOHOBOOKS_CREATE_ESTIMATE
        - ZOHOBOOKS_GET_ESTIMATE
        - ZOHOBOOKS_ACCEPT_ESTIMATE
        
    Bank Operations:
        - ZOHOBOOKS_CREATE_BANK_ACCOUNT
        - ZOHOBOOKS_GET_BANK_ACCOUNT
        - ZOHOBOOKS_LIST_BANK_ACCOUNTS
        - ZOHOBOOKS_CREATE_BANK_TRANSACTION
        - ZOHOBOOKS_DELETE_BANK_TRANSACTION
        - ZOHOBOOKS_LIST_BANK_TRANSACTIONS
        - ZOHOBOOKS_CATEGORIZE_UNCATEGORIZED_TRANSACTION
        
    Organizational:
        - ZOHOBOOKS_LIST_ORGANIZATIONS
        - ZOHOBOOKS_LIST_CURRENCIES
        - ZOHOBOOKS_CREATE_EXCHANGE_RATE
        - ZOHOBOOKS_LIST_CHART_OF_ACCOUNTS
        - ZOHOBOOKS_LIST_INVOICE_PAYMENTS
        
    User Management:
        - ZOHOBOOKS_CREATE_USER
        - ZOHOBOOKS_GET_USER
        - ZOHOBOOKS_LIST_USERS
    
    Args:
        user_id: Your application's user identifier (must have active Zoho Books connection)
        actions: Optional list of specific action IDs to load (loads all if None)
    
    Returns:
        List of LangChain BaseTool objects ready for agent use
        
    Raises:
        Exception: If Composio API is not configured or user has no active connection
        
    Example - Get all tools:
        >>> tools = get_zoho_books_tools("user_123")
        >>> len(tools)
        50+
        
    Example - Get specific tools:
        >>> tools = get_zoho_books_tools(
        ...     "user_123",
        ...     actions=["ZOHOBOOKS_CREATE_INVOICE", "ZOHOBOOKS_LIST_INVOICES"]
        ... )
        >>> len(tools)
        2
        
    Example - Use with agent:
        >>> from langchain_openai import ChatOpenAI
        >>> from langgraph.prebuilt import create_react_agent
        >>> from langchain_core.messages import HumanMessage
        >>> 
        >>> tools = get_zoho_books_tools("user_123")
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> agent = create_react_agent(llm, tools)
        >>> 
        >>> result = agent.invoke({
        ...     "messages": [HumanMessage(
        ...         content="Create an invoice for customer ABC with $1000 consulting service"
        ...     )]
        ... })
    """
    try:
        from services.integrations.composio_tools import get_tools_for_user
        
        if actions:
            # Get specific actions
            logger.info(f"Loading {len(actions)} specific Zoho Books actions for user {user_id}")
            tools = get_tools_for_user(user_id, actions=actions)
        else:
            # Get all Zoho Books tools
            logger.info(f"Loading all Zoho Books tools for user {user_id}")
            tools = get_tools_for_user(user_id, apps=["zohobooks"])
        
        logger.info(f"Successfully loaded {len(tools)} Zoho Books tools for user {user_id}")
        return tools
        
    except Exception as e:
        logger.error(f"Failed to load Zoho Books tools for user {user_id}: {e}", exc_info=True)
        raise


def check_zoho_books_connection(user_id: str) -> dict:
    """
    Check if user has an active Zoho Books connection.
    
    Args:
        user_id: User identifier
        
    Returns:
        {
            "connected": bool,
            "connection_id": str (if connected),
            "status": "active"/"INITIATED"/"disconnected",
            "error": str (if failed)
        }
        
    Example:
        >>> status = check_zoho_books_connection("user_123")
        >>> if status["connected"]:
        ...     tools = get_zoho_books_tools("user_123")
        ... else:
        ...     print("Please connect Zoho Books in Settings")
    """
    try:
        from services.integrations.composio_auth import get_auth_manager
        
        auth_manager = get_auth_manager()
        result = auth_manager.check_connection_status(user_id, app_slug="zohobooks")
        
        # Extract Zoho Books toolkit info
        zoho_toolkit = next(
            (t for t in result.get("all_toolkits", []) if t["slug"] == "zohobooks"),
            None
        )
        
        if zoho_toolkit:
            return {
                "connected": zoho_toolkit["is_connected"] or zoho_toolkit.get("db_status") == "active",
                "connection_id": zoho_toolkit.get("connected_account_id") or zoho_toolkit.get("db_connection_id"),
                "status": zoho_toolkit.get("db_status", "disconnected")
            }
        else:
            return {
                "connected": False,
                "status": "disconnected"
            }
            
    except Exception as e:
        logger.error(f"Failed to check Zoho Books connection: {e}", exc_info=True)
        return {
            "connected": False,
            "error": str(e)
        }


def get_zoho_books_connect_url(user_id: str, redirect_url: Optional[str] = None) -> dict:
    """
    Generate OAuth connection URL for user to link their Zoho Books account.
    
    Args:
        user_id: User identifier
        redirect_url: Optional URL to redirect after successful connection
        
    Returns:
        {
            "success": bool,
            "connect_url": str (if success),
            "error": str (if failed)
        }
        
    Example:
        >>> result = get_zoho_books_connect_url("user_123")
        >>> if result["success"]:
        ...     print(f"Connect at: {result['connect_url']}")
    """
    try:
        from services.integrations import get_auth_manager
        
        auth_manager = get_auth_manager()
        result = auth_manager.start_auth_flow(
            user_id=user_id,
            app_slug="zohobooks",
            redirect_url=redirect_url
        )
        
        return {
            "success": True,
            "connect_url": result.get("redirect_url"),
            "connection_id": result.get("connection_id")
        }
        
    except Exception as e:
        logger.error(f"Failed to get connect URL: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def disconnect_zoho_books(user_id: str) -> dict:
    """
    Disconnect user's Zoho Books account.
    
    Args:
        user_id: User identifier
        
    Returns:
        {
            "success": bool,
            "message": str,
            "error": str (if failed)
        }
        
    Example:
        >>> result = disconnect_zoho_books("user_123")
        >>> print(result["message"])
    """
    try:
        from services.integrations import get_auth_manager
        
        auth_manager = get_auth_manager()
        result = auth_manager.disconnect_app(
            user_id=user_id,
            app_slug="zohobooks"
        )
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Disconnected successfully")
        }
        
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


# Convenience exports for connection management
__all__ = [
    "get_zoho_books_tools",
    "check_zoho_books_connection",
    "get_zoho_books_connect_url",
    "disconnect_zoho_books",
    # Tool entry functions for orchestrator
    "create_zoho_books_invoice",
    "list_zoho_books_invoices",
    "get_zoho_books_invoice",
    "update_zoho_books_invoice",
    "delete_zoho_books_invoice",
    "void_zoho_books_invoice",
    "manage_zoho_books_contacts",
    "manage_zoho_books_items",
    "manage_zoho_books_bank_transactions",
]


# ==================== ORCHESTRATOR TOOL WRAPPERS ====================
# These functions are called directly by the orchestrator's tool registry
# They handle approval prompts for sensitive operations

def _check_approval_needed(action: str, params: Dict[str, Any]) -> Optional[str]:
    """
    Check if an action requires user approval and return the approval message.
    
    Args:
        action: Composio action ID (e.g., "ZOHOBOOKS_DELETE_INVOICE")
        params: Action parameters for context
        
    Returns:
        Approval message if approval needed, None otherwise
    """
    if action in REQUIRES_APPROVAL:
        message = REQUIRES_APPROVAL[action]
        # Try to make message more specific with available context
        if "invoice_id" in params:
            message += f"\n\nInvoice ID: {params['invoice_id']}"
        if "contact_id" in params:
            message += f"\n\nContact ID: {params['contact_id']}"
        if "item_id" in params:
            message += f"\n\nItem ID: {params['item_id']}"
        return message
    return None


def _execute_zoho_action(
    user_id: str, 
    action: str, 
    params: Dict[str, Any],
    pending_approval: bool = False
) -> AgentResponse:
    """
    Execute a Zoho Books action via Composio with approval handling.
    
    Args:
        user_id: User's identifier
        action: Composio action ID
        params: Action parameters
        pending_approval: If True, return NEEDS_INPUT for approval
        
    Returns:
        AgentResponse with status, result, or approval request
    """
    try:
        # Check if approval is needed and not yet granted
        approval_msg = _check_approval_needed(action, params)
        if approval_msg and not pending_approval:
            # Need to ask for approval first
            logger.info(f"[APPROVAL] Action {action} requires user approval")
            return AgentResponse(
                status=AgentResponseStatus.NEEDS_INPUT,
                question=f"{approval_msg}\n\nDo you want to proceed?",
                question_type="confirmation",
                options=["Yes, proceed", "No, cancel"],
                context={
                    "action": action,
                    "params": params,
                    "approval_pending": True,
                    "operation_type": "zoho_books"
                }
            )
        
        # Get Composio tools and execute
        from services.integrations.composio_tools import get_tools_for_user
        
        logger.info(f"Executing Zoho Books action: {action} for user {user_id}")
        
        # Get the specific action tool
        tools = get_tools_for_user(user_id, actions=[action])
        
        if not tools:
            return AgentResponse(
                status=AgentResponseStatus.ERROR,
                error=f"Zoho Books action '{action}' not available. Please check connection."
            )
        
        tool = tools[0]
        
        # Execute the tool
        result = tool.invoke(params)
        
        logger.info(f"Zoho Books action {action} completed successfully")
        
        return AgentResponse(
            status=AgentResponseStatus.COMPLETE,
            result=result,
            context={"action": action, "user_id": user_id}
        )
        
    except Exception as e:
        logger.error(f"Failed to execute Zoho Books action {action}: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Zoho Books operation failed: {str(e)}"
        )


@tool
def create_zoho_books_invoice(
    user_id: str,
    customer_id: str,
    line_items: List[Dict[str, Any]],
    invoice_number: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a new invoice in Zoho Books for a customer.
    
    Args:
        user_id: User's unique identifier
        customer_id: Zoho Books customer ID
        line_items: List of invoice line items with item_id, quantity, rate
        invoice_number: Optional custom invoice number
        
    Returns:
        Created invoice details with invoice_id, number, total, and status
    """
    params = {
        "customer_id": customer_id,
        "line_items": line_items,
        **kwargs
    }
    if invoice_number:
        params["invoice_number"] = invoice_number
    
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_CREATE_INVOICE", params)
    return response.model_dump()


@tool
def list_zoho_books_invoices(
    user_id: str,
    status: Optional[str] = None,
    customer_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    List all invoices with optional filtering by status or customer.
    
    Args:
        user_id: User's unique identifier
        status: Filter by status (draft, sent, overdue, paid, void)
        customer_id: Filter by customer ID
        
    Returns:
        List of invoices with details
    """
    params = {**kwargs}
    if status:
        params["status"] = status
    if customer_id:
        params["customer_id"] = customer_id
    
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_LIST_INVOICES", params)
    return response.model_dump()


@tool
def get_zoho_books_invoice(
    user_id: str,
    invoice_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Get detailed information about a specific invoice.
    
    Args:
        user_id: User's unique identifier
        invoice_id: Zoho Books invoice ID
        
    Returns:
        Complete invoice details including line items, taxes, and payments
    """
    params = {"invoice_id": invoice_id, **kwargs}
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_GET_INVOICE", params)
    return response.model_dump()


@tool
def update_zoho_books_invoice(
    user_id: str,
    invoice_id: str,
    updates: Dict[str, Any],
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Update an existing invoice. Requires user approval.
    
    Args:
        user_id: User's unique identifier
        invoice_id: Invoice ID to update
        updates: Fields to update
        approved: Internal flag indicating approval was granted
        
    Returns:
        Updated invoice details or approval request
    """
    params = {"invoice_id": invoice_id, **updates, **kwargs}
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_UPDATE_INVOICE", params, pending_approval=approved)
    return response.model_dump()


@tool
def delete_zoho_books_invoice(
    user_id: str,
    invoice_id: str,
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Permanently delete an invoice. Requires user approval.
    
    ⚠️ WARNING: This action cannot be undone.
    
    Args:
        user_id: User's unique identifier
        invoice_id: Invoice ID to delete
        approved: Internal flag indicating approval was granted
        
    Returns:
        Deletion confirmation or approval request
    """
    params = {"invoice_id": invoice_id, **kwargs}
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_DELETE_INVOICE", params, pending_approval=approved)
    return response.model_dump()


@tool
def void_zoho_books_invoice(
    user_id: str,
    invoice_id: str,
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Void an invoice (mark as cancelled but keep in records). Requires approval.
    
    Args:
        user_id: User's unique identifier
        invoice_id: Invoice ID to void
        approved: Internal flag indicating approval was granted
        
    Returns:
        Voided invoice confirmation or approval request
    """
    params = {"invoice_id": invoice_id, **kwargs}
    response = _execute_zoho_action(user_id, "ZOHOBOOKS_VOID_INVOICE", params, pending_approval=approved)
    return response.model_dump()


@tool
def manage_zoho_books_contacts(
    user_id: str,
    operation: str,
    contact_data: Optional[Dict[str, Any]] = None,
    contact_id: Optional[str] = None,
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Manage customer and vendor contacts (create, list, update, delete).
    Update and delete operations require approval.
    
    Args:
        user_id: User's unique identifier
        operation: Operation type (create, list, get, update, delete)
        contact_data: Contact details for create/update
        contact_id: Contact ID for get/update/delete
        approved: Internal approval flag
        
    Returns:
        Contact details or operation result
    """
    action_map = {
        "create": "ZOHOBOOKS_CREATE_CONTACT",
        "list": "ZOHOBOOKS_LIST_CONTACTS",
        "get": "ZOHOBOOKS_GET_CONTACT",
        "update": "ZOHOBOOKS_UPDATE_CONTACT",
        "delete": "ZOHOBOOKS_DELETE_CONTACT"
    }
    
    action = action_map.get(operation.lower())
    if not action:
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Invalid operation: {operation}. Must be: create, list, get, update, or delete"
        ).model_dump()
    
    params = {**kwargs}
    if contact_id:
        params["contact_id"] = contact_id
    if contact_data:
        params.update(contact_data)
    
    needs_approval = operation.lower() in ["update", "delete"]
    response = _execute_zoho_action(user_id, action, params, pending_approval=approved if needs_approval else False)
    return response.model_dump()


@tool
def manage_zoho_books_items(
    user_id: str,
    operation: str,
    item_data: Optional[Dict[str, Any]] = None,
    item_id: Optional[str] = None,
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Manage product/service items (create, list, update, delete).
    Update and delete operations require approval.
    
    Args:
        user_id: User's unique identifier
        operation: Operation type (create, list, get, update, delete)
        item_data: Item details (name, price, description, sku)
        item_id: Item ID for get/update/delete
        approved: Internal approval flag
        
    Returns:
        Item details or operation result
    """
    action_map = {
        "create": "ZOHOBOOKS_CREATE_ITEM",
        "list": "ZOHOBOOKS_LIST_ITEMS",
        "get": "ZOHOBOOKS_GET_ITEM",
        "update": "ZOHOBOOKS_UPDATE_ITEM",
        "delete": "ZOHOBOOKS_DELETE_ITEM"
    }
    
    action = action_map.get(operation.lower())
    if not action:
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Invalid operation: {operation}. Must be: create, list, get, update, or delete"
        ).model_dump()
    
    params = {**kwargs}
    if item_id:
        params["item_id"] = item_id
    if item_data:
        params.update(item_data)
    
    needs_approval = operation.lower() in ["update", "delete"]
    response = _execute_zoho_action(user_id, action, params, pending_approval=approved if needs_approval else False)
    return response.model_dump()


@tool
def manage_zoho_books_bank_transactions(
    user_id: str,
    operation: str,
    transaction_data: Optional[Dict[str, Any]] = None,
    transaction_id: Optional[str] = None,
    approved: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Manage bank transactions (create, list, categorize, delete).
    Delete operations require approval.
    
    Args:
        user_id: User's unique identifier
        operation: Operation type (create, list, categorize, delete)
        transaction_data: Transaction details
        transaction_id: Transaction ID for delete
        approved: Internal approval flag
        
    Returns:
        Transaction details or operation result
    """
    action_map = {
        "create": "ZOHOBOOKS_CREATE_BANK_TRANSACTION",
        "list": "ZOHOBOOKS_LIST_BANK_TRANSACTIONS",
        "categorize": "ZOHOBOOKS_CATEGORIZE_UNCATEGORIZED_TRANSACTION",
        "delete": "ZOHOBOOKS_DELETE_BANK_TRANSACTION"
    }
    
    action = action_map.get(operation.lower())
    if not action:
        return AgentResponse(
            status=AgentResponseStatus.ERROR,
            error=f"Invalid operation: {operation}. Must be: create, list, categorize, or delete"
        ).model_dump()
    
    params = {**kwargs}
    if transaction_id:
        params["transaction_id"] = transaction_id
    if transaction_data:
        params.update(transaction_data)
    
    needs_approval = operation.lower() == "delete"
    response = _execute_zoho_action(user_id, action, params, pending_approval=approved if needs_approval else False)
    return response.model_dump()

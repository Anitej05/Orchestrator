"""
Zoho Books Agent - Pharmaceutical Invoice Automation
Provides comprehensive Zoho Books integration for pharmaceutical companies.
Handles invoices, customers, items, payments, and pharmaceutical-specific fields.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, Union
from enum import Enum
from datetime import datetime, timedelta
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException, Query, Body, Form, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# CMS Integration
import sys
from pathlib import Path
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.services.content_management_service import ContentManagementService
from backend.services.canvas_service import CanvasService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize CMS
cms_service = ContentManagementService()

# ============================================================================
# LLM CONFIGURATION (for future intelligent features)
# ============================================================================
# Centralized Inference Service
from backend.services.inference_service import inference_service, InferencePriority
from langchain_core.messages import HumanMessage
from backend.schemas import AgentResponse, StandardAgentResponse, AgentResponseStatus

# Import standardized file manager
try:
    from backend.agents.utils.agent_file_manager import AgentFileManager, FileType, FileStatus
except ImportError:
    from agent_file_manager import AgentFileManager, FileType, FileStatus

from .planner import ZohoPlanner

# Initialize planner (lazy loaded)
_planner = None

def get_planner() -> ZohoPlanner:
    """Get or create planner instance."""
    global _planner
    if _planner is None:
        logger.info("Initializing ZohoPlanner...")
        _planner = ZohoPlanner()
    return _planner

# ============================================================================
# CONFIGURATION & CREDENTIALS
# ============================================================================

# Load credentials from temp.json
# Path: backend/temp.json (from agents/zoho_books directory, go up 3 levels to reach backend root)
CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "temp.json"
_zoho_config = None
_zoho_config_lock = Lock()


def load_zoho_config() -> Dict[str, Any]:
    """
    Load Zoho Books configuration from temp.json with caching.

    Supports two styles of temp.json:
    1) Old style (client credentials):
       {
         "client_id": "...",
         "client_secret": "...",
         "api_domain": "https://www.zohoapis.eu",
         "organization_id": "..."
       }
    2) Token style (what you pasted from Zoho):
       {
         "access_token": "...",
         "refresh_token": "...",
         "scope": "ZohoBooks.fullaccess.all",
         "api_domain": "https://www.zohoapis.eu",
         "token_type": "Bearer",
         "expires_in": 3600,
         "organization_id": "..."
       }
    """
    global _zoho_config
    with _zoho_config_lock:
        if _zoho_config is None:
            try:
                with open(CREDENTIALS_PATH, "r") as f:
                    _zoho_config = json.load(f)
                logger.info("Zoho Books credentials loaded successfully from temp.json")
            except Exception as e:
                logger.error(f"Failed to load Zoho Books credentials: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Zoho Books credentials not found. Please ensure temp.json exists with valid credentials."
                )

            # If temp.json already contains tokens, initialize cache once
            access_token = _zoho_config.get("access_token")
            if access_token:
                logger.info("Initializing OAuth token cache from temp.json")
                expires_in = int(_zoho_config.get("expires_in", 3600))
                refresh_token = _zoho_config.get("refresh_token")
                with _token_lock:
                    _token_cache["access_token"] = access_token
                    _token_cache["refresh_token"] = refresh_token
                    _token_cache["expires_at"] = time.time() + expires_in
                    # Use Zoho's token_type if present, otherwise default
                    if _zoho_config.get("token_type"):
                        _token_cache["token_type"] = _zoho_config["token_type"]

        return _zoho_config.copy()

# OAuth2 Token Management
_token_cache = {
    "access_token": None,
    "refresh_token": None,
    "expires_at": 0,
    "token_type": "Zoho-oauthtoken"
}
_token_lock = Lock()

# API Rate Limiting (1000 calls/day for test system)
_rate_limit = {
    "calls_today": 0,
    "reset_time": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1),
    "lock": Lock()
}

# ============================================================================
# AGENT DEFINITION
# ============================================================================

# Agent Logs: `storage/zoho_books_agent/agent.log`
# Request Logs: `storage/zoho_books_agent/requests.log`
# Error Logs: `storage/zoho_books_agent/errors.log`
AGENT_DEFINITION = {
    "id": "zoho_books_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Zoho Books Agent - Pharmaceutical",
    "description": "Comprehensive Zoho Books integration for pharmaceutical companies. Automates invoice creation, management, customer handling, and pharmaceutical-specific operations with batch tracking, expiry dates, and regulatory compliance.",
    "capabilities": [
        "create invoice",
        "update invoice",
        "get invoice",
        "list invoices",
        "delete invoice",
        "send invoice",
        "mark invoice as sent",
        "mark invoice as paid",
        "create customer",
        "update customer",
        "get customer",
        "list customers",
        "create item",
        "update item",
        "get item",
        "list items",
        "create payment",
        "get payment",
        "list payments",
        "pharmaceutical invoice",
        "batch number tracking",
        "expiry date management",
        "drug name tracking",
        "regulatory compliance",
        "pharmaceutical inventory",
        "invoice automation",
        "invoice management",
        "customer management",
        "payment tracking",
        "estimate creation",
        "estimate management",
        "recurring invoice",
        "invoice template",
        "bulk invoice creation",
        "invoice approval workflow"
    ],
    "price_per_call_usd": 0.01,
    "status": "active",
    "agent_type": "http_rest",
    "connection_config": {
        "base_url": "http://localhost:8050"
    },
    "endpoints": []  # Will be populated below
}

app = FastAPI(title="Zoho Books Agent - Pharmaceutical")

# Initialize file manager
# Initialize file manager
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage" / "zoho_books_agent"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Add file handler to the root logger
logging.getLogger().addHandler(logging.FileHandler(STORAGE_DIR / "debug.log"))

file_manager = AgentFileManager(
    agent_id="zoho_books_agent",
    storage_dir=str(STORAGE_DIR),
    default_ttl_hours=168,  # 7 days
    auto_cleanup=True,
    cleanup_interval_hours=24
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_rate_limit() -> bool:
    """Check if API rate limit allows another call."""
    with _rate_limit["lock"]:
        now = datetime.now()
        # Reset counter if new day
        if now >= _rate_limit["reset_time"]:
            _rate_limit["calls_today"] = 0
            _rate_limit["reset_time"] = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        if _rate_limit["calls_today"] >= 1000:
            logger.warning(f"Rate limit reached: {_rate_limit['calls_today']}/1000 calls today")
            return False
        
        _rate_limit["calls_today"] += 1
        return True

def initialize_oauth_token(access_token: str, refresh_token: Optional[str] = None, expires_in: int = 3600):
    """
    Initialize OAuth tokens manually.
    This should be called after obtaining tokens from Zoho OAuth flow.
    
    To get tokens:
    1. Visit https://accounts.zoho.in/developerconsole
    2. Create a Zoho Books API application
    3. Generate access/refresh tokens using OAuth 2.0 flow
    4. Call this endpoint with the tokens
    """
    with _token_lock:
        _token_cache["access_token"] = access_token
        _token_cache["expires_at"] = time.time() + expires_in
        if refresh_token:
            _token_cache["refresh_token"] = refresh_token
        logger.info("OAuth tokens initialized successfully")

def get_access_token() -> str:
    """Get valid access token, refreshing if necessary."""
    with _token_lock:
        now = time.time()
        
        # If token is valid, return it
        if _token_cache["access_token"] and now < _token_cache["expires_at"] - 60:  # 60s buffer
            return _token_cache["access_token"]
        
        # Need to refresh or get new token
        config = load_zoho_config()
        api_domain = config.get("api_domain", "https://www.zohoapis.in")
        
        # If we have a refresh token, use it
        if _token_cache.get("refresh_token"):
            try:
                response = httpx.post(
                    f"{api_domain}/oauth/v2/token",
                    data={
                        "refresh_token": _token_cache["refresh_token"],
                        "client_id": config["client_id"],
                        "client_secret": config["client_secret"],
                        "grant_type": "refresh_token"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                _token_cache["access_token"] = data["access_token"]
                _token_cache["expires_at"] = now + data.get("expires_in", 3600)
                if "refresh_token" in data:
                    _token_cache["refresh_token"] = data["refresh_token"]
                
                logger.info("Access token refreshed successfully")
                return _token_cache["access_token"]
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                raise HTTPException(
                    status_code=401,
                    detail=f"Failed to refresh Zoho Books access token: {str(e)}"
                )
        else:
            # First time - need to get token via OAuth flow
            raise HTTPException(
                status_code=401,
                detail="No access token available. Please complete OAuth2 authorization flow first. "
                       "Visit Zoho Developer Console to generate access/refresh tokens."
            )

def make_zoho_request(
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    json_data: Optional[Dict] = None,
    retries: int = 3
) -> Dict[str, Any]:
    """
    Make a request to Zoho Books API with fail-safe error handling.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint (e.g., "/invoices")
        params: Query parameters
        json_data: Request body for POST/PUT
        retries: Number of retry attempts
        
    Returns:
        API response as dictionary
        
    Raises:
        HTTPException: On API errors
    """
    if not check_rate_limit():
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded (1000 calls/day). Please try again tomorrow."
        )
    
    config = load_zoho_config()
    api_domain = config.get("api_domain", "https://www.zohoapis.in").rstrip("/")
    organization_id = config.get("organization_id")
    
    if not organization_id:
        raise HTTPException(
            status_code=500,
            detail="Organization ID not found in configuration"
        )
    
    access_token = get_access_token()
    url = f"{api_domain}/books/v3/{endpoint.lstrip('/')}"
    
    headers = {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "Content-Type": "application/json",
        "X-com-zoho-books-organizationid": str(organization_id)
    }
    
    last_error = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = client.get(url, headers=headers, params=params)
                elif method.upper() == "POST":
                    # Try JSON first, but Zoho Books EU might require form data
                    if json_data:
                        # First try JSON
                        response = client.post(url, headers=headers, params=params, json=json_data)
                        # If 400 error with "Invalid value", try form data (but only once per attempt)
                        if response.status_code == 400:
                            try:
                                error_text = response.text
                                if "Invalid value" in error_text and attempt == 0:  # Only try form data on first attempt
                                    logger.info("JSON format failed with 'Invalid value', trying form-encoded data")
                                    # Convert JSON to form data
                                    headers_form = headers.copy()
                                    headers_form["Content-Type"] = "application/x-www-form-urlencoded"
                                    form_data = {}
                                    for key, value in json_data.items():
                                        if isinstance(value, (dict, list)):
                                            form_data[key] = json.dumps(value)
                                        else:
                                            form_data[key] = str(value)
                                    response = client.post(url, headers=headers_form, params=params, data=form_data)
                            except Exception as form_err:
                                logger.warning(f"Form data attempt failed: {form_err}, using original JSON response")
                    else:
                        response = client.post(url, headers=headers, params=params)
                elif method.upper() == "PUT":
                    response = client.put(url, headers=headers, params=params, json=json_data)
                elif method.upper() == "DELETE":
                    response = client.delete(url, headers=headers, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = 60 * (attempt + 1)  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{retries}")
                    time.sleep(wait_time)
                    continue
                
                # Handle token expiration
                if response.status_code == 401:
                    # Force token refresh
                    with _token_lock:
                        _token_cache["access_token"] = None
                        _token_cache["expires_at"] = 0
                    
                    if attempt < retries - 1:
                        access_token = get_access_token()
                        headers["Authorization"] = f"Zoho-oauthtoken {access_token}"
                        continue
                
                # Handle 400 errors - parse and show Zoho's error message
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("message", "Bad Request")
                        error_code = error_data.get("code", "UNKNOWN")
                        # Zoho sometimes has more details in nested structure
                        if "errors" in error_data:
                            errors = error_data["errors"]
                            if isinstance(errors, list) and len(errors) > 0:
                                error_msg = errors[0].get("message", error_msg)
                        logger.error(f"Zoho Books API 400 error: {error_msg} (code: {error_code})")
                        logger.error(f"Request payload that failed: {json.dumps(json_data if json_data else params, indent=2)}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Zoho Books API error: {error_msg} (code: {error_code}). Full response: {json.dumps(error_data, indent=2)}"
                        )
                    except json.JSONDecodeError:
                        # If response isn't JSON, use raw text
                        error_text = response.text[:500]  # Limit length
                        logger.error(f"Zoho Books API 400 error (non-JSON): {error_text}")
                        raise HTTPException(
                            status_code=400,
                            detail=f"Zoho Books API error: {error_text}"
                        )
                
                response.raise_for_status()
                
                # Parse response
                try:
                    result = response.json()
                except Exception:
                    result = {"raw_response": response.text}
                
                # Check for Zoho API errors in successful-looking responses
                if isinstance(result, dict) and result.get("code") and result.get("code") != 0:
                    error_msg = result.get("message", "Unknown Zoho Books API error")
                    logger.error(f"Zoho Books API error: {error_msg}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Zoho Books API error: {error_msg}"
                    )
                
                return result
                
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code in [429, 401] and attempt < retries - 1:
                continue
            logger.error(f"HTTP error on attempt {attempt + 1}/{retries}: {e}")
        except httpx.RequestError as e:
            last_error = e
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
            logger.error(f"Request error on attempt {attempt + 1}/{retries}: {e}")
        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error on attempt {attempt + 1}/{retries}: {e}")
            break
    
    # All retries failed
    error_detail = str(last_error) if last_error else "Unknown error"
    raise HTTPException(
        status_code=502,
        detail=f"Failed to communicate with Zoho Books API after {retries} attempts: {error_detail}"
    )

def add_pharmaceutical_fields(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add pharmaceutical-specific fields to item data.
    Ensures batch numbers, expiry dates, and regulatory info are included.
    """
    # Pharmaceutical-specific custom fields
    pharma_fields = {
        "batch_number": item_data.get("batch_number"),
        "expiry_date": item_data.get("expiry_date"),
        "drug_name": item_data.get("drug_name") or item_data.get("name"),
        "manufacturer": item_data.get("manufacturer"),
        "regulatory_license": item_data.get("regulatory_license"),
        "storage_conditions": item_data.get("storage_conditions"),
        "hazard_class": item_data.get("hazard_class"),
        "controlled_substance": item_data.get("controlled_substance", False),
        "prescription_required": item_data.get("prescription_required", False)
    }
    
    # Add to custom fields if Zoho Books supports it
    # Note: Zoho Books uses custom fields, so we'll add these as line_item_custom_fields
    custom_fields = item_data.get("line_item_custom_fields", [])
    
    for key, value in pharma_fields.items():
        if value is not None:
            custom_fields.append({
                "customfield_id": key,  # This would be set up in Zoho Books
                "value": str(value)
            })
    
    item_data["line_item_custom_fields"] = custom_fields
    return item_data

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class InvoiceItem(BaseModel):
    """Invoice line item with all Zoho Books fields plus pharmaceutical-specific fields."""
    # Required: either item_id OR name+rate
    item_id: Optional[str] = Field(None, description="Item ID from Zoho Books items")
    name: Optional[str] = Field(None, description="Item name (required if item_id not provided)")
    description: Optional[str] = Field(None, description="Item description")
    rate: Optional[float] = Field(None, description="Item rate/price (required if item_id not provided)")
    quantity: float = Field(1.0, description="Quantity")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    
    # Item details
    product_type: Optional[str] = Field(None, description="goods, service, or digital_service")
    hsn_or_sac: Optional[str] = Field(None, description="HSN/SAC code")
    item_order: Optional[int] = Field(None, description="Order of item in invoice")
    bcy_rate: Optional[float] = Field(None, description="Base currency rate")
    
    # Discount
    discount: Optional[float] = Field(None, description="Discount amount")
    discount_amount: Optional[float] = Field(None, description="Discount amount (alternative)")
    
    # Tax information
    tax_id: Optional[str] = Field(None, description="Tax ID")
    tax_name: Optional[str] = Field(None, description="Tax name")
    tax_type: Optional[str] = Field(None, description="Tax type")
    tax_percentage: Optional[float] = Field(None, description="Tax percentage")
    tds_tax_id: Optional[str] = Field(None, description="TDS tax ID")
    tax_treatment_code: Optional[str] = Field(None, description="Tax treatment code")
    
    # Project and time tracking
    project_id: Optional[str] = Field(None, description="Project ID")
    time_entry_ids: Optional[List[str]] = Field(None, description="Time entry IDs")
    expense_id: Optional[str] = Field(None, description="Expense ID")
    
    # Location and inventory
    location_id: Optional[str] = Field(None, description="Location ID")
    
    # Other fields
    tags: Optional[List[Dict[str, Any]]] = Field(None, description="Tags")
    header_name: Optional[str] = Field(None, description="Header name for grouping")
    salesorder_item_id: Optional[str] = Field(None, description="Sales order item ID")
    
    # Pharmaceutical-specific fields (stored in custom_fields or description)
    batch_number: Optional[str] = Field(None, description="Pharmaceutical batch number")
    expiry_date: Optional[str] = Field(None, description="Expiry date (YYYY-MM-DD)")
    drug_name: Optional[str] = Field(None, description="Drug name (generic or brand)")
    manufacturer: Optional[str] = Field(None, description="Manufacturer name")
    regulatory_license: Optional[str] = Field(None, description="Regulatory license number")
    storage_conditions: Optional[str] = Field(None, description="Storage conditions (e.g., '2-8°C')")
    hazard_class: Optional[str] = Field(None, description="Hazard classification")
    controlled_substance: bool = Field(False, description="Whether this is a controlled substance")
    prescription_required: bool = Field(False, description="Whether prescription is required")

class CreateInvoiceRequest(BaseModel):
    """Request to create an invoice with all Zoho Books fields."""
    # Required fields
    customer_id: str = Field(..., description="Contact ID (customer ID)")
    line_items: List[InvoiceItem] = Field(..., description="List of invoice line items")
    
    # Invoice details
    invoice_number: Optional[str] = Field(None, description="Custom invoice number")
    date: Optional[str] = Field(None, description="Invoice date (YYYY-MM-DD)")
    due_date: Optional[str] = Field(None, description="Due date (YYYY-MM-DD)")
    reference_number: Optional[str] = Field(None, description="Reference number (PO number, etc.)")
    
    # Currency and exchange
    currency_id: Optional[str] = Field(None, description="Currency ID")
    currency_code: Optional[str] = Field(None, description="Currency code (e.g., 'USD', 'SEK')")
    exchange_rate: Optional[float] = Field(None, description="Exchange rate")
    
    # Payment terms
    payment_terms: Optional[int] = Field(None, description="Payment terms in days")
    payment_terms_label: Optional[str] = Field(None, description="Payment terms label")
    
    # Discount
    discount: Optional[float] = Field(None, description="Discount amount")
    discount_type: Optional[str] = Field(None, description="Discount type: entity_level or item_level")
    is_discount_before_tax: bool = Field(True, description="Apply discount before tax")
    is_inclusive_tax: Optional[bool] = Field(None, description="Whether tax is inclusive")
    
    # Charges and adjustments
    shipping_charge: Optional[float] = Field(None, description="Shipping charge")
    adjustment: Optional[float] = Field(None, description="Adjustment amount")
    adjustment_description: Optional[str] = Field(None, description="Adjustment description")
    
    # Notes and terms
    notes: Optional[str] = Field(None, description="Invoice notes")
    terms: Optional[str] = Field(None, description="Payment terms and conditions")
    custom_body: Optional[str] = Field(None, description="Custom email body")
    custom_subject: Optional[str] = Field(None, description="Custom email subject")
    
    # Contact persons
    contact_persons: Optional[List[str]] = Field(None, description="Contact person IDs")
    contact_persons_associated: Optional[List[Dict[str, Any]]] = Field(None, description="Contact persons with communication preferences")
    
    # Tax information
    tax_id: Optional[str] = Field(None, description="Tax ID")
    tax_authority_id: Optional[str] = Field(None, description="Tax authority ID")
    tax_exemption_id: Optional[str] = Field(None, description="Tax exemption ID")
    place_of_supply: Optional[str] = Field(None, description="Place of supply")
    vat_treatment: Optional[str] = Field(None, description="VAT treatment")
    tax_treatment: Optional[str] = Field(None, description="Tax treatment")
    gst_treatment: Optional[str] = Field(None, description="GST treatment")
    gst_no: Optional[str] = Field(None, description="GST number")
    is_reverse_charge_applied: Optional[bool] = Field(None, description="Whether reverse charge is applied")
    
    # Location and templates
    location_id: Optional[str] = Field(None, description="Location ID")
    template_id: Optional[str] = Field(None, description="Template ID")
    
    # Related documents
    invoiced_estimate_id: Optional[str] = Field(None, description="Estimate ID (if converting from estimate)")
    recurring_invoice_id: Optional[str] = Field(None, description="Recurring invoice ID")
    
    # Salesperson
    salesperson_name: Optional[str] = Field(None, description="Salesperson name")
    
    # Addresses
    billing_address_id: Optional[str] = Field(None, description="Billing address ID")
    shipping_address_id: Optional[str] = Field(None, description="Shipping address ID")
    
    # Payment options
    payment_options: Optional[Dict[str, Any]] = Field(None, description="Payment options")
    allow_partial_payments: Optional[bool] = Field(None, description="Allow partial payments")
    batch_payments: Optional[List[Dict[str, Any]]] = Field(None, description="Batch payment details")
    
    # Custom fields and tags
    custom_fields: Optional[List[Dict[str, Any]]] = Field(None, description="Custom fields")
    tags: Optional[List[Dict[str, Any]]] = Field(None, description="Tags")
    
    # Avalara (if applicable)
    avatax_use_code: Optional[str] = Field(None, description="Avalara use code")
    avatax_exempt_no: Optional[str] = Field(None, description="Avalara exemption number")
    avatax_tax_code: Optional[str] = Field(None, description="Avalara tax code")
    
    # CFDI (Mexico)
    cfdi_usage: Optional[str] = Field(None, description="CFDI usage code (Mexico)")
    
    # Delivery
    delivery_method: Optional[str] = Field(None, description="Delivery method")
    
    # Reason (for updates/voids)
    reason: Optional[str] = Field(None, description="Reason for update/void")

class InvoiceFromContentRequest(BaseModel):
    """Request to create an invoice from a CMS content item (e.g. PDF/Image)."""
    content_id: str = Field(..., description="CMS Content ID of the source document")
    customer_id: Optional[str] = Field(None, description="Customer ID to associate (optional)")
    instruction: Optional[str] = Field(None, description="Additional instructions for extraction")
    auto_send: bool = Field(False, description="Automatically send after creation if confident")

class UpdateInvoiceRequest(BaseModel):
    """Request to update an invoice."""
    customer_id: Optional[str] = None
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    line_items: Optional[List[InvoiceItem]] = None
    notes: Optional[str] = None
    terms: Optional[str] = None
    currency_code: Optional[str] = None
    exchange_rate: Optional[float] = None
    discount: Optional[float] = None
    is_discount_before_tax: Optional[bool] = None
    shipping_charge: Optional[float] = None
    adjustment: Optional[float] = None
    custom_fields: Optional[List[Dict[str, Any]]] = None

class CustomerRequest(BaseModel):
    """Request to create/update a customer (contact) with all Zoho Books fields."""
    # Required field
    customer_name: str = Field(..., description="Display name for the contact (contact_name in API)")
    
    # Basic information
    company_name: Optional[str] = Field(None, description="Legal or registered company name. Max-length [200]")
    contact_type: Optional[str] = Field("customer", description="Type: customer or vendor")
    customer_sub_type: Optional[str] = Field(None, description="For Customer: individual or business")
    website: Optional[str] = Field(None, description="Official website URL")
    language_code: Optional[str] = Field(None, description="Language code (e.g., 'en', 'sv' for Swedish)")
    
    # Contact person details
    first_name: Optional[str] = Field(None, description="First name of contact person. Max-length [100]")
    last_name: Optional[str] = Field(None, description="Last name of contact person. Max-length [100]")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number. Max-length [50]")
    mobile: Optional[str] = Field(None, description="Mobile number. Max-length [50]")
    
    # Financial settings
    credit_limit: Optional[float] = Field(None, description="Maximum credit amount allowed")
    currency_id: Optional[str] = Field(None, description="Currency ID (use currency_code for code)")
    currency_code: Optional[str] = Field(None, description="Currency code (e.g., 'USD', 'SEK', 'EUR')")
    payment_terms: Optional[int] = Field(None, description="Payment terms in days")
    payment_terms_label: Optional[str] = Field(None, description="Payment terms label")
    pricebook_id: Optional[str] = Field(None, description="Pricebook ID")
    
    # Portal settings
    is_portal_enabled: bool = Field(False, description="Enable portal access")
    
    # Addresses
    billing_address: Optional[Dict[str, Any]] = Field(None, description="Billing address object")
    shipping_address: Optional[Dict[str, Any]] = Field(None, description="Shipping address object")
    
    # Contact persons
    contact_persons: Optional[List[Dict[str, Any]]] = Field(None, description="List of contact persons")
    
    # Tax information (region-specific)
    tax_id: Optional[str] = Field(None, description="Tax ID")
    tax_reg_no: Optional[str] = Field(None, description="Tax registration number")
    vat_reg_no: Optional[str] = Field(None, description="VAT registration number")
    gst_no: Optional[str] = Field(None, description="GST number")
    country_code: Optional[str] = Field(None, description="Two-letter country code")
    vat_treatment: Optional[str] = Field(None, description="VAT treatment (UK)")
    tax_treatment: Optional[str] = Field(None, description="Tax treatment")
    gst_treatment: Optional[str] = Field(None, description="GST treatment")
    is_taxable: Optional[bool] = Field(None, description="Whether customer is subject to tax")
    tax_authority_id: Optional[str] = Field(None, description="Tax authority ID")
    tax_exemption_id: Optional[str] = Field(None, description="Tax exemption ID")
    
    # Additional fields
    notes: Optional[str] = Field(None, description="Notes about the customer")
    tags: Optional[List[Dict[str, Any]]] = Field(None, description="Reporting tags")
    owner_id: Optional[str] = Field(None, description="User ID assigned as owner")
    custom_fields: Optional[List[Dict[str, Any]]] = Field(None, description="Custom fields")
    opening_balances: Optional[List[Dict[str, Any]]] = Field(None, description="Opening balance entries")
    
    # Pharmaceutical-specific (stored in notes/custom_fields)
    license_number: Optional[str] = Field(None, description="Pharmaceutical license number")
    regulatory_authority: Optional[str] = Field(None, description="Regulatory authority (e.g., FDA, EMA)")
    
    # Social media
    twitter: Optional[str] = Field(None, description="Twitter handle")
    facebook: Optional[str] = Field(None, description="Facebook page")

class ItemRequest(BaseModel):
    """Request to create/update an item with all Zoho Books fields."""
    # Required fields
    name: str = Field(..., description="Item name")
    rate: float = Field(..., description="Item rate/price")
    
    # Basic information
    description: Optional[str] = Field(None, description="Item description")
    sku: Optional[str] = Field(None, description="SKU (should be unique)")
    product_type: Optional[str] = Field(None, description="goods, service, digital_service, capital_service, capital_goods")
    item_type: str = Field("sales", description="sales, purchases, sales_and_purchases, or inventory")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    
    # Tax information
    tax_id: Optional[str] = Field(None, description="Tax ID")
    tax_percentage: Optional[float] = Field(None, description="Tax percentage")
    is_taxable: Optional[bool] = Field(None, description="Whether item is taxable")
    tax_exemption_id: Optional[str] = Field(None, description="Tax exemption ID")
    purchase_tax_exemption_id: Optional[str] = Field(None, description="Purchase tax exemption ID")
    sales_tax_rule_id: Optional[str] = Field(None, description="Sales tax rule ID")
    purchase_tax_rule_id: Optional[str] = Field(None, description="Purchase tax rule ID")
    item_tax_preferences: Optional[List[Dict[str, Any]]] = Field(None, description="Item tax preferences")
    
    # Accounts
    account_id: Optional[str] = Field(None, description="Sales account ID")
    purchase_account_id: Optional[str] = Field(None, description="COGS account ID (required for purchase/inventory items)")
    inventory_account_id: Optional[str] = Field(None, description="Stock account ID (required for inventory items)")
    
    # Purchase information
    purchase_description: Optional[str] = Field(None, description="Purchase description")
    purchase_rate: Optional[float] = Field(None, description="Purchase price")
    vendor_id: Optional[str] = Field(None, description="Preferred vendor ID")
    
    # Inventory (for inventory items)
    reorder_level: Optional[float] = Field(None, description="Reorder level")
    locations: Optional[List[Dict[str, Any]]] = Field(None, description="Locations with initial stock")
    
    # Tax codes (region-specific)
    hsn_or_sac: Optional[str] = Field(None, description="HSN/SAC code (India, Kenya, South Africa)")
    sat_item_key_code: Optional[str] = Field(None, description="SAT Item Key Code (Mexico)")
    unitkey_code: Optional[str] = Field(None, description="Unit Key Code (Mexico)")
    
    # Avalara (if applicable)
    avatax_tax_code: Optional[str] = Field(None, description="Avalara tax code")
    avatax_use_code: Optional[str] = Field(None, description="Avalara use code")
    
    # Custom fields
    custom_fields: Optional[List[Dict[str, Any]]] = Field(None, description="Custom fields")
    
    # Pharmaceutical-specific fields (stored in custom_fields or description)
    batch_number: Optional[str] = Field(None, description="Pharmaceutical batch number")
    expiry_date: Optional[str] = Field(None, description="Expiry date (YYYY-MM-DD)")
    drug_name: Optional[str] = Field(None, description="Drug name (generic or brand)")
    manufacturer: Optional[str] = Field(None, description="Manufacturer name")
    regulatory_license: Optional[str] = Field(None, description="Regulatory license number")
    storage_conditions: Optional[str] = Field(None, description="Storage conditions")
    hazard_class: Optional[str] = None
    controlled_substance: bool = False
    prescription_required: bool = False


# ============================================================================
# ORCHESTRATOR UNIFIED INTERFACE MODELS (v2)
# ============================================================================

class AgentResponseStatus(str, Enum):
    SUCCESS = "success"
    COMPLETE = "complete"
    FAILED = "failed"
    ERROR = "error"
    NEEDS_INPUT = "needs_input"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"

class OrchestratorMessage(BaseModel):
    """Standardized message from orchestrator to agent."""
    type: Literal["execute", "continue", "cancel", "context_update"] = "execute"
    action: Optional[str] = None
    prompt: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    """Standardized response from agent to orchestrator."""
    status: AgentResponseStatus
    result: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    question: Optional[str] = None
    question_type: Optional[Literal["choice", "text", "confirmation"]] = None
    context: Optional[Dict[str, Any]] = None

class PaymentInvoiceItem(BaseModel):
    """Invoice item in payment request."""
    invoice_id: str = Field(..., description="Invoice ID")
    amount_applied: float = Field(..., description="Amount applied to this invoice")

class PaymentRequest(BaseModel):
    """Request to record a payment with all Zoho Books fields."""
    # Required fields
    customer_id: str = Field(..., description="Customer (contact) ID")
    payment_mode: str = Field(..., description="Payment mode: cash, check, creditcard, banktransfer, bankremittance, autotransaction, paypal, or others")
    amount: float = Field(..., description="Total payment amount")
    date: str = Field(..., description="Payment date (YYYY-MM-DD)")
    invoices: List[PaymentInvoiceItem] = Field(..., description="List of invoices with amounts applied")
    
    # Optional fields
    reference_number: Optional[str] = Field(None, description="Reference number. Max-length [100]")
    description: Optional[str] = Field(None, description="Payment description")
    exchange_rate: Optional[float] = Field(None, description="Exchange rate")
    bank_charges: Optional[float] = Field(None, description="Bank charges")
    payment_form: Optional[str] = Field(None, description="Payment form")
    account_id: Optional[str] = Field(None, description="Account ID where payment is deposited")
    location_id: Optional[str] = Field(None, description="Location ID")
    custom_fields: Optional[List[Dict[str, Any]]] = Field(None, description="Custom fields")
    contact_persons: Optional[List[str]] = Field(None, description="Contact person IDs for thank you email")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Return agent definition."""
    return AGENT_DEFINITION

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        config = load_zoho_config()
        has_token = bool(_token_cache.get("access_token"))
        token_expired = time.time() >= _token_cache.get("expires_at", 0)
        
        return {
            "status": "healthy" if has_token and not token_expired else "needs_oauth",
            "api_domain": config.get("api_domain"),
            "organization_id": config.get("organization_id"),
            "rate_limit_used": _rate_limit["calls_today"],
            "rate_limit_max": 1000,
            "oauth_configured": has_token,
            "token_expired": token_expired,
            "message": "OAuth tokens required. Use /oauth/initialize endpoint to set tokens." if not has_token or token_expired else "Ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/oauth/initialize")
def initialize_oauth(
    access_token: str = Body(...),
    refresh_token: Optional[str] = Body(None),
    expires_in: int = Body(3600)
):
    """
    Initialize OAuth tokens.
    
    To obtain tokens:
    1. Visit https://accounts.zoho.in/developerconsole
    2. Create a Zoho Books API application with OAuth 2.0
    3. Use the OAuth flow to get access_token and refresh_token
    4. Call this endpoint with the tokens
    
    Note: For production, implement proper OAuth callback flow.
    """
    try:
        initialize_oauth_token(access_token, refresh_token, expires_in)
        return {
            "success": True,
            "message": "OAuth tokens initialized successfully"
        }
    except Exception as e:
        logger.error(f"Error initializing OAuth: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize OAuth: {str(e)}")

# ============================================================================
# INVOICE ENDPOINTS
# ============================================================================

@app.post("/invoices")
def create_invoice(request: CreateInvoiceRequest):
    """
    Create a new invoice with pharmaceutical-specific fields.
    """
    try:
        # Prepare line items with all fields
        line_items = []
        for item in request.line_items:
            item_dict = {}
            
            # Required: either item_id OR name+rate
            if item.item_id:
                item_dict["item_id"] = item.item_id
            else:
                if not item.name or item.rate is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Either item_id or both name and rate are required for line items"
                    )
                item_dict["name"] = item.name
                item_dict["rate"] = item.rate
            
            # Add all optional fields
            if item.description:
                item_dict["description"] = item.description
            if item.quantity is not None:
                item_dict["quantity"] = item.quantity
            if item.unit:
                item_dict["unit"] = item.unit
            if item.product_type:
                item_dict["product_type"] = item.product_type
            if item.hsn_or_sac:
                item_dict["hsn_or_sac"] = item.hsn_or_sac
            if item.item_order is not None:
                item_dict["item_order"] = item.item_order
            if item.bcy_rate is not None:
                item_dict["bcy_rate"] = item.bcy_rate
            if item.discount is not None:
                item_dict["discount"] = item.discount
            if item.discount_amount is not None:
                item_dict["discount_amount"] = item.discount_amount
            if item.tags:
                item_dict["tags"] = item.tags
            if item.tax_id:
                item_dict["tax_id"] = item.tax_id
            if item.tax_name:
                item_dict["tax_name"] = item.tax_name
            if item.tax_type:
                item_dict["tax_type"] = item.tax_type
            if item.tax_percentage is not None:
                item_dict["tax_percentage"] = item.tax_percentage
            if item.tds_tax_id:
                item_dict["tds_tax_id"] = item.tds_tax_id
            if item.tax_treatment_code:
                item_dict["tax_treatment_code"] = item.tax_treatment_code
            if item.project_id:
                item_dict["project_id"] = item.project_id
            if item.time_entry_ids:
                item_dict["time_entry_ids"] = item.time_entry_ids
            if item.expense_id:
                item_dict["expense_id"] = item.expense_id
            if item.location_id:
                item_dict["location_id"] = item.location_id
            if item.header_name:
                item_dict["header_name"] = item.header_name
            if item.salesorder_item_id:
                item_dict["salesorder_item_id"] = item.salesorder_item_id
            
            # Add pharmaceutical fields to description or custom fields
            pharma_details = []
            if item.batch_number:
                pharma_details.append(f"Batch: {item.batch_number}")
            if item.expiry_date:
                pharma_details.append(f"Expiry: {item.expiry_date}")
            if item.drug_name:
                pharma_details.append(f"Drug: {item.drug_name}")
            if item.manufacturer:
                pharma_details.append(f"Manufacturer: {item.manufacturer}")
            if item.regulatory_license:
                pharma_details.append(f"License: {item.regulatory_license}")
            if item.storage_conditions:
                pharma_details.append(f"Storage: {item.storage_conditions}")
            if item.hazard_class:
                pharma_details.append(f"Hazard: {item.hazard_class}")
            if item.controlled_substance:
                pharma_details.append("Controlled Substance")
            if item.prescription_required:
                pharma_details.append("Prescription Required")
            
            if pharma_details:
                existing_desc = item_dict.get("description", "")
                pharma_text = " | ".join(pharma_details)
                item_dict["description"] = f"{existing_desc}\n{pharma_text}".strip() if existing_desc else pharma_text
            
            line_items.append(item_dict)
        
        # Build invoice data with all fields
        invoice_data = {
            "customer_id": request.customer_id,
            "line_items": line_items
        }
        
        # Add all optional fields
        if request.invoice_number:
            invoice_data["invoice_number"] = request.invoice_number
        if request.date:
            invoice_data["date"] = request.date
        if request.due_date:
            invoice_data["due_date"] = request.due_date
        if request.reference_number:
            invoice_data["reference_number"] = request.reference_number
        if request.currency_id:
            invoice_data["currency_id"] = request.currency_id
        elif request.currency_code:
            invoice_data["currency_code"] = request.currency_code
        if request.exchange_rate is not None:
            invoice_data["exchange_rate"] = request.exchange_rate
        if request.payment_terms is not None:
            invoice_data["payment_terms"] = request.payment_terms
        if request.payment_terms_label:
            invoice_data["payment_terms_label"] = request.payment_terms_label
        if request.discount is not None:
            invoice_data["discount"] = request.discount
        if request.discount_type:
            invoice_data["discount_type"] = request.discount_type
        if request.is_discount_before_tax is not None:
            invoice_data["is_discount_before_tax"] = request.is_discount_before_tax
        if request.is_inclusive_tax is not None:
            invoice_data["is_inclusive_tax"] = request.is_inclusive_tax
        if request.shipping_charge is not None:
            invoice_data["shipping_charge"] = request.shipping_charge
        if request.adjustment is not None:
            invoice_data["adjustment"] = request.adjustment
        if request.adjustment_description:
            invoice_data["adjustment_description"] = request.adjustment_description
        if request.notes:
            invoice_data["notes"] = request.notes
        if request.terms:
            invoice_data["terms"] = request.terms
        if request.custom_body:
            invoice_data["custom_body"] = request.custom_body
        if request.custom_subject:
            invoice_data["custom_subject"] = request.custom_subject
        if request.contact_persons:
            invoice_data["contact_persons"] = request.contact_persons
        if request.contact_persons_associated:
            invoice_data["contact_persons_associated"] = request.contact_persons_associated
        if request.tax_id:
            invoice_data["tax_id"] = request.tax_id
        if request.tax_authority_id:
            invoice_data["tax_authority_id"] = request.tax_authority_id
        if request.tax_exemption_id:
            invoice_data["tax_exemption_id"] = request.tax_exemption_id
        if request.place_of_supply:
            invoice_data["place_of_supply"] = request.place_of_supply
        if request.vat_treatment:
            invoice_data["vat_treatment"] = request.vat_treatment
        if request.tax_treatment:
            invoice_data["tax_treatment"] = request.tax_treatment
        if request.gst_treatment:
            invoice_data["gst_treatment"] = request.gst_treatment
        if request.gst_no:
            invoice_data["gst_no"] = request.gst_no
        if request.is_reverse_charge_applied is not None:
            invoice_data["is_reverse_charge_applied"] = request.is_reverse_charge_applied
        if request.location_id:
            invoice_data["location_id"] = request.location_id
        if request.template_id:
            invoice_data["template_id"] = request.template_id
        if request.invoiced_estimate_id:
            invoice_data["invoiced_estimate_id"] = request.invoiced_estimate_id
        if request.recurring_invoice_id:
            invoice_data["recurring_invoice_id"] = request.recurring_invoice_id
        if request.salesperson_name:
            invoice_data["salesperson_name"] = request.salesperson_name
        if request.billing_address_id:
            invoice_data["billing_address_id"] = request.billing_address_id
        if request.shipping_address_id:
            invoice_data["shipping_address_id"] = request.shipping_address_id
        if request.payment_options:
            invoice_data["payment_options"] = request.payment_options
        if request.allow_partial_payments is not None:
            invoice_data["allow_partial_payments"] = request.allow_partial_payments
        if request.batch_payments:
            invoice_data["batch_payments"] = request.batch_payments
        if request.custom_fields:
            invoice_data["custom_fields"] = request.custom_fields
        if request.tags:
            invoice_data["tags"] = request.tags
        if request.avatax_use_code:
            invoice_data["avatax_use_code"] = request.avatax_use_code
        if request.avatax_exempt_no:
            invoice_data["avatax_exempt_no"] = request.avatax_exempt_no
        if request.avatax_tax_code:
            invoice_data["avatax_tax_code"] = request.avatax_tax_code
        if request.cfdi_usage:
            invoice_data["cfdi_usage"] = request.cfdi_usage
        if request.delivery_method:
            invoice_data["delivery_method"] = request.delivery_method
        if request.reason:
            invoice_data["reason"] = request.reason
        
        result = make_zoho_request("POST", "/invoices", json_data=invoice_data)
        return {
            "success": True,
            "invoice": result.get("invoice", result),
            "message": "Invoice created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create invoice: {str(e)}")

@app.post("/invoices/from-content")
async def create_invoice_from_content(request: InvoiceFromContentRequest):
    """
    Create an invoice by extracting data from a CMS document (PDF, Image, etc.).
    """
    try:
        logger.info(f"Creating invoice from content: {request.content_id}")
        
        # 1. Retrieve Content from CMS
        try:
            meta, content = cms_service.get_content(request.content_id)
            if not content:
                raise ValueError("Content not found or empty")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to retrieve content: {e}")
            
        # 2. Extract Invoice Data using LLM
        llm_client = get_llm()
        if not llm_client:
             raise HTTPException(status_code=503, detail="LLM service unavailable")
             
        # Prepare context for LLM
        content_preview = str(content)[:20000] # Truncate if massive text
        if isinstance(content, bytes):
             content_preview = "<Binary Data - Metadata: " + str(meta) + ">"
             # In a real scenario, we'd use a multimodal model or proper parser here
             # For now, assuming text/json content or text extracted by CMS
             
        prompt = f"""You are an expert Invoice Data Extractor.
Extract valid invoice details from the following content to populate a Zoho Books Invoice.
Content Metadata: {meta.name} ({meta.content_type.value})
Content:
{content_preview}

Customer ID Hint: {request.customer_id if request.customer_id else 'None'}
Instruction: {request.instruction or 'Extract all line items, totals, and dates.'}

Return valid JSON adhering to the CreateInvoiceRequest schema. 
Ensure 'line_items' has 'name', 'rate', 'quantity'.
If customer_id is not found, exclude it (or use the hint).
JSON Output:"""

        # Call LLM
        try:
            response = await llm_client.ainvoke(prompt)
            json_str = response.content.replace("```json", "").replace("```", "").strip()
            # Simple cleanup for reasoning tags
            import re
            json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL).strip()
            
            invoice_data = json.loads(json_str) 
        except Exception as llm_err:
             logger.error(f"LLM extraction failed: {llm_err}")
             raise HTTPException(status_code=500, detail=f"Failed to extract invoice data: {llm_err}")
             
        # 3. Validation & Merge
        if request.customer_id:
             invoice_data['customer_id'] = request.customer_id
             
        # Validate against schema
        try:
             validated_request = CreateInvoiceRequest(**invoice_data)
        except Exception as val_err:
             raise HTTPException(status_code=422, detail=f"Extracted data invalid: {val_err}")
             
        # 4. Create Invoice
        # reuse the existing function logic or call it directly? 
        # Calling route function directly is tricky due to dependency injection in FastAPI, 
        # but here it's a simple function call if we pass the object.
        return create_invoice(validated_request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating invoice from content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process content invoice: {str(e)}")

@app.get("/invoices")
def list_invoices(
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=200),
    customer_name: Optional[str] = None,
    item_name: Optional[str] = None,
    item_description: Optional[str] = None,
    item_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    invoice_number: Optional[str] = None,
    status: Optional[str] = None,  # sent, draft, overdue, paid, void
    date: Optional[str] = None,  # YYYY-MM-DD
    due_date: Optional[str] = None,  # YYYY-MM-DD
    total: Optional[float] = None,
    balance: Optional[float] = None,
    customer_email: Optional[str] = None,
    search_text: Optional[str] = None,
    sort_column: Optional[str] = None  # customer_name, invoice_number, date, total, balance, created_time
):
    """
    List invoices with filtering options.
    """
    try:
        params = {
            "page": page,
            "per_page": per_page
        }
        
        # Add filters
        if customer_name:
            params["customer_name"] = customer_name
        if item_name:
            params["item_name"] = item_name
        if item_description:
            params["item_description"] = item_description
        if item_id:
            params["item_id"] = item_id
        if customer_id:
            params["customer_id"] = customer_id
        if invoice_number:
            params["invoice_number"] = invoice_number
        if status:
            params["status"] = status
        if date:
            params["date"] = date
        if due_date:
            params["due_date"] = due_date
        if total:
            params["total"] = total
        if balance:
            params["balance"] = balance
        if customer_email:
            params["customer_email"] = customer_email
        if search_text:
            params["search_text"] = search_text
        if sort_column:
            params["sort_column"] = sort_column
        
        result = make_zoho_request("GET", "/invoices", params=params)
        invoices = result.get("invoices", [])
        
        # Generate Canvas: Spreadsheet View
        headers = ["Date", "Invoice#", "Customer", "Total", "Balance", "Status", "Due Date"]
        rows = []
        for inv in invoices:
            rows.append([
                inv.get("date"),
                inv.get("invoice_number"),
                inv.get("customer_name"),
                inv.get("total"),
                inv.get("balance"),
                inv.get("status"),
                inv.get("due_date")
            ])
            
        canvas = CanvasService.build_spreadsheet_view(
            filename="invoices_list.csv",
            headers=headers,
            rows=rows,
            title=f"Invoices ({len(invoices)})"
        )

        return {
            "success": True,
            "invoices": invoices,
            "page_context": result.get("page_context", {}),
            "standard_response": {
                "canvas_display": canvas.model_dump()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing invoices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list invoices: {str(e)}")

@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    """
    Get invoice details by ID.
    """
    try:
        result = make_zoho_request("GET", f"/invoices/{invoice_id}")
        inv = result.get("invoice", result)
        
        # Generate Canvas: Document View (Markdown Receipt)
        md_content = f"""# Invoice {inv.get('invoice_number')}
**Date:** {inv.get('date')} | **Due:** {inv.get('due_date')}
**Customer:** {inv.get('customer_name')}
**Status:** {inv.get('status')}

## Line Items
| Item | Qty | Rate | Total |
|------|-----|------|-------|
"""
        for item in inv.get('line_items', []):
            md_content += f"| {item.get('name')} | {item.get('quantity')} | {item.get('rate')} | {item.get('item_total')} |\n"
            
        md_content += f"""
---
**Subtotal:** {inv.get('sub_total')}
**Tax:** {inv.get('tax_total')}
**Total:** {inv.get('total')} {inv.get('currency_code')}
"""

        canvas = CanvasService.build_document_view(
            content=md_content,
            title=f"Invoice {inv.get('invoice_number')}",
            format="markdown"
        )

        return {
            "success": True,
            "invoice": inv,
            "standard_response": {
                "canvas_display": canvas.model_dump()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get invoice: {str(e)}")

@app.put("/invoices/{invoice_id}")
def update_invoice(invoice_id: str, request: UpdateInvoiceRequest):
    """
    Update an existing invoice.
    """
    try:
        invoice_data = {}
        
        if request.customer_id:
            invoice_data["customer_id"] = request.customer_id
        if request.invoice_number:
            invoice_data["invoice_number"] = request.invoice_number
        if request.date:
            invoice_data["date"] = request.date
        if request.due_date:
            invoice_data["due_date"] = request.due_date
        if request.line_items:
            line_items = []
            for item in request.line_items:
                item_dict = {
                    "item_id": item.item_id,
                    "name": item.name,
                    "description": item.description,
                    "rate": item.rate,
                    "quantity": item.quantity,
                    "unit": item.unit
                }
                item_dict = add_pharmaceutical_fields({
                    **item_dict,
                    "batch_number": item.batch_number,
                    "expiry_date": item.expiry_date,
                    "drug_name": item.drug_name,
                    "manufacturer": item.manufacturer,
                    "regulatory_license": item.regulatory_license,
                    "storage_conditions": item.storage_conditions,
                    "hazard_class": item.hazard_class,
                    "controlled_substance": item.controlled_substance,
                    "prescription_required": item.prescription_required
                })
                line_items.append(item_dict)
            invoice_data["line_items"] = line_items
        if request.notes is not None:
            invoice_data["notes"] = request.notes
        if request.terms is not None:
            invoice_data["terms"] = request.terms
        if request.currency_code:
            invoice_data["currency_code"] = request.currency_code
        if request.exchange_rate:
            invoice_data["exchange_rate"] = request.exchange_rate
        if request.discount is not None:
            invoice_data["discount"] = request.discount
            invoice_data["is_discount_before_tax"] = request.is_discount_before_tax
        if request.shipping_charge:
            invoice_data["shipping_charge"] = request.shipping_charge
        if request.adjustment:
            invoice_data["adjustment"] = request.adjustment
        if request.custom_fields:
            invoice_data["custom_fields"] = request.custom_fields
        
        result = make_zoho_request("PUT", f"/invoices/{invoice_id}", json_data=invoice_data)
        return {
            "success": True,
            "invoice": result.get("invoice", result),
            "message": "Invoice updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update invoice: {str(e)}")

@app.delete("/invoices/{invoice_id}")
def delete_invoice(invoice_id: str):
    """
    Delete an invoice.
    """
    try:
        result = make_zoho_request("DELETE", f"/invoices/{invoice_id}")
        return {
            "success": True,
            "message": "Invoice deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete invoice: {str(e)}")

@app.post("/invoices/{invoice_id}/send")
def send_invoice(
    invoice_id: str,
    send_from_org_email_id: Optional[bool] = Body(False),
    customer_id: Optional[str] = None,
    email_ids: Optional[List[str]] = None,
    cc_email_ids: Optional[List[str]] = None,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    send_attachment: bool = True
):
    """
    Send invoice via email.
    """
    try:
        send_data = {
            "send_from_org_email_id": send_from_org_email_id,
            "send_attachment": send_attachment
        }
        
        if customer_id:
            send_data["customer_id"] = customer_id
        if email_ids:
            send_data["email_ids"] = email_ids
        if cc_email_ids:
            send_data["cc_email_ids"] = cc_email_ids
        if subject:
            send_data["subject"] = subject
        if body:
            send_data["body"] = body
        
        result = make_zoho_request("POST", f"/invoices/{invoice_id}/email", json_data=send_data)
        return {
            "success": True,
            "message": "Invoice sent successfully",
            "response": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to send invoice: {str(e)}")

@app.post("/invoices/{invoice_id}/mark-as-sent")
def mark_invoice_as_sent(invoice_id: str):
    """
    Mark invoice as sent.
    """
    try:
        result = make_zoho_request("POST", f"/invoices/{invoice_id}/status/sent")
        return {
            "success": True,
            "message": "Invoice marked as sent",
            "invoice": result.get("invoice", result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking invoice as sent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to mark invoice as sent: {str(e)}")

@app.post("/invoices/{invoice_id}/mark-as-paid")
def mark_invoice_as_paid(invoice_id: str, amount: float = Body(...)):
    """
    Mark invoice as paid (full or partial payment).
    """
    try:
        result = make_zoho_request("POST", f"/invoices/{invoice_id}/payments", json_data={
            "amount": amount
        })
        return {
            "success": True,
            "message": "Invoice marked as paid",
            "payment": result.get("payment", result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking invoice as paid: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to mark invoice as paid: {str(e)}")

# ============================================================================
# CUSTOMER ENDPOINTS
# ============================================================================

@app.post("/customers")
def create_customer(request: CustomerRequest):
    """
    Create a new customer.
    """
    try:
        # Validate and sanitize customer_name
        customer_name = (request.customer_name or "").strip()
        if not customer_name:
            raise HTTPException(
                status_code=400,
                detail="customer_name is required and cannot be empty or whitespace"
            )
        if len(customer_name) > 200:  # Zoho Books typically has a 200 char limit
            raise HTTPException(
                status_code=400,
                detail=f"customer_name is too long (max 200 characters, got {len(customer_name)})"
            )
        
        # Sanitize company_name - Zoho Books requires this field
        company_name = (request.company_name or customer_name).strip()
        if not company_name:
            company_name = customer_name
        
        # Check for duplicate customer names - Zoho Books might reject duplicates
        # Add a small unique suffix if needed (but let's try without first)
        # Note: If you get duplicate errors, you may need to check existing customers first
        
        # Build payload - Zoho Books API format
        # Note: Some Zoho Books regions/versions might use "contact_name" instead of "customer_name"
        # Try customer_name first (standard), but we'll fall back if needed
        # Zoho Books API uses /contacts endpoint with contact_name (display name) - REQUIRED
        payload_to_send = {
            "contact_name": customer_name,  # Display name (required field)
            "company_name": company_name,
            "contact_type": "customer"  # Specify this is a customer, not vendor
        }
        
        # Log for debugging
        logger.info(f"Attempting contact creation with contact_name field (display name)")
        
        # Log the exact values being sent for debugging
        logger.info(f"Customer name (len={len(customer_name)}): '{customer_name}'")
        logger.info(f"Company name (len={len(company_name)}): '{company_name}'")
        
        # Add all optional fields from the comprehensive model
        if request.contact_type:
            payload_to_send["contact_type"] = request.contact_type
        if request.customer_sub_type:
            payload_to_send["customer_sub_type"] = request.customer_sub_type
        if request.website:
            payload_to_send["website"] = request.website
        if request.language_code:
            payload_to_send["language_code"] = request.language_code
        if request.first_name:
            payload_to_send["first_name"] = request.first_name
        if request.last_name:
            payload_to_send["last_name"] = request.last_name
        if request.email:
            payload_to_send["email"] = request.email
        if request.phone:
            payload_to_send["phone"] = request.phone
        if request.mobile:
            payload_to_send["mobile"] = request.mobile
        if request.credit_limit is not None:
            payload_to_send["credit_limit"] = request.credit_limit
        if request.currency_id:
            payload_to_send["currency_id"] = request.currency_id
        elif request.currency_code:
            payload_to_send["currency_code"] = request.currency_code
        if request.payment_terms is not None:
            payload_to_send["payment_terms"] = request.payment_terms
        if request.payment_terms_label:
            payload_to_send["payment_terms_label"] = request.payment_terms_label
        if request.pricebook_id:
            payload_to_send["pricebook_id"] = request.pricebook_id
        if request.is_portal_enabled:
            payload_to_send["is_portal_enabled"] = request.is_portal_enabled
        if request.billing_address:
            payload_to_send["billing_address"] = request.billing_address
        if request.shipping_address:
            payload_to_send["shipping_address"] = request.shipping_address
        if request.contact_persons:
            payload_to_send["contact_persons"] = request.contact_persons
        if request.tax_id:
            payload_to_send["tax_id"] = request.tax_id
        if request.tax_reg_no:
            payload_to_send["tax_reg_no"] = request.tax_reg_no
        if request.vat_reg_no:
            payload_to_send["vat_reg_no"] = request.vat_reg_no
        if request.gst_no:
            payload_to_send["gst_no"] = request.gst_no
        if request.country_code:
            payload_to_send["country_code"] = request.country_code
        if request.vat_treatment:
            payload_to_send["vat_treatment"] = request.vat_treatment
        if request.tax_treatment:
            payload_to_send["tax_treatment"] = request.tax_treatment
        if request.gst_treatment:
            payload_to_send["gst_treatment"] = request.gst_treatment
        if request.is_taxable is not None:
            payload_to_send["is_taxable"] = request.is_taxable
        if request.tax_authority_id:
            payload_to_send["tax_authority_id"] = request.tax_authority_id
        if request.tax_exemption_id:
            payload_to_send["tax_exemption_id"] = request.tax_exemption_id
        if request.owner_id:
            payload_to_send["owner_id"] = request.owner_id
        if request.tags:
            payload_to_send["tags"] = request.tags
        if request.custom_fields:
            payload_to_send["custom_fields"] = request.custom_fields
        if request.opening_balances:
            payload_to_send["opening_balances"] = request.opening_balances
        if request.twitter:
            payload_to_send["twitter"] = request.twitter
        if request.facebook:
            payload_to_send["facebook"] = request.facebook
        
        # Add pharmaceutical-specific fields to notes
        pharma_info = []
        if request.license_number:
            pharma_info.append(f"License: {request.license_number}")
        if request.regulatory_authority:
            pharma_info.append(f"Regulatory Authority: {request.regulatory_authority}")
        
        # Combine with existing notes
        existing_notes = request.notes or ""
        if pharma_info:
            pharma_notes = " | ".join(pharma_info)
            payload_to_send["notes"] = f"{existing_notes}\n{pharma_notes}".strip() if existing_notes else pharma_notes
        elif existing_notes:
            payload_to_send["notes"] = existing_notes
            payload_to_send["notes"] = pharma_notes
        
        # Before creating, check if contact already exists (to avoid duplicate error)
        try:
            existing_contacts = make_zoho_request("GET", "/contacts", params={
                "search_text": customer_name,
                "per_page": 10,
                "contact_type": "customer"  # Filter for customers only
            })
            if existing_contacts.get("contacts"):
                for existing in existing_contacts["contacts"]:
                    if existing.get("contact_name", "").strip().lower() == customer_name.lower():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Contact with name '{customer_name}' already exists (ID: {existing.get('contact_id')}). Please use a different name or update the existing contact."
                        )
        except HTTPException as e:
            # If search fails, continue anyway (might be permission issue)
            if "already exists" in str(e.detail):
                raise
            logger.warning(f"Could not check for existing contacts: {e.detail}")
        
        # Zoho Books API uses /contacts endpoint, not /customers
        logger.info(f"Sending contact creation request: {json.dumps(payload_to_send, indent=2)}")
        result = make_zoho_request("POST", "/contacts", json_data=payload_to_send)
        return {
            "success": True,
            "contact": result.get("contact", result),
            "message": "Contact (customer) created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create customer: {str(e)}")

@app.get("/customers")
def list_customers(
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=200),
    search_text: Optional[str] = None,
    sort_column: Optional[str] = None
):
    """
    List customers (contacts with contact_type=customer).
    """
    try:
        params = {
            "page": page,
            "per_page": per_page,
            "contact_type": "customer"  # Filter for customers only
        }
        
        if search_text:
            params["search_text"] = search_text
        if sort_column:
            params["sort_column"] = sort_column
        
        result = make_zoho_request("GET", "/contacts", params=params)
        # Generate Canvas: Spreadsheet View
        customers = result.get("contacts", [])
        headers = ["Name", "Company", "Email", "Phone", "Type"]
        rows = []
        for cust in customers:
            rows.append([
                cust.get("contact_name"),
                cust.get("company_name"),
                cust.get("email"),
                cust.get("phone"),
                cust.get("contact_type")
            ])
            
        canvas = CanvasService.build_spreadsheet_view(
            filename="customers_list.csv",
            headers=headers,
            rows=rows,
            title=f"Customers ({len(customers)})"
        )

        return {
            "success": True,
            "customers": customers,
            "page_context": result.get("page_context", {}),
            "standard_response": {
                "canvas_display": canvas.model_dump()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing customers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list customers: {str(e)}")

@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    """
    Get customer (contact) details by ID.
    """
    try:
        result = make_zoho_request("GET", f"/contacts/{customer_id}")
        return {
            "success": True,
            "customer": result.get("contact", result)  # Zoho returns "contact" not "customer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get customer: {str(e)}")

@app.put("/customers/{customer_id}")
def update_customer(customer_id: str, request: CustomerRequest):
    """
    Update an existing customer.
    """
    try:
        customer_data = {}
        
        if request.customer_name:
            customer_data["contact_name"] = request.customer_name  # Zoho uses contact_name
        if request.company_name:
            customer_data["company_name"] = request.company_name
        if request.customer_type:
            customer_data["customer_type"] = request.customer_type
        if request.customer_sub_type:
            customer_data["customer_sub_type"] = request.customer_sub_type
        if request.credit_limit is not None:
            customer_data["credit_limit"] = request.credit_limit
        if request.is_portal_enabled is not None:
            customer_data["is_portal_enabled"] = request.is_portal_enabled
        if request.language:
            customer_data["language"] = request.language
        if request.currency_code:
            customer_data["currency_code"] = request.currency_code
        if request.payment_terms:
            customer_data["payment_terms"] = request.payment_terms
        if request.payment_terms_label:
            customer_data["payment_terms_label"] = request.payment_terms_label
        if request.billing_address:
            customer_data["billing_address"] = request.billing_address
        if request.shipping_address:
            customer_data["shipping_address"] = request.shipping_address
        if request.contact_persons:
            customer_data["contact_persons"] = request.contact_persons
        
        # Add pharmaceutical-specific fields to notes
        pharma_info = []
        if request.license_number:
            pharma_info.append(f"License: {request.license_number}")
        if request.regulatory_authority:
            pharma_info.append(f"Regulatory Authority: {request.regulatory_authority}")
        if request.tax_id:
            pharma_info.append(f"Tax ID: {request.tax_id}")
        
        # Append pharmaceutical info to notes if provided
        if pharma_info:
            existing_notes = customer_data.get("notes", "")
            pharma_notes = " | ".join(pharma_info)
            customer_data["notes"] = f"{existing_notes}\n{pharma_notes}".strip() if existing_notes else pharma_notes
        
        # Zoho Books API uses /contacts endpoint
        result = make_zoho_request("PUT", f"/contacts/{customer_id}", json_data=customer_data)
        return {
            "success": True,
            "customer": result.get("contact", result),  # Zoho returns "contact" not "customer"
            "message": "Customer updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating customer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update customer: {str(e)}")

# ============================================================================
# ITEM ENDPOINTS
# ============================================================================

@app.post("/items")
def create_item(request: ItemRequest):
    """
    Create a new item with all Zoho Books fields plus pharmaceutical-specific fields.
    """
    try:
        # Required fields
        item_data = {
            "name": request.name,
            "rate": request.rate
        }
        
        # Add all optional fields
        if request.description:
            item_data["description"] = request.description
        if request.sku:
            item_data["sku"] = request.sku
        if request.product_type:
            item_data["product_type"] = request.product_type
        if request.item_type:
            item_data["item_type"] = request.item_type
        if request.unit:
            item_data["unit"] = request.unit
        if request.tax_id:
            item_data["tax_id"] = request.tax_id
        if request.tax_percentage is not None:
            item_data["tax_percentage"] = request.tax_percentage
        if request.is_taxable is not None:
            item_data["is_taxable"] = request.is_taxable
        if request.tax_exemption_id:
            item_data["tax_exemption_id"] = request.tax_exemption_id
        if request.purchase_tax_exemption_id:
            item_data["purchase_tax_exemption_id"] = request.purchase_tax_exemption_id
        if request.sales_tax_rule_id:
            item_data["sales_tax_rule_id"] = request.sales_tax_rule_id
        if request.purchase_tax_rule_id:
            item_data["purchase_tax_rule_id"] = request.purchase_tax_rule_id
        if request.item_tax_preferences:
            item_data["item_tax_preferences"] = request.item_tax_preferences
        if request.account_id:
            item_data["account_id"] = request.account_id
        if request.purchase_account_id:
            item_data["purchase_account_id"] = request.purchase_account_id
        if request.inventory_account_id:
            item_data["inventory_account_id"] = request.inventory_account_id
        if request.purchase_description:
            item_data["purchase_description"] = request.purchase_description
        if request.purchase_rate is not None:
            item_data["purchase_rate"] = request.purchase_rate
        if request.vendor_id:
            item_data["vendor_id"] = request.vendor_id
        if request.reorder_level is not None:
            item_data["reorder_level"] = request.reorder_level
        if request.locations:
            item_data["locations"] = request.locations
        if request.hsn_or_sac:
            item_data["hsn_or_sac"] = request.hsn_or_sac
        if request.sat_item_key_code:
            item_data["sat_item_key_code"] = request.sat_item_key_code
        if request.unitkey_code:
            item_data["unitkey_code"] = request.unitkey_code
        if request.avatax_tax_code:
            item_data["avatax_tax_code"] = request.avatax_tax_code
        if request.avatax_use_code:
            item_data["avatax_use_code"] = request.avatax_use_code
        if request.custom_fields:
            item_data["custom_fields"] = request.custom_fields
        
        # Add pharmaceutical fields to description or custom fields
        pharma_details = []
        if request.batch_number:
            pharma_details.append(f"Batch: {request.batch_number}")
        if request.expiry_date:
            pharma_details.append(f"Expiry: {request.expiry_date}")
        if request.drug_name:
            pharma_details.append(f"Drug: {request.drug_name}")
        elif request.name:  # Use name as drug name if not specified
            pharma_details.append(f"Drug: {request.name}")
        if request.manufacturer:
            pharma_details.append(f"Manufacturer: {request.manufacturer}")
        if request.regulatory_license:
            pharma_details.append(f"License: {request.regulatory_license}")
        if request.storage_conditions:
            pharma_details.append(f"Storage: {request.storage_conditions}")
        
        if pharma_details:
            existing_desc = item_data.get("description", "")
            pharma_text = " | ".join(pharma_details)
            item_data["description"] = f"{existing_desc}\n{pharma_text}".strip() if existing_desc else pharma_text
        
        result = make_zoho_request("POST", "/items", json_data=item_data)
        return {
            "success": True,
            "item": result.get("item", result),
            "message": "Item created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create item: {str(e)}")

@app.get("/items")
def list_items(
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=200),
    name: Optional[str] = None,
    description: Optional[str] = None,
    item_type: Optional[str] = None,
    search_text: Optional[str] = None,
    sort_column: Optional[str] = None
):
    """
    List items.
    """
    try:
        params = {
            "page": page,
            "per_page": per_page
        }
        
        if name:
            params["name"] = name
        if description:
            params["description"] = description
        if item_type:
            params["item_type"] = item_type
        if search_text:
            params["search_text"] = search_text
        if sort_column:
            params["sort_column"] = sort_column
        
        result = make_zoho_request("GET", "/items", params=params)
        # Generate Canvas: Spreadsheet View
        items = result.get("items", [])
        headers = ["Name", "Rate", "Stock", "SKU", "Status"]
        rows = []
        for item in items:
            rows.append([
                item.get("name"),
                item.get("rate"),
                item.get("stock_on_hand"),
                item.get("sku"),
                item.get("status")
            ])
            
        canvas = CanvasService.build_spreadsheet_view(
            filename="items_list.csv",
            headers=headers,
            rows=rows,
            title=f"Items ({len(items)})"
        )

        return {
            "success": True,
            "items": items,
            "page_context": result.get("page_context", {}),
            "standard_response": {
                "canvas_display": canvas.model_dump()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing items: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list items: {str(e)}")

@app.get("/items/{item_id}")
def get_item(item_id: str):
    """
    Get item details by ID.
    """
    try:
        result = make_zoho_request("GET", f"/items/{item_id}")
        return {
            "success": True,
            "item": result.get("item", result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get item: {str(e)}")

@app.put("/items/{item_id}")
def update_item(item_id: str, request: ItemRequest):
    """
    Update an existing item.
    """
    try:
        item_data = {}
        
        if request.name:
            item_data["name"] = request.name
        if request.description is not None:
            item_data["description"] = request.description
        if request.rate is not None:
            item_data["rate"] = request.rate
        if request.unit:
            item_data["unit"] = request.unit
        if request.item_type:
            item_data["item_type"] = request.item_type
        if request.product_type:
            item_data["product_type"] = request.product_type
        if request.sku:
            item_data["sku"] = request.sku
        if request.hsn_or_sac:
            item_data["hsn_or_sac"] = request.hsn_or_sac
        if request.tax_id:
            item_data["tax_id"] = request.tax_id
        if request.account_id:
            item_data["account_id"] = request.account_id
        
        # Add pharmaceutical fields
        item_data = add_pharmaceutical_fields({
            **item_data,
            "batch_number": request.batch_number,
            "expiry_date": request.expiry_date,
            "drug_name": request.drug_name,
            "manufacturer": request.manufacturer,
            "regulatory_license": request.regulatory_license,
            "storage_conditions": request.storage_conditions,
            "hazard_class": request.hazard_class,
            "controlled_substance": request.controlled_substance,
            "prescription_required": request.prescription_required
        })
        
        result = make_zoho_request("PUT", f"/items/{item_id}", json_data=item_data)
        return {
            "success": True,
            "item": result.get("item", result),
            "message": "Item updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating item: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update item: {str(e)}")

# ============================================================================
# PAYMENT ENDPOINTS
# ============================================================================

@app.post("/payments")
def create_payment(request: PaymentRequest):
    """
    Record a payment with all Zoho Books fields.
    """
    try:
        # Build invoices array with invoice_id and amount_applied
        invoices_array = []
        for invoice_item in request.invoices:
            invoices_array.append({
                "invoice_id": invoice_item.invoice_id,
                "amount_applied": invoice_item.amount_applied
            })
        
        payment_data = {
            "customer_id": request.customer_id,
            "payment_mode": request.payment_mode,
            "amount": request.amount,
            "date": request.date,
            "invoices": invoices_array
        }
        
        # Add all optional fields
        if request.reference_number:
            payment_data["reference_number"] = request.reference_number
        if request.description:
            payment_data["description"] = request.description
        if request.exchange_rate is not None:
            payment_data["exchange_rate"] = request.exchange_rate
        if request.bank_charges is not None:
            payment_data["bank_charges"] = request.bank_charges
        if request.payment_form:
            payment_data["payment_form"] = request.payment_form
        if request.account_id:
            payment_data["account_id"] = request.account_id
        if request.location_id:
            payment_data["location_id"] = request.location_id
        if request.custom_fields:
            payment_data["custom_fields"] = request.custom_fields
        if request.contact_persons:
            payment_data["contact_persons"] = request.contact_persons
        
        result = make_zoho_request("POST", "/customerpayments", json_data=payment_data)
        return {
            "success": True,
            "payment": result.get("payment", result),
            "message": "Payment recorded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating payment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create payment: {str(e)}")

@app.get("/payments")
def list_payments(
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=200),
    customer_id: Optional[str] = None,
    invoice_id: Optional[str] = None,
    payment_mode: Optional[str] = None,
    date: Optional[str] = None,
    search_text: Optional[str] = None
):
    """
    List payments.
    """
    try:
        params = {
            "page": page,
            "per_page": per_page
        }
        
        if customer_id:
            params["customer_id"] = customer_id
        if invoice_id:
            params["invoice_id"] = invoice_id
        if payment_mode:
            params["payment_mode"] = payment_mode
        if date:
            params["date"] = date
        if search_text:
            params["search_text"] = search_text
        
        result = make_zoho_request("GET", "/customerpayments", params=params)
        return {
            "success": True,
            "payments": result.get("payments", []),
            "page_context": result.get("page_context", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing payments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list payments: {str(e)}")

@app.get("/payments/{payment_id}")
def get_payment(payment_id: str):
    """
    Get payment details by ID.
    """
    try:
        result = make_zoho_request("GET", f"/customerpayments/{payment_id}")
        return {
            "success": True,
            "payment": result.get("payment", result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting payment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get payment: {str(e)}")

# ============================================================================
# ESTIMATE ENDPOINTS (for pharmaceutical quotes)
# ============================================================================

@app.post("/estimates")
def create_estimate(request: CreateInvoiceRequest):
    """
    Create an estimate (quote) with pharmaceutical fields.
    """
    try:
        # Reuse invoice creation logic but for estimates
        line_items = []
        for item in request.line_items:
            item_dict = {
                "item_id": item.item_id,
                "name": item.name,
                "description": item.description,
                "rate": item.rate,
                "quantity": item.quantity,
                "unit": item.unit
            }
            item_dict = add_pharmaceutical_fields({
                **item_dict,
                "batch_number": item.batch_number,
                "expiry_date": item.expiry_date,
                "drug_name": item.drug_name,
                "manufacturer": item.manufacturer,
                "regulatory_license": item.regulatory_license,
                "storage_conditions": item.storage_conditions,
                "hazard_class": item.hazard_class,
                "controlled_substance": item.controlled_substance,
                "prescription_required": item.prescription_required
            })
            line_items.append(item_dict)
        
        estimate_data = {
            "customer_id": request.customer_id,
            "line_items": line_items,
            "notes": request.notes,
            "terms": request.terms,
            "currency_code": request.currency_code or "USD"
        }
        
        if request.date:
            estimate_data["date"] = request.date
        if request.due_date:
            estimate_data["expiry_date"] = request.due_date
        if request.exchange_rate:
            estimate_data["exchange_rate"] = request.exchange_rate
        if request.discount is not None:
            estimate_data["discount"] = request.discount
            estimate_data["is_discount_before_tax"] = request.is_discount_before_tax
        if request.shipping_charge:
            estimate_data["shipping_charge"] = request.shipping_charge
        if request.adjustment:
            estimate_data["adjustment"] = request.adjustment
        if request.custom_fields:
            estimate_data["custom_fields"] = request.custom_fields
        
        result = make_zoho_request("POST", "/estimates", json_data=estimate_data)
        return {
            "success": True,
            "estimate": result.get("estimate", result),
            "message": "Estimate created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating estimate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create estimate: {str(e)}")

@app.get("/estimates")
def list_estimates(
    page: int = Query(1, ge=1),
    per_page: int = Query(200, ge=1, le=200),
    customer_id: Optional[str] = None,
    status: Optional[str] = None,  # sent, accepted, declined, invoiced
    search_text: Optional[str] = None
):
    """
    List estimates.
    """
    try:
        params = {
            "page": page,
            "per_page": per_page
        }
        
        if customer_id:
            params["customer_id"] = customer_id
        if status:
            params["status"] = status
        if search_text:
            params["search_text"] = search_text
        
        result = make_zoho_request("GET", "/estimates", params=params)
        return {
            "success": True,
            "estimates": result.get("estimates", []),
            "page_context": result.get("page_context", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing estimates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list estimates: {str(e)}")

@app.post("/estimates/{estimate_id}/convert-to-invoice")
def convert_estimate_to_invoice(estimate_id: str):
    """
    Convert an estimate to an invoice.
    """
    try:
        result = make_zoho_request("POST", f"/estimates/{estimate_id}/converttoinvoice")
        return {
            "success": True,
            "invoice": result.get("invoice", result),
            "message": "Estimate converted to invoice successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting estimate to invoice: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to convert estimate to invoice: {str(e)}")

# ============================================================================
# UNIFIED ORCHESTRATOR INTERFACE (v2)
# ============================================================================

@app.post("/execute", response_model=AgentResponse)
async def execute(message: OrchestratorMessage):
    """
    Unified entry point for the Orchestrator.
    Handles routing to granular endpoints and natural language processing.
    """
    logger.info(f"🚀 Unified Execute: type={message.type}, action={message.action}, prompt={message.prompt[:50] if message.prompt else None}")
    
    try:
        # 1. Route based on type
        if message.type == "cancel":
            return AgentResponse(status=AgentResponseStatus.COMPLETE, message="Task cancelled")
            
        if message.type == "context_update":
            return AgentResponse(status=AgentResponseStatus.COMPLETE, message="Context updated")
            
        if message.type == "continue":
            return await continue_task(message)

        # 2. Extract Action and Payload
        action = message.action or ""
        payload = message.payload or {}
        prompt = message.prompt or ""

        method = "POST"
        
        # 3. Autonomous Planning (if prompt provided and no explicit action)
        if not action and prompt:
            try:
                plan = await get_planner().plan(prompt)
                action = plan.action
                payload = plan.payload
                method = plan.method
                logger.info(f"📍 Planned Action: {action}, Method: {method}, Params: {payload.keys()}")
            except Exception as e:
                logger.error(f"Planning error: {e}")
                # Fallback handled by individual checks or return error


        # 4. Delegate to existing endpoints
        # Note: In a production scenario, we'd use an LLM (InferenceService) to map prompt -> function call
        # For now, we manually map common actions.
        
        if action == "/invoices":
            if method == "GET" or (message.type == "execute" and not payload):
                # GET/List
                res = list_invoices()
                return AgentResponse(
                    status=AgentResponseStatus.COMPLETE,
                    result=res,
                    standard_response=StandardAgentResponse(
                        status="success",
                        summary=f"Found {len(res.get('invoices', []))} invoices",
                        data=res,
                        canvas_display=res.get("standard_response", {}).get("canvas_display")
                    )
                )
            else:
                # POST/Create
                req = CreateInvoiceRequest(**payload)
                res = create_invoice(req)
                return AgentResponse(
                    status=AgentResponseStatus.COMPLETE,
                    result=res,
                    standard_response=StandardAgentResponse(
                        status="success",
                        summary="Invoice created",
                        data=res
                    )
                )
                
        if action == "/customers":
            if method == "GET" or (message.type == "execute" and not payload):
                res = list_customers()
                return AgentResponse(
                    status=AgentResponseStatus.COMPLETE,
                    result=res,
                    standard_response=StandardAgentResponse(
                        status="success",
                        summary=f"Found {len(res.get('contacts', []))} customers",
                        data=res
                    )
                )
            else:
                req = CustomerRequest(**payload)
                res = create_customer(req)
                return AgentResponse(
                     status=AgentResponseStatus.COMPLETE,
                     result=res,
                     standard_response=StandardAgentResponse(
                        status="success",
                        summary="Customer created",
                        data=res
                    )
                )

        if action == "/items":
            if method == "GET" or (message.type == "execute" and not payload):
                res = list_items()
                return AgentResponse(
                    status=AgentResponseStatus.COMPLETE,
                    result=res,
                    standard_response=StandardAgentResponse(
                        status="success",
                        summary=f"Found {len(res.get('items', []))} items",
                        data=res,
                        canvas_display=res.get("standard_response", {}).get("canvas_display")
                    )
                )
            else:
                req = ItemRequest(**payload)
                res = create_item(req)
                return AgentResponse(
                    status=AgentResponseStatus.COMPLETE,
                    result=res,
                    standard_response=StandardAgentResponse(
                        status="success",
                        summary="Item created",
                        data=res
                    )
                )

        if action == "/payments":
            # Basic implementation for payments if not already there
             if method == "GET" or (message.type == "execute" and not payload):
                 # Assuming a list_payments function exists or returning placeholder
                 return AgentResponse(status=AgentResponseStatus.COMPLETE, result={"message": "Payment listing not implemented yet"})
             else:
                 return AgentResponse(status=AgentResponseStatus.COMPLETE, result={"message": "Payment creation not implemented yet"})

        # Fallback error
        return AgentResponse(
            status=AgentResponseStatus.ERROR, 
            error=f"Unsupported action: {action}. Please specify /invoices, /customers, /items, or /payments."
        )

    except Exception as e:
        logger.error(f"Error in unified execute: {e}", exc_info=True)
        return AgentResponse(
            status=AgentResponseStatus.ERROR, 
            error=str(e),
            standard_response=StandardAgentResponse(
                status="error",
                summary="Zoho Agent Execution Failed",
                error_message=str(e)
            )
        )

@app.post("/continue", response_model=AgentResponse)
async def continue_task(message: OrchestratorMessage):
    """Handle multi-turn interactions for unified protocol."""
    return AgentResponse(
        status=AgentResponseStatus.ERROR, 
        error="Multi-turn conversation not yet implemented for ZohoBooksAgent. Please use direct actions."
    )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("zoho_books_agent:app", host="0.0.0.0", port=8050, reload=False)


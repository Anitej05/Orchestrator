import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import json
import sys
from pathlib import Path

# Add backend to path if needed
backend_root = Path(__file__).parent.parent.parent.resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.services.inference_service import inference_service

logger = logging.getLogger(__name__)

class ZohoActionPlan(BaseModel):
    action: str = Field(..., description="The endpoint action: /invoices, /customers, /items, /payments")
    method: str = Field("POST", description="HTTP-like method: GET for list/search, POST for create/update")
    payload: Dict[str, Any] = Field(..., description="Parameters/Body for the request. Empty for list/search unless filtering.")
    reasoning: str = Field(..., description="Reasoning for the choice")

class ZohoPlanner:
    """
    Autonomous planner for the Zoho Books Agent.
    Converts natural language prompts into structured API calls.
    """
    
    def __init__(self):
        self.inference = inference_service
        self.model_id = "gpt-oss-120b" 

    async def plan(self, prompt: str) -> ZohoActionPlan:
        """
        Analyze the prompt and return a structured plan.
        """
        logger.info(f"Planning for Zoho prompt: {prompt}")
        
        system_prompt = """You are the Planner for a Zoho Books Agent (Accounting).
Your goal is to map a user's natural language request to one of the following actions:

1. `/invoices`: Create or list invoices.
2. `/customers`: Create or list customers/contacts.
3. `/items`: Create or list items (products/services).
4. `/payments`: Record or list payments.

Method Guidelines:
- Use `GET` if the user wants to "list", "search", "find", "show", or "get".
- Use `POST` if the user wants to "create", "generate", "make", "add", or "new".

Payload Schema:
- For `POST` /invoices: {"customer_id": "...", "line_items": [{"item_id": "...", "rate": 100}], ...}
- For `POST` /customers: {"contact_name": "...", "company_name": "...", ...}
- For `POST` /items: {"name": "...", "rate": 100, ...}
- For `GET` (all): {} (Empty payload usually, unless filtering params are needed)

Reason step-by-step then provide the JSON plan.
If the request is ambiguous, default to `GET` (list) for safety.
If customer/item names are given but IDs are needed, make a best guess or use placeholders that the agent might verify later (for now, extracting names is fine).
"""

        try:
            # Use structured output for reliability
            plan = await self.inference.generate_structured(
                prompt=prompt,
                response_model=ZohoActionPlan,
                system_prompt=system_prompt,
                model=self.model_id,
                temperature=0.1
            )
            
            logger.info(f"Generated Zoho Plan: {plan.action} ({plan.method}) - {plan.reasoning}")
            return plan
            
        except Exception as e:
            logger.error(f"Zoho Planning failed: {e}")
            # Fallback: Default to list invoices safest
            return ZohoActionPlan(
                action="/invoices",
                method="GET",
                payload={},
                reasoning="Planning failed, defaulting to list invoices."
            )

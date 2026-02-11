import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load env immediately
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add backend to path
backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.agents.document_agent_lib import get_agent as get_doc_agent, get_planner as get_doc_planner
from backend.agents.zoho_books.zoho_books_agent import get_planner as get_zoho_planner
from backend.schemas import OrchestratorMessage

async def test_document_planner():
    print("\n--- Testing Document Planner ---")
    planner = get_doc_planner()
    
    prompts = [
        "Summarize the Q3 financial report",
        "Create a new document called notes.txt with content 'Meeting notes'",
        "Edit the executive summary to be more concise"
    ]
    
    for p in prompts:
        print(f"\nPrompt: '{p}'")
        plan = await planner.plan(p)
        print(f"Plan: Action={plan.action}, Params={plan.params}, Reasoning={getattr(plan, 'reasoning', 'N/A')}")
        
        # Verify
        if "Summarize" in p:
            assert plan.action == "/analyze", f"Expected /analyze, got {plan.action}"
        elif "Create" in p:
            assert plan.action == "/create", f"Expected /create, got {plan.action}"
        elif "Edit" in p:
            assert plan.action == "/edit", f"Expected /edit, got {plan.action}"

async def test_zoho_planner():
    print("\n--- Testing Zoho Planner ---")
    planner = get_zoho_planner()
    
    prompts = [
        "Create an invoice for Acme Corp for $500 consulting services",
        "List all invoices for John Doe",
        "Add a new customer named Tech Solutions"
    ]
    
    for p in prompts:
        print(f"\nPrompt: '{p}'")
        plan = await planner.plan(p)
        print(f"Plan: Action={plan.action}, Method={plan.method}, Payload={plan.payload}")
        
        # Verify
        if "invoice" in p and "Create" in p:
             assert plan.action == "/invoices"
             assert plan.method == "POST"
        elif "List" in p:
             assert plan.action == "/invoices"
             assert plan.method == "GET"
        elif "customer" in p:
             assert plan.action == "/customers"

async def main():
    await test_document_planner()
    await test_zoho_planner()
    print("\n✅ All autonomous tests passed!")

if __name__ == "__main__":
    asyncio.run(main())

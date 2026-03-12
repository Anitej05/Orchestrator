import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

async def test_agent(agent_class, name, prompt):
    print(f"\n{'='*50}\nTesting {name}\n{'='*50}")
    try:
        agent = agent_class()
        await agent.initialize()
        print(f"[{name}] Initialized successfully.")
        
        from backend.agents.base.types import AgentRequest
        request = AgentRequest(
            prompt=prompt,
            thread_id="test_thread"
        )
        
        print(f"[{name}] Executing prompt: '{prompt}'...")
        response = await agent.execute(request)
        print(f"[{name}] Response Status: {response.status}")
        
        if response.status == "error":
            print(f"[{name}] Error: {response.error_message}")
        else:
            print(f"[{name}] Summary: {getattr(response, 'summary', 'N/A')}")
            print(f"[{name}] Result Preview: {str(response.result)[:200]}")
            
    except Exception as e:
        import traceback
        print(f"[{name}] Exception during test:")
        traceback.print_exc()

async def main():
    from backend.agents.gmail_agent.base_agent_impl import GmailAgent
    from backend.agents.zoho_books.base_agent_impl import ZohoBooksAgent
    
    await test_agent(GmailAgent, "Gmail Agent", "Draft an email to test@example.com with subject 'Testing Agents' and body 'Hello World'. Do not send.")
    # await test_agent(ZohoBooksAgent, "Zoho Books Agent", "What are the latest invoices?")

if __name__ == "__main__":
    asyncio.run(main())

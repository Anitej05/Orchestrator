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
    from backend.agents.coding_agent.agent import CodingAgent
    from backend.agents.integrations_agent.base_agent_impl import IntegrationsAgent
    from backend.agents.universal_agent.base_agent_impl import UniversalAgent
    
    await test_agent(CodingAgent, "Coding Agent", "Write a simple python function to add two numbers and explain it.")
    await test_agent(IntegrationsAgent, "Integrations Agent", "What integrations are supported?")
    await test_agent(UniversalAgent, "Universal Agent", "What is 2+2?")

if __name__ == "__main__":
    asyncio.run(main())

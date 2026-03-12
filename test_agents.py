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
    from backend.agents.spreadsheet_agent.base_agent_impl import SpreadsheetAgent
    
    from backend.agents.document_agent_lib.base_agent_impl import DocumentAgent
    from backend.agents.ppt_agent.base_agent_impl import PPTAgent
    from backend.agents.spreadsheet_agent.base_agent_impl import SpreadsheetAgent
    
    await test_agent(DocumentAgent, "Document Agent", "Create a new document titled 'Test Doc' with the content 'Hello world from test script'.")
    await test_agent(PPTAgent, "PPT Agent", "Create a new presentation with 1 slide. Title is 'Test', content is 'Hello world'.")
    
    with open("test_data.csv", "w") as f:
        f.write("A,B\n1,10\n2,20\n3,30\n")
    await test_agent(SpreadsheetAgent, "Spreadsheet Agent", "Load test_data.csv and tell me the sum of column B")

if __name__ == "__main__":
    asyncio.run(main())

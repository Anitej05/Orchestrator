import os
import sys
import asyncio
import traceback
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from backend.agents.pdf_agent.base_agent_impl import PDFAgent
from backend.agents.base.types import AgentRequest

async def test_pdf_agent():
    print("Initializing PDF Agent...")
    agent = PDFAgent(agent_id="test_pdf_agent", agent_name="PDF Agent Test")
    await agent.initialize()

    # Let's ask it what it can do
    request = AgentRequest(
        prompt="Hi, just verifying you are alive. What can you do?",
        thread_id="test_thread",
        user_id="test_user"
    )
    
    print("\nExecuting test prompt...")
    try:
        response = await agent.execute(request)
        print("Response received")
    except Exception as e:
        print("Exception caught in caller!")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pdf_agent())

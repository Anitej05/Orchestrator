import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

async def test_universal_agent_complex():
    print("Initializing Universal Agent...")
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(env_path)
        print("Loaded environment variables.")
    except ImportError:
        pass
    
    from agents.universal_agent.base_agent_impl import UniversalAgent
    from agents.base.services import AgentServices
    from agents.base.types import AgentRequest

    services = AgentServices.create_default()
    agent = UniversalAgent(services=services)
    await agent.initialize()

    tasks = [
        "Write a Python script that uses asyncio to concurrently fetch data from 3 different mock JSON APIs and write the aggregated results to a CSV file. Include clear error handling for network timeouts.",
        "Design a system architecture for a real-time multiplayer browser game. Explain the components and the communication protocols (WebSocket vs WebRTC).",
        "If I have a dataset of customer reviews, explain step-by-step how I would build a sentiment analysis pipeline using HuggingFace transformers, including the preprocessing steps."
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'='*80}\nTask {i}: {task}\n{'='*80}")
        try:
            result = await agent.execute(AgentRequest(prompt=task))
            print(f"\nStatus: {result.status}")
            print(f"Summary: {result.summary}")
            
            answer = ""
            if result.result and 'answer' in result.result:
                answer = result.result['answer']
            elif result.result:
                answer = str(result.result)
                
            if answer:
                print(f"\nResponse:\n{answer}")
            else:
                print("No clear answer returned in result.")
        except Exception as e:
            print(f"Error executing task {i}: {e}")

    await agent.terminate()
    print("\nAll tasks completed.")

if __name__ == "__main__":
    asyncio.run(test_universal_agent_complex())

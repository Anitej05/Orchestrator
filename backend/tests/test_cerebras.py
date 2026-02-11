import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

backend_root = Path(__file__).resolve().parents[1]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from backend.services.inference_service import inference_service, ProviderType
from langchain_core.messages import HumanMessage

async def main():
    print("Testing Cerebras Connection...")
    try:
        model = "llama3.1-8b"
        print(f"Requesting model: {model}")
        
        response = await inference_service.generate(
            messages=[HumanMessage(content="Hello, are you there?")],
            model_name=model,
            provider=ProviderType.CEREBRAS,
            fallback_enabled=False
        )
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())

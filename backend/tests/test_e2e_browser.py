import sys
import os
import asyncio
import logging
import uuid
import time
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

logging.basicConfig(level=logging.INFO)

from backend.tests.test_e2e_orchestration import run_e2e_test

async def main():
    print("🚀 Starting End-to-End Orchestrator -> Browser Agent Test")
    
    result = await run_e2e_test(
        test_name="Browser Agent - Simple Navigation",
        prompt="Go to 'example.com' and tell me the heading on the page.",
        expected_agent="browser",
        timeout=300,
    )
    
    print("\n✅ Final Result:")
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

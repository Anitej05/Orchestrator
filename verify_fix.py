
import asyncio
import sys
import os
from pathlib import Path

# Setup paths to match agent environment
backend_root = Path("d:/Internship/Orbimesh/backend").resolve()
agent_root = backend_root / "agents" / "spreadsheet_agent"

sys.path.insert(0, str(backend_root))
sys.path.insert(0, str(agent_root))

# Mock some dependencies if needed, or just import
import pandas as pd

async def verify():
    from client import DataFrameClient
    client = DataFrameClient()
    file_path = "d:/Internship/Orbimesh/storage/spreadsheet_agent/spreadsheet.xlsx"
    
    print(f"Testing load_file with {file_path}")
    try:
        df, detection = await client.load_file(file_path=file_path)
        print(f"SUCCESS (Unexpected): Loaded {df.shape}")
        print(f"Head: {df.head()}")
    except Exception as e:
        print(f"CAUGHT EXPECTED ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(verify())

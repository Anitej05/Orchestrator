
import asyncio
import pandas as pd
import sys
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR / "backend"))
sys.path.insert(0, str(SCRIPT_DIR / "backend" / "agents" / "spreadsheet_agent"))

from backend.agents.spreadsheet_agent.client import DataFrameClient

async def reproduce():
    client = DataFrameClient()
    file_path = "d:/Internship/Orbimesh/storage/spreadsheet_agent/spreadsheet.xlsx"
    
    print(f"--- Testing load_file with {file_path} ---")
    try:
        # We know this file is small (4984 bytes)
        df, detection = await client.load_file(file_path=file_path)
        print(f"Detection Info: {detection}")
        print(f"DataFrame Shape: {df.shape}")
        print("\nHead of DataFrame:")
        print(df.head())
        
        # Check if first value looks like binary
        first_val = str(df.iloc[0, 0])
        if "PK" in first_val or any(ord(c) > 127 for c in first_val[:100]):
            print("\n!!! REPRODUCED: DataFrame contains binary-looking junk !!!")
        else:
            print("\nDataFrame seems to contain normal text (based on first 100 chars of A1)")
            
    except Exception as e:
        print(f"Error during load: {e}")

if __name__ == "__main__":
    asyncio.run(reproduce())

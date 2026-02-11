import sys
from pathlib import Path
import os

# Define paths explicitly
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parent.parent # d:/Internship/Orbimesh/backend
PROJECT_ROOT = BACKEND_DIR.parent # d:/Internship/Orbimesh

print(f"Debug: Project Root: {PROJECT_ROOT}")
print(f"Debug: Backend Dir: {BACKEND_DIR}")

# Setup Path mimicking agent wrapper
sys.path.insert(0, str(BACKEND_DIR)) # For 'import agents'
sys.path.insert(0, str(PROJECT_ROOT)) # For 'import backend'

print("--- sys.path ---")
for p in sys.path[:5]:
    print(p)
print("----------------")

try:
    print("Attempting: from backend.agents.utils.standard_file_interface import AgentFileMetadata")
    from backend.agents.utils.standard_file_interface import AgentFileMetadata
    print("✅ Success: agents.utils...")
except Exception as e:
    print(f"❌ Failed: {e}")

try:
    print("Attempting: from backend.agents.utils.standard_file_interface import AgentFileMetadata")
    from backend.agents.utils.standard_file_interface import AgentFileMetadata
    print("✅ Success: backend.agents.utils...")
except Exception as e:
    print(f"❌ Failed: {e}")

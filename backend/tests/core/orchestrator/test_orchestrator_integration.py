
import asyncio
import os
import sys
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load env vars
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OrchestratorTest")

async def test_orchestrator_flow():
    from backend.orchestrator.graph import graph
    from backend.orchestrator.state import State
    from langchain_core.messages import HumanMessage
    
    print("🚀 Starting Orchestrator Integration Test")
    
    # --- Turn 1: Download Syllabus ---
    prompt1 = "Download VNR VJIET Syllabus for AIML R22 as AIML_R22.pdf"
    print(f"\n📝 Turn 1 Prompt: {prompt1}")
    
    initial_state = {
        "original_prompt": prompt1,
        "messages": [HumanMessage(content=prompt1)],
        "uploaded_files": [],
        "planning_mode": False,  # Skip planning for speed/direct execution check
        "thread_id": "test_integration_001"
    }
    
    # Run the graph
    # We use ainvoke which returns the final state
    config = {"configurable": {"thread_id": "test_integration_001"}}
    
    # Mocking graph execution for safety if credentials are missing, 
    # but we want to try the Real Thing if possible. 
    # If this fails due to missing keys/deps, we'll know.
    try:
        final_state_1 = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        print(f"❌ Graph execution failed: {e}")
        return

    # Verify File Capture
    uploaded_files = final_state_1.get("uploaded_files", [])
    print(f"\n📂 Uploaded Files in State after Turn 1: {len(uploaded_files)}")
    
    found_pdf = False
    for f in uploaded_files:
        path = f.get('file_path') if isinstance(f, dict) else f.file_path
        print(f" - {path}")
        if "AIML_R22.pdf" in str(path):
            found_pdf = True
            
    if found_pdf:
        print("✅ SUCCESS: AIML_R22.pdf was captured in state!")
    else:
        print("❌ FAILURE: AIML_R22.pdf was NOT found in state.")
        # We might continue just to see what happens, but getting here is the crucial fix verification.
        
    # --- Turn 2: Read Document ---
    if found_pdf:
        prompt2 = "Read the downloaded document and tell me what it is about."
        print(f"\n📝 Turn 2 Prompt: {prompt2}")
        
        # Determine how to continue. Standard LangGraph continuation.
        # We update the state with the new message and keeping the uploaded_files
        
        # Create a new human message
        new_messages = final_state_1["messages"] + [HumanMessage(content=prompt2)]
        
        next_state = {
            "messages": new_messages,
            "original_prompt": prompt2,
            "uploaded_files": uploaded_files, # Explicitly passing them ensures they are there
            "planning_mode": False,
            "thread_id": "test_integration_001"
        }

        try:
            final_state_2 = await graph.ainvoke(next_state, config=config)
            print("\n🤖 Final Response Turn 2:")
            print(final_state_2.get("final_response"))
        except Exception as e:
            print(f"❌ Turn 2 failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_flow())

"""
Diagnostic test for agent startup issues
"""

import asyncio
import subprocess
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def diagnose_agent(agent_id, port):
    """Diagnose agent startup issues."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {agent_id}")
    print(f"{'='*70}")
    
    from backend.services.agent_manager import ProcessManager, AgentManager
    
    backend_dir = Path(__file__).parent / "backend"
    pm = ProcessManager(backend_dir)
    
    print(f"\n1. Starting {agent_id} on port {port}...")
    try:
        process = await pm.start_agent(agent_id, port)
        print(f"   Process started: PID {process.pid}")
        
        # Wait a bit for startup
        print(f"   Waiting 5 seconds for startup...")
        await asyncio.sleep(5)
        
        # Check if process is still running
        if pm.is_running(process):
            print(f"   Process still running: YES")
        else:
            print(f"   Process still running: NO (crashed?)")
            stdout, stderr = process.communicate()
            if stdout:
                print(f"\n   STDOUT:\n{stdout.decode()[-500:]}")
            if stderr:
                print(f"\n   STDERR:\n{stderr.decode()[-500:]}")
            return
        
        # Try to connect to health endpoint
        print(f"\n2. Checking health endpoint...")
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/health",
                    timeout=5.0
                )
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   Connection failed: {e}")
        
        # Stop the process
        print(f"\n3. Stopping process...")
        await pm.stop_agent(process)
        print(f"   Stopped")
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Diagnose failing agents."""
    print("AGENT STARTUP DIAGNOSTICS")
    
    # Diagnose mail agent
    await diagnose_agent('mail', 8040)
    
    # Diagnose zoho_books agent
    await diagnose_agent('zoho_books', 8060)


if __name__ == "__main__":
    asyncio.run(main())

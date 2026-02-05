import subprocess
import sys
import time
import requests
import os
import signal
import psutil

# Define agents to test (Name, Script Path, Port)
AGENTS = [
    {
        "name": "Spreadsheet Agent",
        "script": "backend/agents/spreadsheet_agent.py",
        "port": 9000,
        "health_url": "http://localhost:9000/docs"
    },
    {
        "name": "Document Agent",
        "script": "backend/agents/document_agent.py",
        "port": 8070,
        "health_url": "http://localhost:8070/docs"
    },
    {
        "name": "Mail Agent",
        "script": "backend/agents/mail_agent.py",
        "port": 8040,
        "health_url": "http://localhost:8040/docs"
    }
]

def kill_process_on_port(port):
    """Find and kill any process using the specified port (Windows specific)."""
    try:
        # Find PID using netstat
        # Output format: TCP    0.0.0.0:9000           0.0.0.0:0              LISTENING       1234
        cmd = f"netstat -ano | findstr :{port}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout:
            return # No process found
            
        lines = result.stdout.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.split()
            if len(parts) > 4:
                pid = parts[-1]
                pids.add(pid)
        
        for pid in pids:
            if pid == "0": continue
            print(f"    Killing existing process on port {port} (PID: {pid})...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            
    except Exception as e:
        print(f"    Warning: Failed to cleanup port {port}: {e}")

def test_agent(agent):
    print(f"\n--- Testing {agent['name']} ---")
    
    # 1. Cleanup Port
    kill_process_on_port(agent['port'])
    
    # 2. Start Agent
    print(f"    Starting {agent['script']} on port {agent['port']}...")
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() # Ensure root is in path
    
    # Use shell=True for Windows compatibility with python command if needed, but list is safer
    cmd = [sys.executable, agent['script']]
    
    try:
        # Start process
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 3. Wait for startup
        print("    Waiting 10s for startup...")
        time.sleep(10)
        
        # Check if process died early
        if proc.poll() is not None:
             stdout, stderr = proc.communicate()
             print(f"    ❌ Agent died immediately!")
             print(f"    STDOUT: {stdout.decode('utf-8', errors='ignore')[:200]}")
             print(f"    STDERR: {stderr.decode('utf-8', errors='ignore')[:200]}")
             return False

        # 4. Ping Health
        try:
            print(f"    Pinging {agent['health_url']}...")
            resp = requests.get(agent['health_url'], timeout=5)
            if resp.status_code == 200:
                print(f"    ✅ Agent is ONLINE (Status 200 OK)")
                success = True
            else:
                print(f"    ❌ Agent returned status {resp.status_code}")
                success = False
        except requests.exceptions.ConnectionError:
            print("    ❌ Connection Refused (Port closed)")
            success = False
        except Exception as e:
            print(f"    ❌ Error pinging: {e}")
            success = False

        # 5. Cleanup
        print("    Stopping agent...")
        proc.terminate()
        try:
             proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
             proc.kill()
             
        # Double check port is free
        kill_process_on_port(agent['port'])
        
        return success

    except Exception as e:
        print(f"    ❌ Test Execution Error: {e}")
        return False

def main():
    print("=== Individual Agent Lifecycle Test ===")
    results = {}
    for agent in AGENTS:
        results[agent['name']] = test_agent(agent)
    
    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
        if not passed: all_passed = False
        
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()


import subprocess
import os
import logging
import uuid
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("TerminalService")

class TerminalService:
    """
    A service that allows the Orchestrator to execute shell commands.
    Acts as the "Hands" of the agent on the computer.
    """
    
    def __init__(self, base_dir: str = None):
        # Default to backend/storage to give access to all agent outputs
        if not base_dir:
            backend_root = Path(__file__).parent.parent
            base_dir = backend_root / "storage"
        
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_cwd = self.base_dir
        
        logger.info(f"TerminalService initialized at {self.base_dir}")

    def execute_command(self, command: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute a shell command in the current working directory.
        """
        try:
            logger.info(f"Executing command: {command} in {self.current_cwd}")
            
            # Handle 'cd' manually because subprocess doesn't persist state
            if command.strip().startswith("cd "):
                target_dir = command.strip()[3:].strip()
                new_path = (self.current_cwd / target_dir).resolve()
                
                # Check security (optional enforcement, allowing "computer use" implies trust)
                # For now, let's just update the CWD
                if new_path.exists() and new_path.is_dir():
                    self.current_cwd = new_path
                    return {"stdout": f"Changed directory to {self.current_cwd}", "stderr": "", "returncode": 0}
                else:
                    return {"stdout": "", "stderr": f"Directory not found: {target_dir}", "returncode": 1}

            # Run actual command
            process = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_cwd),
                capture_output=True,
                text=True,
                check=False
            )
            
            # If a file was created, we might want to detect it, but for now just return output
            return {
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode,
                "cwd": str(self.current_cwd)
            }
            
        except Exception as e:
            logger.error(f"Terminal execution error: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

# Singleton instance
terminal_service = TerminalService()

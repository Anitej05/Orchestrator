import sys
import os
# Add backend root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.terminal_service import terminal_service

print(f"CWD: {terminal_service.current_cwd}")
res = terminal_service.execute_command("echo Hello > test_iso.txt")
print(f"Result: {res}")

check_path = terminal_service.current_cwd / "test_iso.txt"
print(f"File exists at {check_path}? {check_path.exists()}")

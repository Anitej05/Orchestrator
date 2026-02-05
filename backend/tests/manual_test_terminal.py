
import unittest
from unittest.mock import MagicMock, patch
import sys
import io
import os
from pathlib import Path

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from services.terminal_service import TerminalService

class TestTerminalService(unittest.TestCase):
    def setUp(self):
        # Initialize with a dummy base dir
        self.test_dir = Path("./test_storage").resolve()
        self.service = TerminalService(base_dir=str(self.test_dir))
        
    def test_cd_logic(self):
        print("\n=== Testing CD Logic ===")
        # 1. Create a subdir
        subdir = self.test_dir / "subdir"
        subdir.mkdir(exist_ok=True, parents=True)
        
        # 2. CD into it
        resp = self.service.execute_command("cd subdir")
        print(f"CD Response: {resp}")
        
        self.assertEqual(resp['returncode'], 0)
        self.assertEqual(self.service.current_cwd, subdir)
        print("✅ CD Command Verified")
        
        # 3. CD into invalid
        resp = self.service.execute_command("cd invalid_dir")
        self.assertNotEqual(resp['returncode'], 0)
        self.assertIn("not found", resp['stderr'])
        print("✅ Invalid CD Handled")

    @patch('subprocess.run')
    def test_shell_command(self, mock_run):
        print("\n=== Testing Shell Command ===")
        
        # Mock subprocess result
        mock_result = MagicMock()
        mock_result.stdout = "Hello World"
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        resp = self.service.execute_command("echo Hello")
        
        print(f"Response: {resp}")
        
        mock_run.assert_called_once()
        self.assertEqual(resp['stdout'], "Hello World")
        self.assertEqual(resp['cwd'], str(self.service.current_cwd))
        print("✅ Shell Command Verified")

if __name__ == "__main__":
    unittest.main()

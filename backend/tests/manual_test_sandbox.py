
import unittest
import sys
import io
import os

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.code_sandbox_service import CodeSandboxService

class TestCodeSandbox(unittest.TestCase):
    def setUp(self):
        self.sandbox = CodeSandboxService()

    def test_basic_execution(self):
        print("\n=== Testing Basic Execution ===")
        code = "print('Hello Sandbox')\nresult = 10 + 5"
        resp = self.sandbox.execute_code(code)
        
        print(f"Stdout: {resp.get('stdout')}")
        print(f"Result: {resp.get('result')}")
        
        self.assertTrue(resp['success'])
        self.assertIn("Hello Sandbox", resp['stdout'])
        self.assertEqual(resp['result'], 15)
        print("✅ Basic Execution Verified")

    def test_persistence(self):
        print("\n=== Testing Persistence ===")
        session_id = "test-session"
        
        # Step 1: Set variable
        self.sandbox.execute_code("x = 42", session_id=session_id)
        
        # Step 2: Use variable
        resp = self.sandbox.execute_code("result = x * 2", session_id=session_id)
        
        print(f"Result: {resp.get('result')}")
        self.assertTrue(resp['success'])
        self.assertEqual(resp['result'], 84)
        print("✅ State Persistence Verified")

    def test_error_handling(self):
        print("\n=== Testing Error Handling ===")
        code = "result = 1 / 0"
        resp = self.sandbox.execute_code(code)
        
        print(f"Error: {resp.get('error')}")
        self.assertFalse(resp['success'])
        self.assertIn("division by zero", resp['error'])
        print("✅ Error Handling Verified")

    def test_library_access(self):
        print("\n=== Testing Library Access (pandas) ===")
        code = """
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
print(df)
result = df
"""
        resp = self.sandbox.execute_code(code)
        
        print(f"Stdout:\n{resp.get('stdout')}")
        print(f"Result: {resp.get('result')}")
        
        self.assertTrue(resp['success'])
        self.assertIn("<DataFrame shape=(2, 2)>", str(resp['result'])) # Serializer behavior
        print("✅ Library (Pandas) Verified")

if __name__ == "__main__":
    unittest.main()

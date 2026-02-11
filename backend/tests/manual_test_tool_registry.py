
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import io
import asyncio
import os

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set path (assumes run from root or backend)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.tool_registry_service import ToolRegistryService, ToolDefinition

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        # Patch telemetry
        self.telemetry_patcher = patch('services.tool_registry_service.telemetry')
        self.mock_telemetry = self.telemetry_patcher.start()
        
        self.registry = ToolRegistryService()
        # Skip real initialization
        self.registry._initialized = True 
        
    def tearDown(self):
        self.telemetry_patcher.stop()

    async def test_execution_flow(self):
        print("\n=== Testing Tool Execution Flow ===")
        
        # Create a Mock Tool
        mock_tool_instance = AsyncMock()
        mock_tool_instance.ainvoke.return_value = "Success Result"
        
        # Inject into registry
        tool_def = ToolDefinition(
            function_name="mock_tool",
            display_name="Mock Tool",
            description="Does nothing",
            category="test",
            parameters=[],
            use_when="",
            not_for="",
            keywords=set(),
            tool_instance=mock_tool_instance
        )
        self.registry._tools["mock_tool"] = tool_def
        
        # Execute
        result = await self.registry.execute_tool("mock_tool", {"param": 1})
        
        print(f"Result: {result}")
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], "Success Result")
        
        # Verify Telemetry
        self.mock_telemetry.log_tool_call.assert_called_with(
            "mock_tool", success=True, duration_ms=unittest.mock.ANY
        )
        print("✅ Tool Execution Logic Verified")

    async def test_tool_not_found(self):
        print("\n=== Testing Tool Not Found ===")
        result = await self.registry.execute_tool("missing_tool", {})
        
        print(f"Result: {result}")
        self.assertFalse(result['success'])
        self.assertIn("not found", result['error'])
        
        self.mock_telemetry.log_error.assert_called()
        print("✅ Error Handling Verified")

if __name__ == "__main__":
    from unittest import IsolatedAsyncioTestCase
    
    class AsyncTest(IsolatedAsyncioTestCase, TestToolRegistry):
        pass
        
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

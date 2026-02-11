
import sys
import os
import unittest
from unittest.mock import MagicMock

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.orchestrator.hands import Hands
from backend.orchestrator.schemas import ActionResult

class TestHandsCanvas(unittest.TestCase):
    def test_update_state_with_canvas(self):
        hands = Hands()
        
        # Initial state
        state = {
            "iteration_count": 1,
            "action_history": [],
            "decision": {"action_type": "agent", "resource_id": "mock_agent"}
        }
        
        # Mock result with StandardAgentResponse and canvas_display
        mock_output = {
            "status": "success",
            "standard_response": {
                "status": "success",
                "summary": "Created a canvas",
                "canvas_display": {
                    "canvas_type": "html",
                    "canvas_content": "<h1>Hello Canvas</h1>",
                    "heading": "Test Canvas"
                }
            }
        }
        
        result = ActionResult(
            action_id="test_action",
            success=True,
            output=mock_output,
            execution_time_ms=100
        )
        
        # Call the method under test
        updates = hands._update_state_with_result(state, result)
        
        # Assertions
        print("\n--- Test Results ---")
        print(f"Updates keys: {updates.keys()}")
        
        self.assertTrue(updates.get("has_canvas"), "has_canvas should be True")
        self.assertEqual(updates.get("canvas_type"), "html", "canvas_type should be 'html'")
        self.assertEqual(updates.get("canvas_content"), "<h1>Hello Canvas</h1>", "canvas_content mismatch")
        self.assertEqual(updates.get("canvas_title"), "Test Canvas", "canvas_title mismatch")
        self.assertEqual(updates.get("browser_view"), "<h1>Hello Canvas</h1>", "browser_view should be set for html type")
        self.assertEqual(updates.get("current_view"), "browser", "current_view should be 'browser'")
        
        print("✅ Canvas propagation test passed!")

if __name__ == "__main__":
    unittest.main()


import unittest
from unittest.mock import MagicMock, patch
import sys
import io
import os
import time

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from services.telemetry_service import TelemetryService

class TestTelemetryService(unittest.TestCase):
    def setUp(self):
        self.service = TelemetryService()

    @patch('psutil.Process')
    def test_metrics_accumulation(self, mock_process_cls):
        print("\n=== Testing Metric Accumulation ===")
        
        # Mock memory
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100 # 100 MB
        mock_process_cls.return_value = mock_process
        
        # 1. Log Requests
        self.service.log_request(success=True, latency_ms=100)
        self.service.log_request(success=False, latency_ms=50)
        
        metrics = self.service.get_metrics()
        reqs = metrics['requests']
        perf = metrics['performance']
        
        self.assertEqual(reqs['total'], 2)
        self.assertEqual(reqs['successful'], 1)
        self.assertEqual(reqs['failed'], 1)
        self.assertEqual(perf['requests_completed'], 2)
        self.assertEqual(perf['avg_latency_ms'], 75.0)
        print("✅ Request Metrics Verified")
        
        # 2. Log Tools
        self.service.log_tool_call("search", success=True)
        self.service.log_tool_call("search", success=True)
        self.service.log_tool_call("calculator", success=False)
        
        tools = self.service.get_metrics()['tools']
        self.assertEqual(tools['total_calls'], 3)
        self.assertEqual(tools['by_tool']['search'], 2)
        self.assertEqual(tools['by_tool']['calculator'], 1)
        print("✅ Tool Metrics Verified")
        
        # 3. Log Agents
        self.service.log_agent_call("writer", success=True)
        
        agents = self.service.get_metrics()['agents']
        self.assertEqual(agents['total_calls'], 1)
        print("✅ Agent Metrics Verified")
        
        # 4. Log Errors
        self.service.log_error("planning", "Plan failed: Timeout")
        
        errs = self.service.get_metrics()['errors']
        self.assertEqual(errs['total'], 1)
        self.assertEqual(errs['planning_errors'], 1)
        self.assertEqual(errs['by_type']['Plan failed'], 1)
        print("✅ Error Metrics Verified")

    def test_metrics_snapshot(self):
        print("\n=== Testing Snapshot Structure ===")
        snapshot = self.service.get_metrics()
        required_keys = ['uptime_seconds', 'success_rate', 'requests', 'agents', 'tools', 'errors']
        for key in required_keys:
            self.assertIn(key, snapshot)
        print("✅ Snapshot Structure Verified")

if __name__ == "__main__":
    unittest.main()

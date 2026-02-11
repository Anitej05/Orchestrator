
import unittest
from unittest.mock import MagicMock, patch, AsyncMock, ANY
import sys
import io
import os
from datetime import datetime

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.workflow_scheduler import WorkflowScheduler

class TestWorkflowScheduler(unittest.TestCase):
    def setUp(self):
        # Patch BackgroundScheduler to prevent actual scheduling
        self.scheduler_patcher = patch('services.workflow_scheduler.BackgroundScheduler')
        self.mock_scheduler_cls = self.scheduler_patcher.start()
        self.mock_scheduler_inst = self.mock_scheduler_cls.return_value
        
        self.service = WorkflowScheduler()
        
    def tearDown(self):
        self.scheduler_patcher.stop()

    def test_add_schedule_success(self):
        print("\n=== Testing Add Schedule (Success) ===")
        
        # Valid cron: "0 9 * * 1" (Every Monday at 9:00 AM)
        cron = "0 9 * * 1"
        result = self.service.add_schedule(
            "sched-1", "wf-1", cron, {}, "user-1", MagicMock()
        )
        
        self.assertTrue(result)
        self.mock_scheduler_inst.add_job.assert_called_once()
        
        # Verify trigger args
        call_args = self.mock_scheduler_inst.add_job.call_args
        _, kwargs = call_args
        trigger = kwargs.get('trigger')
        
        # We can't easily inspect CronTrigger object attributes directly if it's the real class,
        # but we know it didn't raise ValueError.
        print("✅ Schedule added successfully")

    def test_add_schedule_invalid_cron(self):
        print("\n=== Testing Add Schedule (Invalid Cron) ===")
        
        cron = "invalid cron"
        with self.assertRaises(ValueError):
            self.service.add_schedule(
                "sched-1", "wf-1", cron, {}, "user-1", MagicMock()
            )
        print("✅ Invalid cron rejected")

    @patch('models.WorkflowExecution')
    @patch('models.WorkflowSchedule')
    def test_execute_scheduled_workflow_logic(self, mock_wf_sched_cls, mock_wf_exec_cls):
        print("\n=== Testing Execution Logic ===")
        
        # We need to mock datetime.utcnow() but since it's a builtin, 
        # it's safer to just let it run or use freezegun if strict time check needed.
        # For this test, verifying flow control is enough.
        
        # Mock DB
        mock_db = MagicMock()
        mock_session_factory = MagicMock(return_value=mock_db)
        
        # Mock Workflow lookup
        mock_workflow = MagicMock()
        mock_workflow.blueprint = {"task_plan": []}
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_workflow, # 1st query: Workflow
            MagicMock()    # 2nd query: Schedule (to update last_run_at)
        ]
        
        # Mock async execution to avoid running real graph
        # We patch the instance method _async_execute_workflow
        self.service._async_execute_workflow = AsyncMock()
        
        # Run
        self.service._execute_scheduled_workflow(
            "sched-1", "wf-1", {}, "user-1", mock_session_factory
        )
        
        # Verify execution record creation
        mock_db.add.assert_called() # Should add WorkflowExecution
        mock_db.commit.assert_called()
        
        # Verify async execution was called
        # Note: _execute_scheduled_workflow creates a new loop and runs until complete.
        # Since we mocked _async_execute_workflow, that mock should have been awaited.
        self.service._async_execute_workflow.assert_called_once_with(
            ANY, "wf-1", mock_workflow.blueprint, {}, "user-1"
        )
        
        print("✅ Execution logic verified (DB record + Async trigger)")

if __name__ == "__main__":
    unittest.main()

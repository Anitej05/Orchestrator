"""
Integration Tests for Real-World Browser Agent Scenarios
These tests actually run the agent and measure its intelligence and adaptability
"""

import pytest
import requests
import time
import json
from typing import Dict, Any

# Base URL for browser agent
AGENT_URL = "http://localhost:8070/browse"

class TestRealWorldScenarios:
    """Test agent with real-world scenarios"""
    
    def _run_task(self, task: str, max_steps: int = 15, timeout: int = 120) -> Dict[str, Any]:
        """Helper to run a task and return results"""
        payload = {
            "task": task,
            "max_steps": max_steps,
            "extract_data": True
        }
        
        response = requests.post(AGENT_URL, json=payload, timeout=timeout)
        assert response.status_code == 200, f"Agent returned {response.status_code}"
        
        return response.json()
    
    def test_simple_navigation_and_search(self):
        """Test: Navigate to Google and perform a search"""
        result = self._run_task(
            "Go to google.com and search for 'Python programming'",
            max_steps=5
        )
        
        # Assertions
        assert result['success'], f"Task failed: {result.get('error')}"
        assert len(result['actions_taken']) > 0, "No actions were taken"
        
        # Check if navigation happened
        navigate_actions = [a for a in result['actions_taken'] if a.get('action') == 'navigate']
        assert len(navigate_actions) > 0, "Agent didn't navigate"
        
        # Check if typing happened
        type_actions = [a for a in result['actions_taken'] if a.get('action') == 'type']
        assert len(type_actions) > 0, "Agent didn't type search query"
        
        print(f"✅ Simple navigation test passed")
        print(f"   Actions: {len(result['actions_taken'])}")
        print(f"   Success rate: {result.get('action_success_rate', 'N/A')}")
    
    def test_stuck_detection_and_recovery(self):
        """Test: Agent should detect when stuck and adapt"""
        result = self._run_task(
            "Go to example.com and click on a non-existent button called 'DoesNotExist'",
            max_steps=10
        )
        
        # Agent should realize button doesn't exist and complete/skip
        assert result['success'] or result.get('task_summary', '').lower().find('not found') >= 0
        
        # Should not repeat same failed action many times
        failed_actions = result.get('actions_failed', [])
        if len(failed_actions) > 0:
            # Check for repeated failures
            action_types = [a.get('action') for a in failed_actions]
            max_repeats = max([action_types.count(a) for a in set(action_types)])
            assert max_repeats <= 3, f"Agent repeated same action {max_repeats} times (stuck!)"
        
        print(f"✅ Stuck detection test passed")
        print(f"   Failed actions: {len(failed_actions)}")
    
    def test_multi_step_workflow(self):
        """Test: Complex multi-step task"""
        result = self._run_task(
            "Go to wikipedia.org, search for 'Artificial Intelligence', and tell me the first paragraph",
            max_steps=15
        )
        
        assert result['success'], f"Multi-step task failed: {result.get('error')}"
        
        # Should have multiple action types
        actions = result['actions_taken']
        action_types = set([a.get('action') for a in actions])
        assert 'navigate' in action_types, "Missing navigation"
        assert 'type' in action_types or 'click' in action_types, "Missing interaction"
        
        # Should have extracted some data
        assert result.get('extracted_data') or result.get('task_summary'), "No data extracted"
        
        print(f"✅ Multi-step workflow test passed")
        print(f"   Action types used: {action_types}")
        print(f"   Total steps: {len(actions)}")
    
    def test_element_finding_accuracy(self):
        """Test: Can agent find and interact with common elements?"""
        result = self._run_task(
            "Go to github.com and click on the 'Sign in' button",
            max_steps=8
        )
        
        assert result['success'], "Failed to find Sign in button"
        
        # Check if click action was successful
        succeeded = result.get('actions_succeeded', [])
        click_succeeded = any(a.get('action') == 'click' for a in succeeded)
        assert click_succeeded, "Click action didn't succeed"
        
        print(f"✅ Element finding test passed")
    
    def test_form_interaction(self):
        """Test: Can agent interact with forms?"""
        result = self._run_task(
            "Go to google.com, type 'test query' in the search box, but don't submit",
            max_steps=8
        )
        
        assert result['success'], "Form interaction failed"
        
        # Should have typed something
        type_actions = [a for a in result.get('actions_succeeded', []) if a.get('action') == 'type']
        assert len(type_actions) > 0, "Didn't type in form"
        
        print(f"✅ Form interaction test passed")
    
    def test_adaptive_planning(self):
        """Test: Does agent adapt its plan based on page structure?"""
        result = self._run_task(
            "Go to example.com and extract all the text content",
            max_steps=10
        )
        
        assert result['success'], "Adaptive planning failed"
        
        # Should have created a plan
        assert len(result.get('actions_planned', [])) > 0, "No plan created"
        
        # Should have extracted data
        assert result.get('extracted_data') or result.get('task_summary'), "No extraction"
        
        print(f"✅ Adaptive planning test passed")
    
    def test_error_recovery(self):
        """Test: Can agent recover from errors?"""
        result = self._run_task(
            "Go to httpstat.us/500 and then go to example.com",
            max_steps=10
        )
        
        # Should handle 500 error and continue
        assert result['success'] or 'example.com' in result.get('task_summary', '').lower()
        
        print(f"✅ Error recovery test passed")
    
    def test_context_understanding(self):
        """Test: Does agent understand context and make smart decisions?"""
        result = self._run_task(
            "Go to google.com. If you see a search box, type 'hello'. If not, just complete the task.",
            max_steps=10
        )
        
        assert result['success'], "Context understanding failed"
        
        # Agent should have made a decision based on page content
        actions = result['actions_taken']
        assert len(actions) > 1, "Agent didn't make contextual decisions"
        
        print(f"✅ Context understanding test passed")
    
    def test_efficiency(self):
        """Test: Does agent complete tasks efficiently?"""
        start_time = time.time()
        
        result = self._run_task(
            "Go to example.com",
            max_steps=5
        )
        
        elapsed = time.time() - start_time
        
        assert result['success'], "Simple task failed"
        assert elapsed < 30, f"Task took too long: {elapsed}s"
        assert len(result['actions_taken']) <= 3, f"Too many actions: {len(result['actions_taken'])}"
        
        print(f"✅ Efficiency test passed")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Actions: {len(result['actions_taken'])}")
    
    def test_selector_reliability(self):
        """Test: Can agent find elements with various selector types?"""
        result = self._run_task(
            "Go to github.com and find the search input field",
            max_steps=8
        )
        
        assert result['success'], "Selector reliability test failed"
        
        # Check if agent found and interacted with search
        actions = result['actions_taken']
        assert any('search' in str(a).lower() for a in actions), "Didn't find search element"
        
        print(f"✅ Selector reliability test passed")


class TestAgentIntelligence:
    """Test the agent's decision-making intelligence"""
    
    def _run_task(self, task: str, max_steps: int = 15) -> Dict[str, Any]:
        """Helper to run a task"""
        payload = {"task": task, "max_steps": max_steps, "extract_data": True}
        response = requests.post(AGENT_URL, json=payload, timeout=120)
        return response.json()
    
    def test_knows_when_to_stop(self):
        """Test: Agent should know when task is complete"""
        result = self._run_task(
            "Go to example.com",
            max_steps=10
        )
        
        # Should complete quickly, not use all steps
        assert len(result['actions_taken']) < 5, "Agent didn't recognize task completion"
        assert result['success'], "Simple task failed"
        
        print(f"✅ Completion detection test passed")
        print(f"   Used {len(result['actions_taken'])}/{result.get('max_steps', 10)} steps")
    
    def test_prioritizes_essential_actions(self):
        """Test: Agent should focus on essential actions"""
        result = self._run_task(
            "Go to google.com and search for 'test'",
            max_steps=10
        )
        
        actions = result['actions_taken']
        action_types = [a.get('action') for a in actions]
        
        # Should have navigate and type (essential), not many scrolls or waits
        assert 'navigate' in action_types, "Missing essential navigation"
        assert 'type' in action_types, "Missing essential typing"
        
        # Shouldn't have too many non-essential actions
        scroll_count = action_types.count('scroll')
        assert scroll_count <= 2, f"Too many scrolls: {scroll_count}"
        
        print(f"✅ Action prioritization test passed")
    
    def test_adapts_to_page_structure(self):
        """Test: Agent should adapt strategy based on page"""
        result = self._run_task(
            "Go to example.com and extract the main heading",
            max_steps=10
        )
        
        assert result['success'], "Adaptation test failed"
        
        # Should have extracted something
        summary = result.get('task_summary', '')
        assert len(summary) > 10, "Didn't extract meaningful content"
        
        print(f"✅ Page adaptation test passed")
    
    def test_handles_ambiguity(self):
        """Test: Agent should handle ambiguous instructions"""
        result = self._run_task(
            "Go to a popular search engine and search for something interesting",
            max_steps=15
        )
        
        # Agent should make reasonable choices
        assert result['success'] or len(result['actions_taken']) > 2
        
        # Should have navigated somewhere
        actions = result['actions_taken']
        assert any(a.get('action') == 'navigate' for a in actions), "Didn't navigate"
        
        print(f"✅ Ambiguity handling test passed")


class TestPerformanceMetrics:
    """Test and measure agent performance"""
    
    def _run_task(self, task: str, max_steps: int = 15) -> Dict[str, Any]:
        """Helper to run a task"""
        payload = {"task": task, "max_steps": max_steps}
        response = requests.post(AGENT_URL, json=payload, timeout=120)
        return response.json()
    
    def test_measure_success_rate(self):
        """Measure overall success rate across multiple tasks"""
        tasks = [
            "Go to example.com",
            "Go to google.com and search for 'test'",
            "Go to github.com",
            "Go to wikipedia.org",
            "Go to stackoverflow.com"
        ]
        
        results = []
        for task in tasks:
            try:
                result = self._run_task(task, max_steps=8)
                results.append({
                    'task': task,
                    'success': result.get('success', False),
                    'actions': len(result.get('actions_taken', [])),
                    'time': result.get('metrics', {}).get('total_time', 0)
                })
            except Exception as e:
                results.append({
                    'task': task,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r.get('success'))
        success_rate = (success_count / len(tasks)) * 100
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Tasks Completed: {success_count}/{len(tasks)}")
        
        for r in results:
            status = "✅" if r.get('success') else "❌"
            print(f"   {status} {r['task'][:50]}")
        
        # Should have at least 60% success rate
        assert success_rate >= 60, f"Success rate too low: {success_rate}%"
    
    def test_measure_efficiency(self):
        """Measure action efficiency"""
        result = self._run_task("Go to example.com", max_steps=10)
        
        actions = len(result.get('actions_taken', []))
        success_rate = result.get('action_success_rate', '0/0')
        
        print(f"\n⚡ Efficiency Metrics:")
        print(f"   Actions taken: {actions}")
        print(f"   Success rate: {success_rate}")
        
        # Simple task should be efficient
        assert actions <= 3, f"Too many actions for simple task: {actions}"


if __name__ == '__main__':
    print("=" * 80)
    print("REAL-WORLD INTEGRATION TESTS")
    print("=" * 80)
    print("\nThese tests measure the agent's actual intelligence and adaptability")
    print("Not just whether functions work, but whether decisions are smart\n")
    
    pytest.main([__file__, '-v', '-s'])

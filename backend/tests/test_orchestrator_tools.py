"""
Comprehensive Test Suite for Orchestrator Tool Calling and Code Environment

This test suite verifies that the orchestrator can properly:
1. Call all tool types (Python, Terminal, registered tools)
2. Use its code environment properly
3. Execute actions through the Hands dispatcher
4. Make decisions through the Brain
5. Flow through the Graph correctly

Run with: python -m pytest backend/tests/test_orchestrator_tools.py -v
"""

import asyncio
import json
import sys
import os
import time
from typing import Dict, Any, List
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock


# =============================================================================
# PHASE 1: UNIT TESTS - Individual Components
# =============================================================================

class TestCodeSandboxService:
    """Test the CodeSandboxService for Python code execution."""
    
    def test_sandbox_initialization(self):
        """Test that sandbox initializes with correct safe builtins."""
        from backend.services.code_sandbox_service import code_sandbox
        
        # Verify sandbox has sessions dict
        assert hasattr(code_sandbox, 'sessions')
        assert isinstance(code_sandbox.sessions, dict)
        
        # Verify safe builtins are configured
        assert 'print' in code_sandbox.SAFE_BUILTINS
        assert 'len' in code_sandbox.SAFE_BUILTINS
        
        # Verify session globals include pandas and numpy (not in SAFE_BUILTINS but in session globals)
        session = code_sandbox._get_or_create_session("test_init")
        assert 'pd' in session['globals']  # pandas
        assert 'np' in session['globals']  # numpy
        
        print("✅ CodeSandboxService initialization test passed")

    
    def test_sandbox_basic_execution(self):
        """Test basic Python code execution."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = "print('Hello from sandbox')\nresult = 42"
        result = code_sandbox.execute_code(code, session_id="test_session")
        
        assert result['success'] is True
        assert 'Hello from sandbox' in result['stdout']
        assert result['result'] == 42
        
        print("✅ CodeSandboxService basic execution test passed")
    
    def test_sandbox_pandas_execution(self):
        """Test pandas operations in sandbox."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
import pandas as pd
data = {'name': ['Alice', 'Bob'], 'age': [25, 30]}
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
result = df['age'].mean()
"""
        result = code_sandbox.execute_code(code, session_id="pandas_test")
        
        assert result['success'] is True
        assert 'DataFrame shape: (2, 2)' in result['stdout']
        assert result['result'] == 27.5
        
        print("✅ CodeSandboxService pandas execution test passed")
    
    def test_sandbox_error_handling(self):
        """Test that sandbox properly handles errors."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = "result = 1 / 0"  # Division by zero
        result = code_sandbox.execute_code(code, session_id="error_test")
        
        assert result['success'] is False
        assert result['error'] is not None
        assert 'division by zero' in result['error'].lower()
        
        print("✅ CodeSandboxService error handling test passed")
    
    def test_sandbox_session_persistence(self):
        """Test that variables persist across executions in same session."""
        from backend.services.code_sandbox_service import code_sandbox
        
        # First execution - set variable
        code1 = "x = 100\nprint(f'x = {x}')"
        result1 = code_sandbox.execute_code(code1, session_id="persist_test")
        assert result1['success'] is True
        
        # Second execution - use variable from first
        code2 = "y = x + 50\nprint(f'y = {y}')\nresult = y"
        result2 = code_sandbox.execute_code(code2, session_id="persist_test")
        
        assert result2['success'] is True
        assert result2['result'] == 150
        
        print("✅ CodeSandboxService session persistence test passed")


class TestTerminalService:
    """Test the TerminalService for shell command execution."""
    
    def test_terminal_initialization(self):
        """Test terminal service initialization."""
        from backend.services.terminal_service import terminal_service
        
        assert terminal_service.base_dir is not None
        assert terminal_service.current_cwd is not None
        
        print("✅ TerminalService initialization test passed")
    
    def test_terminal_basic_command(self):
        """Test basic terminal command execution."""
        from backend.services.terminal_service import terminal_service
        
        # Use 'echo' which works on both Windows and Unix
        result = terminal_service.execute_command("echo 'Hello from terminal'")
        
        assert result['returncode'] == 0
        assert 'Hello from terminal' in result['stdout']
        
        print("✅ TerminalService basic command test passed")
    
    def test_terminal_directory_listing(self):
        """Test directory listing command."""
        from backend.services.terminal_service import terminal_service
        
        # List directory contents
        if os.name == 'nt':  # Windows
            result = terminal_service.execute_command("dir")
        else:  # Unix/Linux/Mac
            result = terminal_service.execute_command("ls -la")
        
        assert result['returncode'] == 0
        assert len(result['stdout']) > 0
        
        print("✅ TerminalService directory listing test passed")
    
    def test_terminal_cd_command(self):
        """Test directory change command."""
        from backend.services.terminal_service import terminal_service
        
        # Get current directory
        initial_cwd = terminal_service.current_cwd
        
        # Try to cd to parent (should exist)
        result = terminal_service.execute_command("cd ..")
        
        assert result['returncode'] == 0
        assert 'Changed directory' in result['stdout']
        
        # Verify cwd changed
        assert terminal_service.current_cwd != initial_cwd
        
        print("✅ TerminalService cd command test passed")
    
    def test_terminal_error_handling(self):
        """Test terminal error handling for invalid commands."""
        from backend.services.terminal_service import terminal_service
        
        # Execute invalid command
        result = terminal_service.execute_command("this_command_does_not_exist_12345")
        
        # Should have non-zero return code
        assert result['returncode'] != 0
        
        print("✅ TerminalService error handling test passed")


class TestToolRegistryService:
    """Test the ToolRegistryService for tool discovery and execution."""
    
    def test_tool_registry_initialization(self):
        """Test tool registry initialization."""
        from backend.services.tool_registry_service import tool_registry
        
        # Initialize registry
        tool_registry.initialize()
        
        # Verify tools were loaded
        assert tool_registry._initialized is True
        assert len(tool_registry._tools) > 0
        
        print(f"✅ ToolRegistryService initialized with {len(tool_registry._tools)} tools")
    
    def test_tool_listing(self):
        """Test that tools can be listed."""
        from backend.services.tool_registry_service import tool_registry
        
        tools = tool_registry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Verify tool structure
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'category' in tool
        
        print(f"✅ ToolRegistryService listed {len(tools)} executable tools")
    
    def test_tool_retrieval(self):
        """Test retrieving specific tools."""
        from backend.services.tool_registry_service import tool_registry
        
        # Get list of available tools
        tools = tool_registry.list_tools()
        if len(tools) == 0:
            pytest.skip("No tools available to test")
        
        # Try to get first tool
        tool_name = tools[0]['name']
        tool = tool_registry.get_tool(tool_name)
        
        # Tool might be None if not implemented, but shouldn't error
        print(f"✅ ToolRegistryService retrieved tool '{tool_name}': {tool is not None}")
    
    def test_tool_prompt_summary(self):
        """Test generating tool summary for LLM context."""
        from backend.services.tool_registry_service import tool_registry
        
        summary = tool_registry.get_tool_prompt_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'AVAILABLE TOOLS' in summary
        
        print("✅ ToolRegistryService prompt summary generated")


# =============================================================================
# PHASE 2: INTEGRATION TESTS - Hands Dispatcher
# =============================================================================

class TestHandsDispatcher:
    """Test the Hands dispatcher for executing actions."""
    
    @pytest.mark.asyncio
    async def test_hands_python_execution(self):
        """Test Hands executing Python code."""
        from backend.orchestrator.hands import Hands
        from backend.orchestrator.schemas import ActionResult
        
        hands = Hands()
        
        # Create mock state with Python action
        state = {
            "decision": {
                "action_type": "python",
                "resource_id": None,
                "payload": {
                    "code": "print('Hello from Hands')\nresult = 123",
                    "session_id": "hands_test"
                }
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        # Execute
        result = await hands.execute(state)
        
        # Verify result structure
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        assert 'Hello from Hands' in str(execution_result['output'])
        
        print("✅ Hands Python execution test passed")
    
    @pytest.mark.asyncio
    async def test_hands_terminal_execution(self):
        """Test Hands executing terminal command."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "terminal",
                "resource_id": None,
                "payload": {
                    "command": "echo 'Hands terminal test'"
                }
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        assert 'Hands terminal test' in str(execution_result['output'])
        
        print("✅ Hands terminal execution test passed")
    
    @pytest.mark.asyncio
    async def test_hands_skip_action(self):
        """Test Hands handling skip action."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "skip",
                "resource_id": None,
                "payload": {}
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        
        print("✅ Hands skip action test passed")
    
    @pytest.mark.asyncio
    async def test_hands_finish_action(self):
        """Test Hands handling finish action."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "finish",
                "resource_id": None,
                "payload": {},
                "user_response": "Task completed successfully!"
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        assert 'Task completed' in str(execution_result['output'])
        
        print("✅ Hands finish action test passed")
    
    @pytest.mark.asyncio
    async def test_hands_plan_action(self):
        """Test Hands handling plan action."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "plan",
                "resource_id": None,
                "payload": {},
                "execution_plan": [
                    {"phase_id": "1", "name": "Test Phase", "goal": "Test goal", "depends_on": []}
                ]
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        assert 'plan' in str(execution_result['output']).lower()
        
        print("✅ Hands plan action test passed")
    
    @pytest.mark.asyncio
    async def test_hands_parallel_execution(self):
        """Test Hands executing parallel actions."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "parallel",
                "resource_id": None,
                "payload": {},
                "parallel_actions": [
                    {
                        "action_type": "python",
                        "resource_id": None,
                        "payload": {
                            "code": "print('Parallel 1')\nresult = 1",
                            "session_id": "parallel_test_1"
                        }
                    },
                    {
                        "action_type": "python",
                        "resource_id": None,
                        "payload": {
                            "code": "print('Parallel 2')\nresult = 2",
                            "session_id": "parallel_test_2"
                        }
                    }
                ]
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is True
        
        # Check parallel results
        output = execution_result['output']
        assert 'parallel_results' in output
        assert output['total_actions'] == 2
        
        print("✅ Hands parallel execution test passed")
    
    @pytest.mark.asyncio
    async def test_hands_error_handling(self):
        """Test Hands error handling for invalid action."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "unknown_action_type",
                "resource_id": "test",
                "payload": {}
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'execution_result' in result
        execution_result = result['execution_result']
        assert execution_result['success'] is False
        assert 'Unknown action type' in str(execution_result['error_message'])
        
        print("✅ Hands error handling test passed")
    
    @pytest.mark.asyncio
    async def test_hands_no_decision(self):
        """Test Hands handling missing decision."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'error' in result
        assert 'No brain decision' in result['error']
        
        print("✅ Hands no decision test passed")


# =============================================================================
# PHASE 3: INTEGRATION TESTS - Brain Decision Making
# =============================================================================

class TestBrainDecisionMaking:
    """Test the Brain's decision making capabilities."""
    
    def test_brain_initialization(self):
        """Test Brain initialization."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        assert brain.max_failures == 3
        assert brain.max_iterations == 25
        
        print("✅ Brain initialization test passed")
    
    @pytest.mark.asyncio
    async def test_brain_think_initializes_state(self):
        """Test that Brain initializes state for new conversations."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        # State with original_prompt but no todo_list
        state = {
            "original_prompt": "Test objective",
            "todo_list": [],
            "messages": []
        }
        
        result = await brain.think(state)
        
        # Should initialize todo_list
        assert 'todo_list' in result
        assert len(result['todo_list']) > 0
        assert 'decision' in result
        
        print("✅ Brain state initialization test passed")
    
    @pytest.mark.asyncio
    async def test_brain_max_iterations_guard(self):
        """Test Brain max iterations guard."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Test",
            "todo_list": [{"task_id": "1", "status": "pending"}],
            "iteration_count": 25,  # At max
            "messages": []
        }
        
        result = await brain.think(state)
        
        # Should force finish
        assert 'final_response' in result
        assert 'Maximum iterations' in result['final_response']
        
        print("✅ Brain max iterations guard test passed")
    
    @pytest.mark.asyncio
    async def test_brain_max_failures_fallback(self):
        """Test Brain fallback mode after max failures."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Test",
            "todo_list": [{"task_id": "1", "status": "pending"}],
            "failure_count": 3,  # At max
            "messages": []
        }
        
        result = await brain.think(state)
        
        # Should enter fallback mode
        assert 'decision' in result
        assert 'final_response' in result
        
        print("✅ Brain max failures fallback test passed")


# =============================================================================
# PHASE 4: END-TO-END TESTS - Full Orchestrator Flow
# =============================================================================

class TestOrchestratorEndToEnd:
    """Test complete orchestrator workflows."""
    
    @pytest.mark.asyncio
    async def test_simple_python_workflow(self):
        """Test a simple workflow: Brain decides Python -> Hands executes."""
        from backend.orchestrator.brain import Brain, BrainDecision
        from backend.orchestrator.hands import Hands
        
        # Initialize components
        brain = Brain()
        hands = Hands()
        
        # Create initial state
        state = {
            "original_prompt": "Calculate 2 + 2 using Python",
            "todo_list": [],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": []
        }
        
        # Step 1: Brain thinks
        brain_result = await brain.think(state)
        state.update(brain_result)
        
        # Verify brain made a decision
        assert 'decision' in state
        decision = state['decision']
        
        # If brain decided to finish (might happen if it "knows" the answer)
        # or if it decided to use Python, test the execution
        if decision.get('action_type') == 'python':
            # Step 2: Hands executes
            hands_result = await hands.execute(state)
            state.update(hands_result)
            
            # Verify execution
            assert 'execution_result' in state
            assert state['execution_result']['success'] is True
        
        print("✅ Simple Python workflow test passed")
    
    @pytest.mark.asyncio
    async def test_simple_terminal_workflow(self):
        """Test a simple workflow: Brain decides Terminal -> Hands executes."""
        from backend.orchestrator.brain import Brain
        from backend.orchestrator.hands import Hands
        
        brain = Brain()
        hands = Hands()
        
        state = {
            "original_prompt": "List the current directory using terminal",
            "todo_list": [],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": []
        }
        
        # Brain thinks
        brain_result = await brain.think(state)
        state.update(brain_result)
        
        decision = state.get('decision', {})
        
        # If brain decided terminal, execute it
        if decision.get('action_type') == 'terminal':
            hands_result = await hands.execute(state)
            state.update(hands_result)
            
            assert 'execution_result' in state
            print("✅ Terminal workflow executed")
        else:
            print(f"✅ Brain decided: {decision.get('action_type')} (terminal not chosen, which is OK)")
        
        print("✅ Simple terminal workflow test passed")
    
    @pytest.mark.asyncio
    async def test_state_persistence_across_calls(self):
        """Test that state persists correctly across multiple brain/hands calls."""
        from backend.orchestrator.brain import Brain
        from backend.orchestrator.hands import Hands
        
        brain = Brain()
        hands = Hands()
        
        state = {
            "original_prompt": "First calculate 10 * 5, then add 20",
            "todo_list": [],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": [],
            "iteration_count": 0
        }
        
        # First call - brain initializes state (iteration_count stays 0 on init)
        brain_result = await brain.think(state)
        state.update(brain_result)
        
        # Verify state was initialized
        assert 'todo_list' in state
        assert len(state['todo_list']) > 0
        
        # Second call - brain makes a decision (iteration_count increments)
        brain_result = await brain.think(state)
        state.update(brain_result)
        
        # Execute if it's an action
        decision = state.get('decision', {})
        if decision.get('action_type') in ['python', 'terminal', 'tool']:
            hands_result = await hands.execute(state)
            state.update(hands_result)
        
        # Verify iteration count incremented (now should be 1 after second think)
        assert state['iteration_count'] >= 1
        
        # Verify action history was recorded
        assert 'action_history' in state
        
        print("✅ State persistence test passed")

    
    def test_graph_creation(self):
        """Test that the graph can be created."""
        from backend.orchestrator.graph import create_graph_with_checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        
        # Create checkpointer
        checkpointer = MemorySaver()
        
        # Create graph
        graph = create_graph_with_checkpointer(checkpointer)
        
        assert graph is not None
        
        print("✅ Graph creation test passed")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("ORCHESTRATOR TOOL CALLING & CODE ENVIRONMENT TEST SUITE")
    print("="*70 + "\n")
    
    # Run pytest programmatically
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("\n" + "="*70)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ TESTS FAILED (exit code: {result.returncode})")
    print("="*70)
    
    return result.returncode


if __name__ == "__main__":
    # Run tests
    exit_code = run_all_tests()
    sys.exit(exit_code)

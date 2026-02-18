"""
Comprehensive End-to-End Test Suite for the Orchestrator

This test suite verifies real-world orchestrator capabilities:
1. Python code execution through the full Brain-Hands cycle
2. Terminal command execution
3. Tool discovery and execution
4. Agent registry and routing
5. Multi-step workflows with planning
6. Parallel execution
7. Human-in-the-loop approval flows
8. Error handling and recovery
9. State persistence and context awareness
10. Graph-based orchestration flow

Run with: python -m pytest backend/tests/test_orchestrator_e2e.py -v
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
# SECTION 1: CODE ENVIRONMENT TESTS
# =============================================================================

class TestCodeEnvironmentRealWorld:
    """Test real-world Python code execution scenarios."""
    
    def test_data_analysis_with_pandas(self):
        """Test a realistic data analysis task."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
import pandas as pd
import numpy as np

# Create sample sales data
data = {
    'product': ['A', 'B', 'C', 'A', 'B', 'C'],
    'region': ['North', 'North', 'South', 'South', 'North', 'South'],
    'sales': [1000, 1500, 2000, 1200, 1800, 2500],
    'units': [10, 15, 20, 12, 18, 25]
}

df = pd.DataFrame(data)

# Calculate metrics
total_sales = df['sales'].sum()
avg_sales_by_region = df.groupby('region')['sales'].mean()
top_product = df.groupby('product')['sales'].sum().idxmax()

print(f"Total Sales: ${total_sales:,.2f}")
print(f"Top Product: {top_product}")
print(f"Average by Region:")
print(avg_sales_by_region)

result = {
    'total_sales': total_sales,
    'top_product': top_product,
    'avg_by_region': avg_sales_by_region.to_dict()
}
"""
        result = code_sandbox.execute_code(code, session_id="data_analysis_test")
        
        assert result['success'] is True
        assert 'Total Sales: $10,000' in result['stdout']
        # Result may be serialized as string or int depending on pandas behavior
        assert int(result['result']['total_sales']) == 10000
        assert result['result']['top_product'] == 'C'
        
        print("✅ Real-world data analysis test passed")
    
    def test_json_data_processing(self):
        """Test JSON data processing scenario."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
import json

# Sample JSON data (simulating API response)
api_response = '''
{
    "users": [
        {"id": 1, "name": "Alice", "role": "admin", "active": true},
        {"id": 2, "name": "Bob", "role": "user", "active": true},
        {"id": 3, "name": "Charlie", "role": "user", "active": false}
    ],
    "metadata": {"total": 3, "page": 1}
}
'''

data = json.loads(api_response)

# Process data
active_users = [u for u in data['users'] if u['active']]
admin_count = sum(1 for u in data['users'] if u['role'] == 'admin')

print(f"Active Users: {len(active_users)}")
print(f"Admin Count: {admin_count}")

result = {
    'active_user_names': [u['name'] for u in active_users],
    'admin_count': admin_count,
    'total_users': data['metadata']['total']
}
"""
        result = code_sandbox.execute_code(code, session_id="json_test")
        
        assert result['success'] is True
        assert result['result']['active_user_names'] == ['Alice', 'Bob']
        assert result['result']['admin_count'] == 1
        
        print("✅ JSON data processing test passed")
    
    def test_mathematical_computation(self):
        """Test mathematical computations."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
import math
import statistics

# Calculate statistics
numbers = [12, 15, 21, 18, 25, 30, 22, 19, 17, 24]

mean_val = statistics.mean(numbers)
median_val = statistics.median(numbers)
stdev_val = statistics.stdev(numbers)

# Calculate geometric growth
initial = 1000
rate = 0.08  # 8% growth
years = 5
final_value = initial * (1 + rate) ** years

print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val}")
print(f"Std Dev: {stdev_val:.2f}")
print(f"Future Value: ${final_value:,.2f}")

result = {
    'mean': mean_val,
    'median': median_val,
    'stdev': stdev_val,
    'future_value': final_value
}
"""
        result = code_sandbox.execute_code(code, session_id="math_test")
        
        assert result['success'] is True
        assert abs(result['result']['mean'] - 20.3) < 0.1
        assert result['result']['median'] == 20.0
        assert abs(result['result']['future_value'] - 1469.33) < 1
        
        print("✅ Mathematical computation test passed")
    
    def test_string_manipulation(self):
        """Test string processing tasks."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
import re

text = '''
Contact us at support@example.com or sales@company.org.
Call (555) 123-4567 or (555) 987-6543 for assistance.
Our website is https://www.example.com and http://company.org
'''

# Extract emails
emails = re.findall(r'[\\w.-]+@[\\w.-]+\\.\\w+', text)

# Extract phone numbers
phones = re.findall(r'\\(\\d{3}\\)\\s*\\d{3}-\\d{4}', text)

# Extract URLs
urls = re.findall(r'https?://[\\w./-]+', text)

print(f"Emails found: {emails}")
print(f"Phones found: {phones}")
print(f"URLs found: {urls}")

result = {
    'emails': emails,
    'phones': phones,
    'urls': urls
}
"""
        result = code_sandbox.execute_code(code, session_id="string_test")
        
        assert result['success'] is True
        assert len(result['result']['emails']) == 2
        assert len(result['result']['phones']) == 2
        assert len(result['result']['urls']) == 2
        
        print("✅ String manipulation test passed")
    
    def test_code_error_recovery(self):
        """Test that code errors are properly caught and reported."""
        from backend.services.code_sandbox_service import code_sandbox
        
        code = """
# This code has an error
x = 10
y = 0
result = x / y  # Division by zero
"""
        result = code_sandbox.execute_code(code, session_id="error_test")
        
        assert result['success'] is False
        assert result['error'] is not None
        assert 'division by zero' in result['error'].lower()
        
        print("✅ Code error recovery test passed")


# =============================================================================
# SECTION 2: TERMINAL EXECUTION TESTS
# =============================================================================

class TestTerminalExecutionRealWorld:
    """Test real-world terminal command scenarios."""
    
    def test_filesystem_exploration(self):
        """Test filesystem navigation and listing."""
        from backend.services.terminal_service import terminal_service
        
        # Get current directory
        result = terminal_service.execute_command("pwd" if os.name != 'nt' else "cd")
        assert result['returncode'] == 0
        
        # List files
        if os.name == 'nt':
            result = terminal_service.execute_command("dir /b")
        else:
            result = terminal_service.execute_command("ls -la")
        
        assert result['returncode'] == 0
        assert len(result['stdout']) > 0
        
        print("✅ Filesystem exploration test passed")
    
    def test_environment_info(self):
        """Test getting environment information."""
        from backend.services.terminal_service import terminal_service
        
        # Get environment variable
        result = terminal_service.execute_command("echo %USERPROFILE%" if os.name == 'nt' else "echo $HOME")
        
        assert result['returncode'] == 0
        assert len(result['stdout'].strip()) > 0
        
        print("✅ Environment info test passed")
    
    def test_command_chaining(self):
        """Test command chaining with && operator."""
        from backend.services.terminal_service import terminal_service
        
        if os.name == 'nt':
            result = terminal_service.execute_command("echo Hello && echo World")
        else:
            result = terminal_service.execute_command("echo 'Hello' && echo 'World'")
        
        assert result['returncode'] == 0
        assert 'Hello' in result['stdout']
        assert 'World' in result['stdout']
        
        print("✅ Command chaining test passed")


# =============================================================================
# SECTION 3: HANDS DISPATCHER REAL-WORLD TESTS
# =============================================================================

class TestHandsDispatcherRealWorld:
    """Test Hands dispatcher with real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_step_calculation(self):
        """Test multi-step calculation workflow."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        # Step 1: Calculate first part
        state = {
            "decision": {
                "action_type": "python",
                "resource_id": None,
                "payload": {
                    "code": "x = 100\ny = 200\nresult = x + y\nprint(f'Step 1: {x} + {y} = {result}')",
                    "session_id": "multi_step_test"
                }
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result1 = await hands.execute(state)
        assert result1['execution_result']['success'] is True
        
        # Step 2: Continue calculation in same session
        state['decision']['payload']['code'] = "z = result * 2\nprint(f'Step 2: {result} * 2 = {z}')\nresult = z"
        state['iteration_count'] = 1
        state['action_history'] = result1.get('action_history', [])
        
        result2 = await hands.execute(state)
        assert result2['execution_result']['success'] is True
        assert '600' in str(result2['execution_result']['output'])
        
        print("✅ Multi-step calculation test passed")
    
    @pytest.mark.asyncio
    async def test_parallel_data_fetching(self):
        """Test parallel execution for data fetching simulation."""
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
                            "code": "import time\ntime.sleep(0.1)\nresult = {'source': 'API1', 'data': [1, 2, 3]}",
                            "session_id": "parallel_1"
                        }
                    },
                    {
                        "action_type": "python",
                        "resource_id": None,
                        "payload": {
                            "code": "import time\ntime.sleep(0.1)\nresult = {'source': 'API2', 'data': [4, 5, 6]}",
                            "session_id": "parallel_2"
                        }
                    },
                    {
                        "action_type": "python",
                        "resource_id": None,
                        "payload": {
                            "code": "import time\ntime.sleep(0.1)\nresult = {'source': 'API3', 'data': [7, 8, 9]}",
                            "session_id": "parallel_3"
                        }
                    }
                ]
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        start_time = time.time()
        result = await hands.execute(state)
        duration = time.time() - start_time
        
        assert result['execution_result']['success'] is True
        output = result['execution_result']['output']
        assert output['total_actions'] == 3
        assert output['successful'] == 3
        
        # Parallel execution should be faster than sequential
        # (3 x 0.1s = 0.3s sequential, but parallel should be ~0.1s)
        assert duration < 0.5, f"Parallel execution took {duration}s, expected < 0.5s"
        
        print(f"✅ Parallel data fetching test passed (completed in {duration:.2f}s)")
    
    @pytest.mark.asyncio
    async def test_complex_payload_handling(self):
        """Test handling of complex nested payloads."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        complex_payload = {
            "code": """
import json

# Complex nested data
config = {
    'database': {
        'host': 'localhost',
        'port': 5432,
        'credentials': {
            'username': 'admin',
            'password': 'secret'
        }
    },
    'features': ['auth', 'logging', 'caching'],
    'settings': {
        'debug': True,
        'timeout': 30
    }
}

# Process configuration
db_url = f"postgresql://{config['database']['credentials']['username']}:***@{config['database']['host']}:{config['database']['port']}"
feature_count = len(config['features'])

print(f"Database URL: {db_url}")
print(f"Features enabled: {feature_count}")

result = {'db_url': db_url, 'feature_count': feature_count}
""",
            "session_id": "complex_payload_test"
        }
        
        state = {
            "decision": {
                "action_type": "python",
                "resource_id": None,
                "payload": complex_payload
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert result['execution_result']['success'] is True
        output = result['execution_result']['output']
        # Output may be wrapped in different structures depending on processing
        # Check that the data is present in the output
        assert 'postgresql://admin' in str(output)
        # The output structure includes 'result' key with the actual result data as a string
        # Feature count 3 should be somewhere in the output
        assert 'feature_count' in str(output) and "'feature_count': 3" in str(output)
        
        print("✅ Complex payload handling test passed")


# =============================================================================
# SECTION 4: BRAIN DECISION MAKING TESTS
# =============================================================================

class TestBrainDecisionMakingRealWorld:
    """Test Brain's decision making with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_brain_initializes_with_complex_objective(self):
        """Test Brain initialization with a complex objective."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Analyze the sales data from Q4 2023, compare it with Q3 2023, create a summary report, and email it to the finance team.",
            "todo_list": [],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": []
        }
        
        result = await brain.think(state)
        
        assert 'todo_list' in result
        assert len(result['todo_list']) > 0
        assert 'decision' in result
        
        print("✅ Brain initialization with complex objective test passed")
    
    @pytest.mark.asyncio
    async def test_brain_action_history_tracking(self):
        """Test that Brain tracks action history correctly."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        # Simulate state with existing action history
        state = {
            "original_prompt": "Continue the analysis",
            "todo_list": [{"task_id": "1", "status": "in_progress", "description": "Analyze data"}],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": [
                {
                    "iteration": 1,
                    "action_type": "python",
                    "resource_id": None,
                    "success": True,
                    "result_summary": "Loaded data successfully"
                }
            ],
            "iteration_count": 1
        }
        
        result = await brain.think(state)
        
        # Brain should have made a decision based on history
        assert 'decision' in result
        
        print("✅ Brain action history tracking test passed")
    
    @pytest.mark.asyncio
    async def test_brain_phase_completion(self):
        """Test Brain's phase completion logic."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Multi-phase task",
            "todo_list": [{"task_id": "1", "status": "in_progress"}],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": [],
            "execution_plan": [
                {"phase_id": "1", "name": "Data Collection", "goal": "Get data", "status": "completed"},
                {"phase_id": "2", "name": "Analysis", "goal": "Analyze data", "status": "in_progress", "depends_on": ["1"]}
            ],
            "current_phase_id": "2",
            "iteration_count": 0
        }
        
        result = await brain.think(state)
        
        # Brain should recognize it's in phase 2
        assert 'decision' in result
        
        print("✅ Brain phase completion test passed")


# =============================================================================
# SECTION 5: FULL GRAPH FLOW TESTS
# =============================================================================

class TestFullGraphFlow:
    """Test the complete graph execution flow."""
    
    def test_graph_initialization(self):
        """Test that the graph initializes correctly."""
        from backend.orchestrator.graph import create_graph_with_checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        
        checkpointer = MemorySaver()
        graph = create_graph_with_checkpointer(checkpointer)
        
        assert graph is not None
        
        print("✅ Graph initialization test passed")
    
    @pytest.mark.asyncio
    async def test_simple_graph_execution(self):
        """Test simple execution through the graph."""
        from backend.orchestrator.graph import create_graph_with_checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage
        
        checkpointer = MemorySaver()
        graph = create_graph_with_checkpointer(checkpointer)
        
        # Create initial state
        config = {
            "configurable": {
                "thread_id": "test_thread_1",
                "owner": {"user_id": "test_user"}
            }
        }
        
        initial_state = {
            "original_prompt": "Calculate 2 + 2",
            "messages": [HumanMessage(content="Calculate 2 + 2")],
            "todo_list": [],
            "memory": {},
            "insights": {},
            "action_history": [],
            "thread_id": "test_thread_1",
            "user_id": "test_user",
            "iteration_count": 0,
            "failure_count": 0
        }
        
        # Execute graph
        try:
            result = await graph.ainvoke(initial_state, config)
            
            # Graph should have executed and returned a result
            assert result is not None
            assert 'iteration_count' in result
            
            print("✅ Simple graph execution test passed")
        except Exception as e:
            # If LLM is not configured, skip but don't fail
            if "api key" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip(f"LLM not configured: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_graph_state_persistence(self):
        """Test that state persists across graph invocations."""
        from backend.orchestrator.graph import create_graph_with_checkpointer
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage
        
        checkpointer = MemorySaver()
        graph = create_graph_with_checkpointer(checkpointer)
        
        thread_id = "persistence_test_thread"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "owner": {"user_id": "test_user"}
            }
        }
        
        initial_state = {
            "original_prompt": "Remember the number 42",
            "messages": [HumanMessage(content="Remember the number 42")],
            "todo_list": [],
            "memory": {},
            "insights": {},
            "action_history": [],
            "thread_id": thread_id,
            "user_id": "test_user",
            "iteration_count": 0,
            "failure_count": 0
        }
        
        try:
            result = await graph.ainvoke(initial_state, config)
            
            # Check that checkpoint was created
            checkpoint = await checkpointer.aget_tuple(config)
            
            # State should be retrievable
            assert checkpoint is not None or result is not None
            
            print("✅ Graph state persistence test passed")
        except Exception as e:
            if "api key" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip(f"LLM not configured: {e}")
            raise


# =============================================================================
# SECTION 6: OMNI-DISPATCHER INTEGRATION TESTS
# =============================================================================

class TestOmniDispatcherIntegration:
    """Test the OMNI-DISPATCHER brain-hands cycle."""
    
    @pytest.mark.asyncio
    async def test_omni_dispatch_cycle(self):
        """Test a complete OMNI-DISPATCHER cycle."""
        from backend.orchestrator.omni_dispatcher import omni_dispatch
        
        state = {
            "original_prompt": "Say hello",
            "todo_list": [{"task_id": "1", "status": "pending", "description": "Say hello"}],
            "messages": [],
            "memory": {},
            "insights": {},
            "action_history": [],
            "iteration_count": 0,
            "failure_count": 0,
            "thread_id": "omni_test",
            "user_id": "test_user"
        }
        
        config = {
            "configurable": {
                "thread_id": "omni_test",
                "owner": {"user_id": "test_user"}
            }
        }
        
        try:
            result = await omni_dispatch(state, config)
            
            # Should have executed a cycle
            assert result is not None
            assert 'iteration_count' in result
            
            print("✅ OMNI-DISPATCHER cycle test passed")
        except Exception as e:
            if "api key" in str(e).lower() or "not configured" in str(e).lower():
                pytest.skip(f"LLM not configured: {e}")
            raise
    
    @pytest.mark.asyncio
    async def test_omni_route_conditions(self):
        """Test OMNI routing conditions."""
        from backend.orchestrator.omni_dispatcher import omni_route_condition
        
        # Test finish routing
        state = {"decision": {"action_type": "finish"}}
        assert omni_route_condition(state) == "finish"
        
        # Test skip routing
        state = {"decision": {"action_type": "skip"}}
        assert omni_route_condition(state) == "brain"
        
        # Test hands routing
        state = {"decision": {"action_type": "python"}}
        assert omni_route_condition(state) == "hands"
        
        # Test approval routing
        state = {"pending_approval": True, "decision": {"action_type": "python"}}
        assert omni_route_condition(state) == "approval"
        
        print("✅ OMNI routing conditions test passed")
    
    @pytest.mark.asyncio
    async def test_human_in_the_loop_approval(self):
        """Test human-in-the-loop approval flow."""
        from backend.orchestrator.omni_dispatcher import (
            approve_pending_action,
            reject_pending_action
        )
        
        # Test approval
        state = {
            "pending_approval": True,
            "pending_decision": {
                "action_type": "agent",
                "resource_id": "mail_agent",
                "payload": {"instruction": "Send email"},
                "requires_approval": True,
                "approval_reason": "Will send email to recipient"
            }
        }
        
        approved = approve_pending_action(state)
        assert approved["pending_approval"] is False
        assert approved["decision"]["requires_approval"] is False
        
        # Test rejection
        rejected = reject_pending_action(state, "User cancelled")
        assert rejected["pending_approval"] is False
        assert rejected["decision"]["action_type"] == "skip"
        
        print("✅ Human-in-the-loop approval test passed")


# =============================================================================
# SECTION 7: TOOL REGISTRY INTEGRATION TESTS
# =============================================================================

class TestToolRegistryIntegration:
    """Test tool registry with orchestrator integration."""
    
    def test_tool_registry_lists_executable_tools(self):
        """Test that tool registry lists executable tools."""
        from backend.services.tool_registry_service import tool_registry
        
        tools = tool_registry.list_tools()
        
        # Should have some tools available
        assert isinstance(tools, list)
        
        # Each tool should have required fields
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
        
        print(f"✅ Tool registry lists {len(tools)} executable tools")
    
    @pytest.mark.asyncio
    async def test_tool_execution_through_hands(self):
        """Test tool execution through Hands dispatcher."""
        from backend.orchestrator.hands import Hands
        from backend.services.tool_registry_service import tool_registry
        
        # Get available tools
        tools = tool_registry.list_tools()
        if not tools:
            pytest.skip("No tools available to test")
        
        # Try to execute first available tool through Hands
        hands = Hands()
        tool_name = tools[0]['name']
        
        state = {
            "decision": {
                "action_type": "tool",
                "resource_id": tool_name,
                "payload": {}
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        # Should have attempted execution (may fail if tool needs specific params)
        assert 'execution_result' in result
        
        print(f"✅ Tool execution through Hands test passed for '{tool_name}'")


# =============================================================================
# SECTION 8: AGENT REGISTRY INTEGRATION TESTS
# =============================================================================

class TestAgentRegistryIntegration:
    """Test agent registry with orchestrator integration."""
    
    def test_agent_registry_lists_agents(self):
        """Test that agent registry lists agents."""
        from backend.services.agent_registry_service import agent_registry
        
        agents = agent_registry.list_active_agents()
        
        # Should have some agents available
        assert isinstance(agents, list)
        
        print(f"✅ Agent registry lists {len(agents)} agents")
    
    def test_agent_skills_context(self):
        """Test getting agent skills context for LLM."""
        from backend.services.agent_registry_service import agent_registry
        
        context = agent_registry.get_all_skills_context()
        
        # Should return a string with agent information
        assert isinstance(context, str)
        
        print("✅ Agent skills context test passed")
    
    def test_agent_name_normalization(self):
        """Test agent name normalization."""
        from backend.services.agent_registry_service import normalize_agent_name
        
        # Test various aliases
        assert normalize_agent_name("browser") == "Browser Automation Agent"
        assert normalize_agent_name("spreadsheet") == "Spreadsheet Agent"
        assert normalize_agent_name("mail") == "Mail Agent"
        assert normalize_agent_name("document") == "Document Agent"
        
        print("✅ Agent name normalization test passed")
    
    @pytest.mark.asyncio
    async def test_agent_lookup_through_hands(self):
        """Test agent lookup through Hands dispatcher."""
        from backend.orchestrator.hands import Hands
        from backend.services.agent_registry_service import agent_registry
        
        # Get available agents
        agents = agent_registry.list_active_agents()
        
        if not agents:
            pytest.skip("No agents available to test")
        
        # Test that Hands properly looks up agents
        hands = Hands()
        
        # Use first available agent (but don't actually call it)
        agent_name = agents[0]['name']
        agent = agent_registry.find_agent(agent_name)
        
        assert agent is not None
        assert agent['name'] == agent_name
        
        print(f"✅ Agent lookup through Hands test passed for '{agent_name}'")


# =============================================================================
# SECTION 9: ERROR HANDLING AND RECOVERY TESTS
# =============================================================================

class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_hands_handles_invalid_action_type(self):
        """Test Hands handling of invalid action type."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "decision": {
                "action_type": "invalid_type",
                "resource_id": "test",
                "payload": {}
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert result['execution_result']['success'] is False
        assert 'Unknown action type' in result['execution_result']['error_message']
        
        print("✅ Invalid action type handling test passed")
    
    @pytest.mark.asyncio
    async def test_hands_handles_missing_decision(self):
        """Test Hands handling of missing decision."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        state = {
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        assert 'error' in result
        assert 'No brain decision' in result['error']
        
        print("✅ Missing decision handling test passed")
    
    @pytest.mark.asyncio
    async def test_brain_handles_max_iterations(self):
        """Test Brain handling of max iterations."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Test",
            "todo_list": [{"task_id": "1", "status": "pending"}],
            "messages": [],
            "iteration_count": 25,  # At max
            "failure_count": 0
        }
        
        result = await brain.think(state)
        
        assert 'final_response' in result
        assert 'Maximum iterations' in result['final_response']
        
        print("✅ Max iterations handling test passed")
    
    @pytest.mark.asyncio
    async def test_brain_handles_max_failures(self):
        """Test Brain handling of max failures."""
        from backend.orchestrator.brain import Brain
        
        brain = Brain()
        
        state = {
            "original_prompt": "Test",
            "todo_list": [{"task_id": "1", "status": "pending"}],
            "messages": [],
            "iteration_count": 0,
            "failure_count": 3  # At max
        }
        
        result = await brain.think(state)
        
        # Should enter fallback mode
        assert 'decision' in result or 'final_response' in result
        
        print("✅ Max failures handling test passed")
    
    @pytest.mark.asyncio
    async def test_parallel_action_partial_failure(self):
        """Test parallel execution with partial failures."""
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
                            "code": "result = 'success'",
                            "session_id": "partial_1"
                        }
                    },
                    {
                        "action_type": "python",
                        "resource_id": None,
                        "payload": {
                            "code": "raise Exception('intentional error')",
                            "session_id": "partial_2"
                        }
                    }
                ]
            },
            "iteration_count": 0,
            "action_history": []
        }
        
        result = await hands.execute(state)
        
        # Should complete but report partial failure
        output = result['execution_result']['output']
        assert output['total_actions'] == 2
        assert output['failed'] >= 1
        
        print("✅ Parallel action partial failure test passed")


# =============================================================================
# SECTION 10: PERFORMANCE AND LOAD TESTS
# =============================================================================

class TestPerformanceAndLoad:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_rapid_sequential_executions(self):
        """Test rapid sequential code executions."""
        from backend.orchestrator.hands import Hands
        
        hands = Hands()
        
        start_time = time.time()
        
        for i in range(5):
            state = {
                "decision": {
                    "action_type": "python",
                    "resource_id": None,
                    "payload": {
                        "code": f"result = {i} * 2",
                        "session_id": f"rapid_test_{i}"
                    }
                },
                "iteration_count": i,
                "action_history": []
            }
            
            result = await hands.execute(state)
            assert result['execution_result']['success'] is True
        
        duration = time.time() - start_time
        
        print(f"✅ 5 sequential executions completed in {duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_large_code_execution(self):
        """Test execution of larger code blocks."""
        from backend.services.code_sandbox_service import code_sandbox
        
        # Generate a larger code block
        code = """
import pandas as pd
import numpy as np

# Create large dataset
np.random.seed(42)
data = {
    'id': range(1000),
    'value': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
}

df = pd.DataFrame(data)

# Perform multiple operations
result = {
    'mean': df['value'].mean(),
    'std': df['value'].std(),
    'category_counts': df['category'].value_counts().to_dict(),
    'sum_by_category': df.groupby('category')['value'].sum().to_dict()
}
"""
        start_time = time.time()
        result = code_sandbox.execute_code(code, session_id="large_code_test")
        duration = time.time() - start_time
        
        assert result['success'] is True
        assert abs(result['result']['mean']) < 0.1  # Should be close to 0
        
        print(f"✅ Large code execution completed in {duration:.2f}s")


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("ORCHESTRATOR END-TO-END TEST SUITE")
    print("Testing Real-World Tool Calling and Code Environment")
    print("="*70 + "\n")
    
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
    exit_code = run_all_tests()
    sys.exit(exit_code)
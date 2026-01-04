#!/usr/bin/env python3
"""
Test script for Multi-Stage Planning features
Tests the new 4-stage workflow: Propose → Revise → Simulate → Execute
"""

import asyncio
import logging
import sys
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import components
from agents.spreadsheet_agent.actions import (
    FilterAction, SortAction, AddColumnAction, RenameColumnAction,
    DropColumnAction, GroupByAction, FillNaAction, DropDuplicatesAction,
    AddSerialNumberAction, ActionParser, ActionExecutor
)
from agents.spreadsheet_agent.planner import (
    ExecutionPlan, PlanHistory, MultiStagePlanner, PlanStage, planner
)
from agents.spreadsheet_agent.llm_agent import query_agent


# ============================================================================
# TEST DATA
# ============================================================================

def create_test_dataframe() -> pd.DataFrame:
    """Create sample sales data for testing"""
    data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                 '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10'],
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Mouse',
                    'Monitor', 'Laptop', 'Keyboard', 'Mouse', 'Monitor'],
        'Quantity': [2, 5, 3, 1, 4, 2, 3, 2, 6, 1],
        'Price': [1000, 25, 50, 1000, 25, 300, 1000, 50, 25, 300],
        'Region': ['North', 'South', 'North', 'East', 'West',
                   'North', 'South', 'East', 'North', 'West']
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def load_sales_data() -> pd.DataFrame:
    """Load actual sales data if available"""
    test_data_path = Path(__file__).parent.parent / "test_data" / "sales_data.csv"
    if test_data_path.exists():
        return pd.read_csv(test_data_path)
    return create_test_dataframe()


# ============================================================================
# UNIT TESTS - ACTIONS
# ============================================================================

def test_filter_action():
    """Test FilterAction"""
    print("\n" + "="*80)
    print("🧪 TEST: FilterAction")
    print("="*80)
    
    df = create_test_dataframe()
    print(f"Original shape: {df.shape}")
    
    # Test 1: Numeric filter
    action = FilterAction(column="Price", operator=">", value=100)
    error = action.validate_against_df(df)
    assert error is None, f"Validation failed: {error}"
    
    result_df = action.execute(df)
    print(f"After filter (Price > 100): {result_df.shape}")
    assert len(result_df) < len(df), "Filter should reduce rows"
    assert all(result_df['Price'] > 100), "All prices should be > 100"
    
    # Test 2: String contains
    action2 = FilterAction(column="Product", operator="contains", value="Lap")
    result_df2 = action2.execute(df)
    print(f"After filter (Product contains 'Lap'): {result_df2.shape}")
    assert all("Lap" in str(p) for p in result_df2['Product']), "All products should contain 'Lap'"
    
    print("✅ FilterAction tests passed\n")


def test_sort_action():
    """Test SortAction"""
    print("\n" + "="*80)
    print("🧪 TEST: SortAction")
    print("="*80)
    
    df = create_test_dataframe()
    
    # Test ascending sort
    action = SortAction(columns=["Price"], ascending=[True])
    result_df = action.execute(df)
    
    print(f"Original first price: {df['Price'].iloc[0]}")
    print(f"Sorted first price: {result_df['Price'].iloc[0]}")
    
    # Check if sorted
    prices = result_df['Price'].tolist()
    assert prices == sorted(prices), "Prices should be sorted ascending"
    
    # Test descending sort
    action2 = SortAction(columns=["Price"], ascending=[False])
    result_df2 = action2.execute(df)
    prices2 = result_df2['Price'].tolist()
    assert prices2 == sorted(prices2, reverse=True), "Prices should be sorted descending"
    
    print("✅ SortAction tests passed\n")


def test_add_column_action():
    """Test AddColumnAction"""
    print("\n" + "="*80)
    print("🧪 TEST: AddColumnAction")
    print("="*80)
    
    df = create_test_dataframe()
    original_cols = len(df.columns)
    
    # Add calculated column
    action = AddColumnAction(
        new_column="Total",
        formula="{Quantity} * {Price}"
    )
    
    error = action.validate_against_df(df)
    assert error is None, f"Validation failed: {error}"
    
    result_df = action.execute(df)
    
    print(f"Original columns: {original_cols}")
    print(f"After adding Total: {len(result_df.columns)}")
    
    assert len(result_df.columns) == original_cols + 1, "Should have one more column"
    assert 'Total' in result_df.columns, "Total column should exist"
    
    # Verify calculation
    expected_total = df['Quantity'].iloc[0] * df['Price'].iloc[0]
    actual_total = result_df['Total'].iloc[0]
    assert actual_total == expected_total, f"Calculation incorrect: {actual_total} != {expected_total}"
    
    print("✅ AddColumnAction tests passed\n")


def test_groupby_action():
    """Test GroupByAction"""
    print("\n" + "="*80)
    print("🧪 TEST: GroupByAction")
    print("="*80)
    
    df = create_test_dataframe()
    
    # Group by Product, sum Quantity
    action = GroupByAction(
        group_columns=["Product"],
        agg_column="Quantity",
        agg_function="sum"
    )
    
    result_df = action.execute(df)
    
    print(f"Original shape: {df.shape}")
    print(f"After groupby: {result_df.shape}")
    print(f"Groups: {result_df.index.tolist()}")
    
    # Should have fewer rows (one per product)
    assert len(result_df) <= df['Product'].nunique(), "Should have one row per product"
    assert 'Quantity' in result_df.columns, "Quantity column should exist"
    
    print("✅ GroupByAction tests passed\n")


def test_add_serial_number_action():
    """Test AddSerialNumberAction"""
    print("\n" + "="*80)
    print("🧪 TEST: AddSerialNumberAction")
    print("="*80)
    
    df = create_test_dataframe()
    
    action = AddSerialNumberAction(
        column_name="Sl.No.",
        start=1,
        position="first"
    )
    
    result_df = action.execute(df)
    
    print(f"First 5 serial numbers: {result_df['Sl.No.'].head().tolist()}")
    
    assert 'Sl.No.' in result_df.columns, "Serial number column should exist"
    assert result_df.columns[0] == 'Sl.No.', "Should be first column"
    assert result_df['Sl.No.'].iloc[0] == 1, "Should start from 1"
    assert result_df['Sl.No.'].iloc[-1] == len(df), "Should end at row count"
    
    print("✅ AddSerialNumberAction tests passed\n")


# ============================================================================
# INTEGRATION TESTS - ACTION PARSER & EXECUTOR
# ============================================================================

def test_action_parser():
    """Test ActionParser"""
    print("\n" + "="*80)
    print("🧪 TEST: ActionParser")
    print("="*80)
    
    parser = ActionParser()
    
    # Test parsing single action
    action_dict = {
        "action_type": "filter",
        "column": "Price",
        "operator": ">",
        "value": 100
    }
    
    action = parser.parse(action_dict)
    print(f"Parsed action type: {type(action).__name__}")
    assert isinstance(action, FilterAction), "Should parse to FilterAction"
    assert action.column == "Price", "Column should match"
    
    # Test parsing multiple actions
    actions_list = [
        {"action_type": "filter", "column": "Price", "operator": ">", "value": 50},
        {"action_type": "sort", "columns": ["Date"], "ascending": [True]},
        {"action_type": "add_serial_number", "column_name": "ID", "start": 1, "position": "first"}
    ]
    
    actions = parser.parse_multiple(actions_list)
    print(f"Parsed {len(actions)} actions")
    assert len(actions) == 3, "Should parse all actions"
    assert isinstance(actions[0], FilterAction), "First should be FilterAction"
    assert isinstance(actions[1], SortAction), "Second should be SortAction"
    
    print("✅ ActionParser tests passed\n")


def test_action_executor():
    """Test ActionExecutor"""
    print("\n" + "="*80)
    print("🧪 TEST: ActionExecutor")
    print("="*80)
    
    df = create_test_dataframe()
    executor = ActionExecutor()
    
    # Create action sequence
    actions = [
        FilterAction(column="Price", operator=">", value=50),
        SortAction(columns=["Price"], ascending=[False]),
        AddColumnAction(new_column="Total", formula="{Quantity} * {Price}")
    ]
    
    # Execute sequence
    result_df, exec_result = executor.execute_actions(actions, df)
    
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {result_df.shape}")
    # exec_result is a list of log entries
    all_success = all(step.get('success', False) for step in exec_result)
    print(f"All actions succeeded: {all_success}")
    print(f"Total actions: {len(exec_result)}")
    
    # exec_result is a list of log entries, check if all succeeded
    assert all_success, "Execution should succeed"
    assert len(exec_result) == 3, "Should execute all 3 actions"
    assert 'Total' in result_df.columns, "Total column should be added"
    assert all(result_df['Price'] > 50), "All prices should be > 50"
    
    print("✅ ActionExecutor tests passed\n")


# ============================================================================
# INTEGRATION TESTS - MULTI-STAGE PLANNER
# ============================================================================

async def test_propose_plan():
    """Test plan proposal"""
    print("\n" + "="*80)
    print("🧪 TEST: Propose Plan")
    print("="*80)
    
    df = load_sales_data()
    instruction = "Filter products where Amount is greater than 50000 and sort by Date"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Instruction: {instruction}")
    
    # Propose plan
    plan = await planner.propose_plan(df, instruction, df_context)
    
    print(f"\n📋 Generated Plan:")
    print(f"  Plan ID: {plan.plan_id}")
    print(f"  Stage: {plan.stage}")
    print(f"  Actions: {len(plan.actions)}")
    print(f"  Reasoning: {plan.reasoning[:100]}...")
    
    for i, action in enumerate(plan.actions):
        print(f"  {i+1}. {action.action_type}: {action.dict()}")
    
    assert plan.stage == PlanStage.PROPOSING, "Should be in PROPOSING stage"
    assert len(plan.actions) > 0, "Should have at least one action"
    
    print("✅ Propose plan test passed\n")
    return plan, df


async def test_simulate_plan():
    """Test plan simulation"""
    print("\n" + "="*80)
    print("🧪 TEST: Simulate Plan")
    print("="*80)
    
    # First create a plan
    df = load_sales_data()
    instruction = "Filter Amount > 50000 and sort by Date"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    plan = await planner.propose_plan(df, instruction, df_context)
    print(f"Created plan with {len(plan.actions)} actions")
    
    # Simulate
    sim_result = planner.simulate_plan(plan, df)
    
    print(f"\n🔬 Simulation Result:")
    print(f"  Success: {sim_result['success']}")
    print(f"  Before shape: {sim_result['observation']['before_shape']}")
    print(f"  After shape: {sim_result['observation']['after_shape']}")
    print(f"  Changes: {sim_result['observation']['changes_summary']}")
    
    if sim_result['warnings']:
        print(f"  ⚠️  Warnings:")
        for warning in sim_result['warnings']:
            print(f"    - {warning}")
    
    assert sim_result['success'], "Simulation should succeed"
    assert 'observation' in sim_result, "Should have observation data"
    
    print("✅ Simulate plan test passed\n")
    return plan, df


async def test_execute_plan():
    """Test plan execution"""
    print("\n" + "="*80)
    print("🧪 TEST: Execute Plan")
    print("="*80)
    
    # Create and simulate plan
    df = load_sales_data()
    instruction = "Add a serial number column at the beginning"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    plan = await planner.propose_plan(df, instruction, df_context)
    print(f"Created plan: {plan.plan_id}")
    
    # Simulate first
    sim_result = planner.simulate_plan(plan, df)
    print(f"Simulation: {sim_result['success']}")
    
    # Execute
    result_df, exec_result = planner.execute_plan(plan, df)
    
    print(f"\n⚡ Execution Result:")
    print(f"  Success: {exec_result['success']}")
    print(f"  Actions executed: {exec_result['actions_executed']}")
    print(f"  Final shape: {exec_result['final_shape']}")
    print(f"  New columns: {result_df.columns.tolist()[:3]}...")
    
    assert exec_result['success'], "Execution should succeed"
    assert result_df.shape[0] == df.shape[0], "Row count should match"
    
    print("✅ Execute plan test passed\n")


async def test_revise_plan():
    """Test plan revision"""
    print("\n" + "="*80)
    print("🧪 TEST: Revise Plan")
    print("="*80)
    
    df = load_sales_data()
    instruction = "Sort by Amount"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    # Initial plan
    plan = await planner.propose_plan(df, instruction, df_context)
    print(f"Initial plan: {len(plan.actions)} actions")
    print(f"  Action: {plan.actions[0].dict() if plan.actions else 'None'}")
    
    # Revise with feedback
    feedback = "Sort in descending order instead"
    revised_plan = await planner.revise_plan(plan, feedback, df)
    
    print(f"\n📝 Revised Plan:")
    print(f"  Revisions count: {len(revised_plan.revisions)}")
    print(f"  Stage: {revised_plan.stage}")
    if revised_plan.revisions:
        print(f"  Last revision: {revised_plan.revisions[-1]['feedback']}")
    
    assert revised_plan.plan_id == plan.plan_id, "Should be same plan"
    assert len(revised_plan.revisions) > 0, "Should have revision history"
    assert revised_plan.stage == PlanStage.REVISING, "Should be in REVISING stage"
    
    print("✅ Revise plan test passed\n")


async def test_error_correction():
    """Test error correction with plan history"""
    print("\n" + "="*80)
    print("🧪 TEST: Error Correction with History")
    print("="*80)
    
    df = create_test_dataframe()
    
    # Simulate a failure with common keywords
    failed_instruction = "Filter by Price > 1000"  
    planner.history.add_failure(
        instruction=failed_instruction,
        error="Column 'Price' has no values > 1000. Maximum value is 1000.",
        actions=[{"action_type": "filter", "column": "Price", "operator": ">", "value": 1000}]
    )
    
    print(f"Added failure to history")
    print(f"Total failures tracked: {len(planner.history.failed_patterns)}")
    
    # Try similar instruction with common keyword "Price"
    similar_instruction = "Filter where Price exceeds 500"
    similar_failures = planner.history.get_similar_failures(similar_instruction, top_k=2)
    
    print(f"\nSimilar failures found: {len(similar_failures)}")
    for failure in similar_failures:
        print(f"  - Instruction: {failure['instruction']}")
        print(f"    Error: {failure['error'][:50]}...")
    
    assert len(similar_failures) > 0, "Should find similar failures"
    
    print("✅ Error correction test passed\n")


# ============================================================================
# END-TO-END WORKFLOW TESTS
# ============================================================================

async def test_complete_workflow():
    """Test complete 4-stage workflow"""
    print("\n" + "="*80)
    print("🧪 TEST: Complete 4-Stage Workflow")
    print("="*80)
    
    df = load_sales_data()
    instruction = "Show products with Amount greater than 40000, sorted by Date descending, with a serial number column"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    print(f"Starting workflow...")
    print(f"  Instruction: {instruction}")
    print(f"  Original shape: {df.shape}")
    
    # STAGE 1: Propose
    print("\n📋 STAGE 1: PROPOSE")
    plan = await planner.propose_plan(df, instruction, df_context)
    print(f"  Generated {len(plan.actions)} actions")
    for i, action in enumerate(plan.actions):
        print(f"    {i+1}. {action.action_type}")
    
    # STAGE 2: Simulate
    print("\n🔬 STAGE 2: SIMULATE")
    sim_result = planner.simulate_plan(plan, df)
    print(f"  Success: {sim_result['success']}")
    print(f"  Shape: {sim_result['observation']['before_shape']} → {sim_result['observation']['after_shape']}")
    if sim_result['warnings']:
        print(f"  Warnings: {len(sim_result['warnings'])}")
    
    # STAGE 3: Execute
    print("\n⚡ STAGE 3: EXECUTE")
    result_df, exec_result = planner.execute_plan(plan, df)
    print(f"  Success: {exec_result['success']}")
    print(f"  Final shape: {result_df.shape}")
    print(f"  Columns: {result_df.columns.tolist()}")
    
    assert exec_result['success'], "Complete workflow should succeed"
    assert result_df.shape[0] <= df.shape[0], "Should have same or fewer rows"
    
    print("\n✅ Complete workflow test passed\n")


# ============================================================================
# LLM INTEGRATION TESTS
# ============================================================================

async def test_llm_action_generation():
    """Test LLM-powered action generation"""
    print("\n" + "="*80)
    print("🧪 TEST: LLM Action Generation")
    print("="*80)
    
    df = create_test_dataframe()
    instruction = "Filter products where Price is above 100 and add a Total column"
    
    df_context = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(3).to_dict(orient='records')
    }
    
    print(f"Testing LLM action generation...")
    print(f"  Instruction: {instruction}")
    
    try:
        actions_json, reasoning = await query_agent.generate_actions(df, instruction, df_context)
        
        print(f"\n🤖 LLM Generated:")
        print(f"  Actions count: {len(actions_json)}")
        print(f"  Reasoning: {reasoning[:100]}...")
        
        for i, action in enumerate(actions_json):
            print(f"  {i+1}. {action.get('action_type')}: {action}")
        
        assert len(actions_json) > 0, "Should generate at least one action"
        print("\n✅ LLM action generation test passed\n")
        
    except Exception as e:
        print(f"⚠️  LLM action generation failed (may be expected if no API keys): {e}")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_performance():
    """Test performance of action execution"""
    print("\n" + "="*80)
    print("🧪 TEST: Performance")
    print("="*80)
    
    import time
    
    # Create larger dataset
    large_df = pd.DataFrame({
        'A': range(10000),
        'B': [i * 2 for i in range(10000)],
        'C': [f"Item_{i}" for i in range(10000)]
    })
    
    print(f"Testing with {len(large_df)} rows")
    
    executor = ActionExecutor()
    actions = [
        FilterAction(column="A", operator=">", value=5000),
        SortAction(columns=["B"], ascending=[False]),
        AddColumnAction(new_column="D", formula="{A} + {B}")
    ]
    
    start = time.time()
    result_df, exec_result = executor.execute_actions(actions, large_df)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Performance:")
    print(f"  Actions: {len(actions)}")
    print(f"  Rows processed: {len(large_df)}")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Throughput: {len(large_df)/elapsed:.0f} rows/second")
    
    # exec_result is a list of log entries
    all_success = all(step.get('success', False) for step in exec_result)
    assert all_success, "Execution should succeed"
    assert elapsed < 5.0, "Should complete in reasonable time"
    
    print("✅ Performance test passed\n")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 MULTI-STAGE PLANNING TEST SUITE")
    print("="*80)
    print(f"Testing new 4-stage planning workflow...")
    print(f"Actions → Planner → LLM Integration")
    print("="*80 + "\n")
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    tests = [
        # Unit tests
        ("Filter Action", test_filter_action, False),
        ("Sort Action", test_sort_action, False),
        ("Add Column Action", test_add_column_action, False),
        ("GroupBy Action", test_groupby_action, False),
        ("Add Serial Number Action", test_add_serial_number_action, False),
        
        # Parser tests
        ("Action Parser", test_action_parser, False),
        ("Action Executor", test_action_executor, False),
        
        # Planner tests
        ("Propose Plan", test_propose_plan, True),
        ("Simulate Plan", test_simulate_plan, True),
        ("Execute Plan", test_execute_plan, True),
        ("Revise Plan", test_revise_plan, True),
        ("Error Correction", test_error_correction, True),
        
        # End-to-end
        ("Complete Workflow", test_complete_workflow, True),
        
        # LLM tests
        ("LLM Action Generation", test_llm_action_generation, True),
        
        # Performance
        ("Performance", test_performance, False),
    ]
    
    for test_name, test_func, is_async in tests:
        try:
            if is_async:
                await test_func()
            else:
                test_func()
            test_results["passed"] += 1
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append((test_name, str(e)))
            print(f"❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}\n")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    print(f"📈 Success Rate: {test_results['passed']/(test_results['passed']+test_results['failed'])*100:.1f}%")
    
    if test_results['errors']:
        print("\n❌ Failed Tests:")
        for test_name, error in test_results['errors']:
            print(f"  - {test_name}: {error[:100]}")
    
    print("="*80 + "\n")
    
    return test_results['failed'] == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

"""
Test Artifact Integration with Orchestrator

This test verifies that the artifact system automatically integrates
with the orchestrator for context management.
"""

import os
import sys
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.artifact_integration import (
    hooks,
    ArtifactConfig,
    compress_task_result,
    compress_canvas_content,
    compress_state_for_saving,
    expand_state_from_saved,
    get_optimized_llm_context
)
from services.artifact_service import get_artifact_store


def test_task_result_compression():
    """Test automatic compression of large task results"""
    print("\n=== Test: Task Result Compression ===")
    
    # Small result - should not be compressed
    small_result = {"status": "success", "data": "small"}
    compressed = compress_task_result(small_result, "small_task", "test-thread-1")
    assert "_artifact_ref" not in compressed, "Small result should not be compressed"
    print("✓ Small results kept inline")
    
    # Large result - should be compressed
    large_result = {
        "status": "success",
        "data": "x" * 5000,  # 5KB of data
        "nested": {"more": "data" * 100}
    }
    compressed = compress_task_result(large_result, "large_task", "test-thread-1")
    assert "_artifact_ref" in compressed, "Large result should be compressed"
    assert "summary" in compressed, "Compressed result should have summary"
    print(f"✓ Large result compressed to artifact: {compressed['_artifact_ref']['id']}")
    
    return True


def test_canvas_compression():
    """Test automatic compression of canvas content"""
    print("\n=== Test: Canvas Compression ===")
    
    canvas_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test Dashboard</title></head>
    <body>
        <h1>Dashboard</h1>
        <div class="chart">Chart content here...</div>
    </body>
    </html>
    """ * 10  # Make it large
    
    result = compress_canvas_content(canvas_html, "html", "test-thread-2")
    assert "_artifact_ref" in result, "Canvas should be compressed"
    assert result["canvas_type"] == "html"
    print(f"✓ Canvas compressed to artifact: {result['_artifact_ref']['id']}")
    
    return True


def test_state_compression_and_expansion():
    """Test state compression for saving and expansion for loading"""
    print("\n=== Test: State Compression/Expansion ===")
    
    # Create a state with large fields
    state = {
        "thread_id": "test-thread-3",
        "original_prompt": "Test prompt",
        "final_response": "Test response",
        "canvas_content": "<html>" + "x" * 2000 + "</html>",
        "canvas_type": "html",
        "completed_tasks": [
            {"task_name": f"task_{i}", "result": {"data": "x" * 500}}
            for i in range(5)
        ],
        "messages": [
            {"type": "user", "content": "Hello"},
            {"type": "assistant", "content": "Hi there!"}
        ]
    }
    
    # Compress for saving
    compressed = compress_state_for_saving(state, "test-thread-3")
    
    # Check that large fields were compressed
    has_refs = "_artifact_refs" in compressed
    print(f"✓ State compressed, has artifact refs: {has_refs}")
    
    if has_refs:
        print(f"  Compressed fields: {list(compressed['_artifact_refs'].keys())}")
    
    # Expand after loading
    expanded = expand_state_from_saved(compressed)
    
    # Verify expansion worked
    if has_refs:
        # Canvas should be expanded back
        assert "canvas_content" in expanded
        print("✓ State expanded successfully")
    
    return True


def test_context_optimization():
    """Test context optimization for LLM calls"""
    print("\n=== Test: Context Optimization ===")
    
    # Create a state with various fields
    state = {
        "original_prompt": "Search for AI news and create a dashboard",
        "messages": [
            {"type": "user", "content": "Search for AI news"},
            {"type": "assistant", "content": "Searching..." * 50},
        ],
        "completed_tasks": [
            {"task_name": "search", "result": {"articles": list(range(100))}}
        ],
        "task_agent_pairs": [
            {"task_name": "search", "primary": {"name": "SearchAgent", "capabilities": ["search"] * 20}}
        ]
    }
    
    # Get optimized context
    result = get_optimized_llm_context(state, "test-thread-4", max_tokens=2000)
    
    print(f"✓ Context optimized:")
    print(f"  Tokens used: {result['tokens_used']}")
    print(f"  Tokens saved: {result['tokens_saved']}")
    print(f"  Artifact refs: {len(result['artifact_refs'])}")
    print(f"  Context preview: {result['context'][:200]}...")
    
    return True


def test_hooks_integration():
    """Test that hooks are properly integrated"""
    print("\n=== Test: Hooks Integration ===")
    
    # Test on_task_complete hook
    result = hooks.on_task_complete(
        task_name="test_task",
        result={"data": "x" * 3000},
        thread_id="test-thread-5"
    )
    print(f"✓ on_task_complete hook works: {'_artifact_ref' in result}")
    
    # Test on_canvas_generated hook
    result = hooks.on_canvas_generated(
        canvas_content="<html>Test</html>" * 100,
        canvas_type="html",
        thread_id="test-thread-5"
    )
    print(f"✓ on_canvas_generated hook works: {'_artifact_ref' in result}")
    
    # Test before_save hook
    state = {"canvas_content": "<html>Large content</html>" * 100}
    result = hooks.before_save(state, "test-thread-5")
    print(f"✓ before_save hook works")
    
    # Test after_load hook
    result = hooks.after_load(result)
    print(f"✓ after_load hook works")
    
    return True


def test_orchestrator_integration():
    """Test that orchestrator imports work"""
    print("\n=== Test: Orchestrator Integration ===")
    
    from orchestrator.graph import ARTIFACT_INTEGRATION_ENABLED, artifact_hooks
    
    assert ARTIFACT_INTEGRATION_ENABLED, "Artifact integration should be enabled"
    assert artifact_hooks is not None, "Artifact hooks should be available"
    
    print(f"✓ ARTIFACT_INTEGRATION_ENABLED: {ARTIFACT_INTEGRATION_ENABLED}")
    print(f"✓ artifact_hooks available: {artifact_hooks is not None}")
    
    return True


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("ARTIFACT INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_task_result_compression,
        test_canvas_compression,
        test_state_compression_and_expansion,
        test_context_optimization,
        test_hooks_integration,
        test_orchestrator_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Show storage stats
    store = get_artifact_store()
    stats = store.get_stats()
    print(f"\nArtifact Storage Stats:")
    print(f"  Total artifacts: {stats['total_artifacts']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

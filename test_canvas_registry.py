"""Verification tests for Canvas Registry."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.canvas_registry import CanvasRegistry, get_canvas_registry
from backend.schemas import CanvasEntry, CanvasRegistryState

print("=" * 60)
print("CANVAS REGISTRY UNIT TESTS")
print("=" * 60)

# Test 1: Register single canvas
r = CanvasRegistry("test_thread")
e = r.register_sync(
    "c1", "spreadsheet", 
    source_agent="spreadsheet_agent", 
    canvas_data={"rows": [[1,2]], "headers": ["a","b"]},
    canvas_title="Sales Data"
)
assert e.canvas_id == "c1"
assert e.version == 1
assert e.priority == 50  # PRIORITY_RESULT for spreadsheet
assert e.source_agent == "spreadsheet_agent"
print("1. Single canvas registration: PASS")

# Test 2: Register confirmation canvas (should auto-focus)
e2 = r.register_sync(
    "c2", "email_preview", 
    source_agent="mail_agent",
    requires_confirmation=True,
    confirmation_message="Send this email?"
)
assert e2.priority == 100  # PRIORITY_CONFIRMATION
assert r.get_active_id() == "c2"  # Confirmation should auto-focus
print("2. Confirmation canvas auto-focus: PASS")

# Test 3: Multiple canvases coexist
state = r.get_registry_state()
assert len(state.canvases) == 2
assert "c1" in state.canvases
assert "c2" in state.canvases
print("3. Multiple canvases coexist: PASS")

# Test 4: Canvas order (by priority desc)
assert state.canvas_order[0] == "c2"  # Higher priority first
assert state.canvas_order[1] == "c1"
print("4. Priority ordering: PASS")

# Test 5: Version auto-increment
e1_updated = r.register_sync(
    "c1", "spreadsheet",
    source_agent="spreadsheet_agent",
    canvas_data={"rows": [[1,2],[3,4]], "headers": ["a","b"]},
    canvas_title="Updated Sales Data"
)
assert e1_updated.version == 2
assert e1_updated.canvas_title == "Updated Sales Data"
print("5. Version auto-increment: PASS")

# Test 6: Backward compat fields
compat = r.get_backward_compat_fields()
assert compat["has_canvas"] == True
assert compat["canvas_type"] == "email_preview"  # Active canvas type
print("6. Backward compat fields: PASS")

# Test 7: Get by agent
agent_canvases = r.get_by_agent("spreadsheet_agent")
assert len(agent_canvases) == 1
assert agent_canvases[0].canvas_id == "c1"
print("7. Get by agent: PASS")

# Test 8: Thread-level singleton
r2 = get_canvas_registry("test_thread_2")
r2.register_sync("c3", "document", source_agent="doc_agent")
r1 = get_canvas_registry("test_thread_2")
assert r1 is r2
assert len(r1.get_all()) == 1
print("8. Thread singleton: PASS")

# Test 9: Registry state serialization
state = r.get_registry_state()
dumped = state.model_dump()
assert isinstance(dumped, dict)
assert "canvases" in dumped
assert "active_canvas_id" in dumped
assert "canvas_order" in dumped
print("9. Serialization: PASS")

# Test 10: Dismiss
import asyncio
async def test_dismiss():
    success = await r.dismiss("c2")
    assert success
    assert r.get_active_id() == "c1"  # Should auto-focus to c1 after dismissing c2
    state = r.get_registry_state()
    assert len(state.canvases) == 1  # Only active canvases shown
    return True

asyncio.run(test_dismiss())
print("10. Dismiss + auto-refocus: PASS")

print()
print("=" * 60)
print("ALL 10 TESTS PASSED")
print("=" * 60)

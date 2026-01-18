"""
Unit tests for DialogueManager

Tests response formatting for success/error/needs_input cases and anomaly response formatting.

Requirements: 9.1, 9.2, 9.3, 9.4, 12.1, 12.2, 12.4, 12.5
"""

import pytest
import time
from typing import Dict, Any
from dataclasses import dataclass, field

# Import the dialogue manager components
import sys
from pathlib import Path
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from agents.spreadsheet_agent.dialogue_manager import (
    DialogueManager,
    ResponseStatus,
    ExecutionMetrics,
    AgentResponse
)

# Import anomaly detector for testing anomaly responses
from agents.spreadsheet_agent.anomaly_detector import Anomaly, AnomalyFix


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dialogue_manager():
    """Create a fresh DialogueManager instance for each test"""
    return DialogueManager()


@pytest.fixture
def sample_metrics():
    """Create sample execution metrics"""
    return ExecutionMetrics(
        latency_ms=123.45,
        rows_processed=100,
        columns_affected=5,
        llm_calls=2,
        cache_hits=1,
        token_usage={"prompt": 50, "completion": 30}
    )


@pytest.fixture
def sample_anomaly():
    """Create a sample dtype drift anomaly"""
    return Anomaly(
        type='dtype_drift',
        columns=['Revenue'],
        sample_values={'Revenue': ['N/A', 'TBD', '$1,234']},
        suggested_fixes=[
            AnomalyFix(
                action='convert_numeric',
                description="Convert 'Revenue' to numeric, replacing invalid values with NaN",
                safe=True,
                parameters={'column': 'Revenue', 'method': 'coerce'}
            ),
            AnomalyFix(
                action='ignore_rows',
                description="Filter out rows with non-numeric values in 'Revenue'",
                safe=False,
                parameters={'column': 'Revenue', 'keep_numeric_only': True}
            ),
            AnomalyFix(
                action='treat_as_text',
                description="Keep 'Revenue' as text (may cause calculation failures)",
                safe=True,
                parameters={'column': 'Revenue', 'no_conversion': True}
            )
        ],
        severity='warning',
        message="Column 'Revenue' contains 85.0% numeric values but has object dtype.",
        metadata={
            'numeric_percentage': 0.85,
            'total_values': 100,
            'numeric_values': 85,
            'non_numeric_values': 15
        }
    )


# ============================================================================
# TEST SUCCESS RESPONSES (Task 9.1, Req 12.1, 12.2)
# ============================================================================

def test_create_success_response_basic(dialogue_manager):
    """Test creating a basic success response"""
    result = {"data": [1, 2, 3], "count": 3}
    explanation = "Successfully retrieved 3 rows"
    
    response = dialogue_manager.create_success_response(
        result=result,
        explanation=explanation
    )
    
    assert response.status == ResponseStatus.COMPLETE
    assert response.result == result
    assert response.explanation == explanation
    assert response.error is None


def test_create_success_response_with_metrics(dialogue_manager, sample_metrics):
    """Test success response includes execution metrics"""
    result = {"sum": 1000}
    explanation = "Calculated sum of column"
    
    response = dialogue_manager.create_success_response(
        result=result,
        explanation=explanation,
        metrics=sample_metrics
    )
    
    assert response.status == ResponseStatus.COMPLETE
    assert response.metrics == sample_metrics
    
    # Verify metrics serialization
    response_dict = response.to_dict()
    assert "metrics" in response_dict
    assert response_dict["metrics"]["latency_ms"] == 123.45
    assert response_dict["metrics"]["rows_processed"] == 100


def test_create_success_response_with_metadata(dialogue_manager):
    """Test success response includes metadata"""
    result = [{"name": "Alice", "age": 30}]
    explanation = "Found 1 matching record"
    metadata = {"query_type": "filter", "cache_hit": True}
    
    response = dialogue_manager.create_success_response(
        result=result,
        explanation=explanation,
        metadata=metadata
    )
    
    assert response.status == ResponseStatus.COMPLETE
    assert response.metadata == metadata
    
    response_dict = response.to_dict()
    assert response_dict["metadata"]["query_type"] == "filter"
    assert response_dict["metadata"]["cache_hit"] is True


# ============================================================================
# TEST ERROR RESPONSES (Task 9.1, Req 12.1)
# ============================================================================

def test_create_error_response_basic(dialogue_manager):
    """Test creating a basic error response"""
    error_message = "Column 'Revenue' not found"
    
    response = dialogue_manager.create_error_response(
        error_message=error_message
    )
    
    assert response.status == ResponseStatus.ERROR
    assert response.error == error_message
    assert "Operation failed" in response.explanation
    assert response.result is None


def test_create_error_response_with_details(dialogue_manager):
    """Test error response includes detailed error information"""
    error_message = "Unable to compute average"
    error_details = {
        "error_type": "TypeError",
        "column": "Revenue",
        "sample_invalid_values": ["N/A", "TBD"],
        "suggested_action": "Convert to numeric or filter invalid rows"
    }
    
    response = dialogue_manager.create_error_response(
        error_message=error_message,
        error_details=error_details
    )
    
    assert response.status == ResponseStatus.ERROR
    assert response.error == error_message
    
    response_dict = response.to_dict()
    assert "metadata" in response_dict
    assert "error_details" in response_dict["metadata"]
    assert response_dict["metadata"]["error_details"]["error_type"] == "TypeError"


def test_create_error_response_with_metrics(dialogue_manager, sample_metrics):
    """Test error response includes execution metrics"""
    error_message = "Operation timed out"
    
    response = dialogue_manager.create_error_response(
        error_message=error_message,
        metrics=sample_metrics
    )
    
    assert response.status == ResponseStatus.ERROR
    assert response.metrics == sample_metrics


# ============================================================================
# TEST PARTIAL RESPONSES (Task 9.1, Req 12.5)
# ============================================================================

def test_create_partial_response(dialogue_manager):
    """Test creating a partial result response"""
    partial_result = {"processed_rows": 500, "total_rows": 1000}
    explanation = "Processing 50% complete"
    progress = 0.5
    
    response = dialogue_manager.create_partial_response(
        partial_result=partial_result,
        explanation=explanation,
        progress=progress
    )
    
    assert response.status == ResponseStatus.PARTIAL
    assert response.partial_result == partial_result
    assert response.progress == 0.5
    assert response.explanation == explanation


def test_partial_response_serialization(dialogue_manager):
    """Test partial response serializes correctly"""
    partial_result = {"current_step": 2, "total_steps": 5}
    
    response = dialogue_manager.create_partial_response(
        partial_result=partial_result,
        explanation="Step 2 of 5 complete",
        progress=0.4
    )
    
    response_dict = response.to_dict()
    assert response_dict["status"] == "partial"
    assert response_dict["partial_result"] == partial_result
    assert response_dict["progress"] == 0.4


# ============================================================================
# TEST NEEDS_INPUT RESPONSES (Task 9.2, Req 9.1, 9.2)
# ============================================================================

def test_create_needs_input_response_basic(dialogue_manager):
    """Test creating a basic NEEDS_INPUT response"""
    question = "Which column should be used for grouping?"
    
    response = dialogue_manager.create_needs_input_response(
        question=question,
        question_type="text"
    )
    
    assert response.status == ResponseStatus.NEEDS_INPUT
    assert response.question == question
    assert response.question_type == "text"
    assert "User input required" in response.explanation


def test_create_needs_input_response_with_choices(dialogue_manager):
    """Test NEEDS_INPUT response with multiple choices"""
    question = "How should missing values be handled?"
    choices = [
        {"id": "drop", "label": "Drop rows", "description": "Remove rows with missing values"},
        {"id": "fill", "label": "Fill with mean", "description": "Replace with column mean"},
        {"id": "ignore", "label": "Ignore", "description": "Proceed without changes"}
    ]
    
    response = dialogue_manager.create_needs_input_response(
        question=question,
        question_type="choice",
        choices=choices
    )
    
    assert response.status == ResponseStatus.NEEDS_INPUT
    assert response.question == question
    assert response.question_type == "choice"
    assert response.choices == choices
    assert len(response.choices) == 3


def test_create_needs_input_response_with_context(dialogue_manager):
    """Test NEEDS_INPUT response includes context"""
    question = "Confirm deletion of 50 rows?"
    context = {
        "operation": "delete",
        "affected_rows": 50,
        "total_rows": 1000,
        "reason": "duplicate_detection"
    }
    
    response = dialogue_manager.create_needs_input_response(
        question=question,
        question_type="confirmation",
        context=context
    )
    
    assert response.status == ResponseStatus.NEEDS_INPUT
    assert response.context == context
    
    response_dict = response.to_dict()
    assert response_dict["context"]["affected_rows"] == 50


# ============================================================================
# TEST ANOMALY RESPONSES (Task 9.2, Req 10.2)
# ============================================================================

def test_create_anomaly_response_dtype_drift(dialogue_manager, sample_anomaly):
    """Test creating NEEDS_INPUT response for dtype drift anomaly"""
    response = dialogue_manager.create_anomaly_response(
        anomaly=sample_anomaly
    )
    
    assert response.status == ResponseStatus.NEEDS_INPUT
    assert "Revenue" in response.question
    assert "numeric values" in response.question
    assert response.question_type == "choice"
    assert len(response.choices) == 3


def test_anomaly_response_includes_all_fix_options(dialogue_manager, sample_anomaly):
    """Test anomaly response includes all suggested fixes as choices"""
    response = dialogue_manager.create_anomaly_response(
        anomaly=sample_anomaly
    )
    
    # Verify all fix actions are present
    choice_ids = [choice["id"] for choice in response.choices]
    assert "convert_numeric" in choice_ids
    assert "ignore_rows" in choice_ids
    assert "treat_as_text" in choice_ids
    
    # Verify safety indicators
    for choice in response.choices:
        assert "is_safe" in choice
        if choice["id"] == "ignore_rows":
            assert choice["is_safe"] is False  # Not safe (removes data)
        elif choice["id"] == "convert_numeric":
            assert choice["is_safe"] is True  # Safe


def test_anomaly_response_includes_context(dialogue_manager, sample_anomaly):
    """Test anomaly response includes detailed context"""
    response = dialogue_manager.create_anomaly_response(
        anomaly=sample_anomaly
    )
    
    assert response.context is not None
    assert response.context["anomaly_type"] == "dtype_drift"
    assert response.context["affected_columns"] == ["Revenue"]
    assert response.context["severity"] == "warning"
    assert "sample_values" in response.context
    assert "Revenue" in response.context["sample_values"]


def test_anomaly_response_serialization(dialogue_manager, sample_anomaly):
    """Test anomaly response serializes correctly to dict"""
    response = dialogue_manager.create_anomaly_response(
        anomaly=sample_anomaly
    )
    
    response_dict = response.to_dict()
    
    assert response_dict["status"] == "needs_input"
    assert "question" in response_dict
    assert "choices" in response_dict
    assert "context" in response_dict
    assert len(response_dict["choices"]) == 3


# ============================================================================
# TEST DIALOGUE STATE MANAGEMENT (Req 9.3, 9.4)
# ============================================================================

def test_save_and_load_state(dialogue_manager):
    """Test saving and loading dialogue state"""
    thread_id = "thread_123"
    state = {
        "file_id": "file_456",
        "operation": "aggregate",
        "last_query": "sum of revenue"
    }
    
    dialogue_manager.save_state(thread_id, state)
    loaded_state = dialogue_manager.load_state(thread_id)
    
    assert loaded_state["file_id"] == "file_456"
    assert loaded_state["operation"] == "aggregate"
    assert loaded_state["last_query"] == "sum of revenue"


def test_load_state_nonexistent_thread(dialogue_manager):
    """Test loading state for non-existent thread returns empty dict"""
    loaded_state = dialogue_manager.load_state("nonexistent_thread")
    assert loaded_state == {}


def test_clear_state(dialogue_manager):
    """Test clearing dialogue state"""
    thread_id = "thread_123"
    state = {"file_id": "file_456"}
    
    dialogue_manager.save_state(thread_id, state)
    assert dialogue_manager.load_state(thread_id) == state
    
    dialogue_manager.clear_state(thread_id)
    assert dialogue_manager.load_state(thread_id) == {}


def test_set_and_get_pending_question(dialogue_manager):
    """Test setting and getting pending questions"""
    thread_id = "thread_123"
    question = "How should missing values be handled?"
    context = {"column": "Revenue", "missing_count": 10}
    
    dialogue_manager.set_pending_question(thread_id, question, context)
    
    pending = dialogue_manager.get_pending_question(thread_id)
    assert pending == question
    
    # Verify context was saved
    loaded_state = dialogue_manager.load_state(thread_id)
    assert loaded_state["column"] == "Revenue"


def test_clear_pending_question(dialogue_manager):
    """Test clearing pending questions"""
    thread_id = "thread_123"
    question = "Confirm operation?"
    
    dialogue_manager.set_pending_question(thread_id, question, {})
    assert dialogue_manager.get_pending_question(thread_id) == question
    
    dialogue_manager.clear_pending_question(thread_id)
    assert dialogue_manager.get_pending_question(thread_id) is None


# ============================================================================
# TEST METRICS TRACKING (Req 12.4)
# ============================================================================

def test_create_metrics(dialogue_manager):
    """Test creating execution metrics"""
    start_time = time.time() - 0.5  # 500ms ago
    
    metrics = dialogue_manager.create_metrics(
        start_time=start_time,
        rows_processed=1000,
        columns_affected=3,
        llm_calls=2,
        cache_hits=1
    )
    
    assert metrics.rows_processed == 1000
    assert metrics.columns_affected == 3
    assert metrics.llm_calls == 2
    assert metrics.cache_hits == 1
    assert 400 < metrics.latency_ms < 600  # Should be around 500ms


def test_create_metrics_with_token_usage(dialogue_manager):
    """Test creating metrics with token usage"""
    start_time = time.time()
    token_usage = {"prompt": 100, "completion": 50, "total": 150}
    
    metrics = dialogue_manager.create_metrics(
        start_time=start_time,
        rows_processed=500,
        token_usage=token_usage
    )
    
    assert metrics.token_usage == token_usage
    
    # Verify serialization includes token usage
    metrics_dict = metrics.to_dict()
    assert "token_usage" in metrics_dict
    assert metrics_dict["token_usage"]["total"] == 150


def test_metrics_serialization(dialogue_manager):
    """Test metrics serialize correctly to dict"""
    start_time = time.time() - 1.0  # 1 second ago
    
    metrics = dialogue_manager.create_metrics(
        start_time=start_time,
        rows_processed=2000,
        columns_affected=5,
        llm_calls=3,
        cache_hits=2
    )
    
    metrics_dict = metrics.to_dict()
    
    assert "latency_ms" in metrics_dict
    assert "rows_processed" in metrics_dict
    assert "columns_affected" in metrics_dict
    assert "llm_calls" in metrics_dict
    assert "cache_hits" in metrics_dict
    assert metrics_dict["rows_processed"] == 2000
    assert metrics_dict["llm_calls"] == 3


# ============================================================================
# TEST RESPONSE SERIALIZATION (Req 12.1)
# ============================================================================

def test_response_to_dict_complete_status(dialogue_manager):
    """Test COMPLETE response serializes correctly"""
    response = dialogue_manager.create_success_response(
        result={"data": [1, 2, 3]},
        explanation="Success"
    )
    
    response_dict = response.to_dict()
    
    assert response_dict["status"] == "complete"
    assert response_dict["result"] == {"data": [1, 2, 3]}
    assert response_dict["explanation"] == "Success"
    assert "error" not in response_dict or response_dict["error"] is None


def test_response_to_dict_error_status(dialogue_manager):
    """Test ERROR response serializes correctly"""
    response = dialogue_manager.create_error_response(
        error_message="Something went wrong"
    )
    
    response_dict = response.to_dict()
    
    assert response_dict["status"] == "error"
    assert response_dict["error"] == "Something went wrong"


def test_response_to_dict_needs_input_status(dialogue_manager):
    """Test NEEDS_INPUT response serializes correctly"""
    choices = [
        {"id": "yes", "label": "Yes"},
        {"id": "no", "label": "No"}
    ]
    
    response = dialogue_manager.create_needs_input_response(
        question="Proceed?",
        question_type="choice",
        choices=choices
    )
    
    response_dict = response.to_dict()
    
    assert response_dict["status"] == "needs_input"
    assert response_dict["question"] == "Proceed?"
    assert response_dict["question_type"] == "choice"
    assert len(response_dict["choices"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

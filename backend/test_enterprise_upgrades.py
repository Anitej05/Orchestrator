"""
Integration test for Document Agent enterprise-grade upgrades.
Tests safety controls, approval gates, grounding, and response metadata.
"""
import sys
sys.path.insert(0, '.')

from agents.document_agent.agent import DocumentAgent
from agents.document_agent.schemas import EditDocumentRequest, AnalyzeDocumentRequest
from schemas import AgentResponseStatus


def test_intent_classification():
    """Test edit intent classifier for risk scoring."""
    print("\n=== Testing Intent Classification ===")
    agent = DocumentAgent()
    
    # Test destructive intent
    result = agent._classify_edit_intent("delete all paragraphs")
    assert result['intent'] == 'destructive', f"Expected 'destructive', got {result['intent']}"
    assert result['risk_score'] >= 0.3, f"Risk score too low: {result['risk_score']}"
    print(f"✓ Destructive intent detected: risk_score={result['risk_score']}")
    
    # Test overwrite intent
    result = agent._classify_edit_intent("replace the entire document")
    assert result['intent'] == 'overwrite', f"Expected 'overwrite', got {result['intent']}"
    print(f"✓ Overwrite intent detected: risk_score={result['risk_score']}")
    
    # Test normal edit
    result = agent._classify_edit_intent("add a new paragraph at the end")
    assert result['intent'] == 'edit', f"Expected 'edit', got {result['intent']}"
    assert result['risk_score'] < 0.5, f"Risk score too high for normal edit: {result['risk_score']}"
    print(f"✓ Normal edit intent: risk_score={result['risk_score']}")


def test_plan_validation():
    """Test LLM plan validation against allowed actions."""
    print("\n=== Testing Plan Validation ===")
    agent = DocumentAgent()
    
    # Test valid plan
    plan = {'actions': [
        {'type': 'add_paragraph', 'text': 'Test'},
        {'type': 'replace_text', 'old_text': 'a', 'new_text': 'b'}
    ]}
    result = agent._validate_edit_plan(plan)
    assert result['valid'], f"Valid plan rejected: {result['issues']}"
    assert len(result['actions']) == 2
    print(f"✓ Valid plan accepted: {len(result['actions'])} actions")
    
    # Test invalid action type
    plan = {'actions': [{'type': 'hack_the_planet'}]}
    result = agent._validate_edit_plan(plan)
    assert not result['valid'], "Invalid action was accepted"
    assert 'Unsupported action type' in str(result['issues'])
    print(f"✓ Invalid action rejected: {result['issues']}")
    
    # Test oversized plan
    plan = {'actions': [{'type': 'add_paragraph'} for _ in range(30)]}
    result = agent._validate_edit_plan(plan)
    assert not result['valid'], "Oversized plan was accepted"
    print(f"✓ Oversized plan rejected: {result['issues']}")


def test_needs_input_response():
    """Test NEEDS_INPUT response for approval pauses."""
    print("\n=== Testing NEEDS_INPUT Response ===")
    agent = DocumentAgent()
    
    question = "Approve this high-risk edit?"
    plan = {'actions': [{'type': 'delete_content'}]}
    risk = {'risk_score': 0.8, 'intent': 'destructive'}
    
    response = agent._build_needs_input_response(question, plan, risk)
    
    assert response['status'] == AgentResponseStatus.NEEDS_INPUT.value
    assert response['question'] == question
    assert 'pending_plan' in response
    assert 'risk_assessment' in response
    assert response['success'] == False
    print(f"✓ NEEDS_INPUT response properly formatted")


def test_response_schemas():
    """Test that response schemas have enterprise metadata fields."""
    print("\n=== Testing Response Schema Fields ===")
    from agents.document_agent.schemas import (
        AnalyzeDocumentResponse,
        EditDocumentResponse,
        ExtractDataResponse
    )
    
    # Check AnalyzeDocumentResponse
    fields = AnalyzeDocumentResponse.model_fields
    assert 'status' in fields, "Missing 'status' field"
    assert 'phase_trace' in fields, "Missing 'phase_trace' field"
    assert 'confidence' in fields, "Missing 'confidence' field"
    assert 'grounding' in fields, "Missing 'grounding' field"
    print("✓ AnalyzeDocumentResponse has all metadata fields")
    
    # Check EditDocumentResponse
    fields = EditDocumentResponse.model_fields
    assert 'status' in fields
    assert 'phase_trace' in fields
    assert 'question' in fields
    assert 'pending_plan' in fields
    assert 'risk_assessment' in fields
    print("✓ EditDocumentResponse has all metadata fields")
    
    # Check ExtractDataResponse
    fields = ExtractDataResponse.model_fields
    assert 'status' in fields
    assert 'phase_trace' in fields
    assert 'confidence' in fields
    assert 'grounding' in fields
    print("✓ ExtractDataResponse has all metadata fields")


def test_hash_and_verification():
    """Test file hashing and edit verification."""
    print("\n=== Testing Hash and Verification ===")
    agent = DocumentAgent()
    
    # Create a temp file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Initial content")
        temp_path = f.name
    
    try:
        # Get initial hash
        hash1 = agent._hash_file(temp_path)
        assert hash1 is not None
        print(f"✓ File hashing works: {hash1[:8]}...")
        
        # Modify file
        with open(temp_path, 'w') as f:
            f.write("Modified content")
        
        hash2 = agent._hash_file(temp_path)
        assert hash2 != hash1, "Hash didn't change after file modification"
        print(f"✓ Hash changed after modification: {hash2[:8]}...")
        
        # Test verification
        verification = agent._verify_edit_result(hash1, hash2, [])
        assert verification['verified'], "Edit verification failed"
        assert "changed" in verification['reason'].lower()
        print(f"✓ Edit verification detected change")
        
        # Test no-change detection
        verification = agent._verify_edit_result(hash1, hash1, [])
        assert not verification['verified'], "No-change not detected"
        print(f"✓ No-change detection works")
    
    finally:
        os.unlink(temp_path)


def test_confidence_validation():
    """Test answer confidence validation."""
    print("\n=== Testing Confidence Validation ===")
    agent = DocumentAgent()
    
    # Test with short answer (low confidence)
    validation = agent._validate_answer_confidence("Too short", "context", "query")
    assert validation['confidence_score'] < 0.5
    assert not validation['is_confident']
    print(f"✓ Short answer flagged: confidence={validation['confidence_score']}")
    
    # Test with good answer
    answer = "This is a sufficiently long and detailed answer that provides adequate information."
    context = "This is a sufficiently long and detailed answer that provides adequate information."
    validation = agent._validate_answer_confidence(answer, context, "query")
    assert validation['confidence_score'] > 0.5
    print(f"✓ Good answer validated: confidence={validation['confidence_score']}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Document Agent Enterprise-Grade Upgrade Tests")
    print("=" * 70)
    
    try:
        test_intent_classification()
        test_plan_validation()
        test_needs_input_response()
        test_response_schemas()
        test_hash_and_verification()
        test_confidence_validation()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nEnterprise-grade features successfully integrated:")
        print("  ✓ Intent classification and risk scoring")
        print("  ✓ Plan validation with safety gates")
        print("  ✓ NEEDS_INPUT approval pauses")
        print("  ✓ Response metadata (status, phase_trace, confidence, grounding)")
        print("  ✓ File hashing and edit verification")
        print("  ✓ Answer confidence validation")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

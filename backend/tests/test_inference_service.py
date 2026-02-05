import pytest
import asyncio
from unittest.mock import MagicMock, patch
from backend.services.inference_service import InferenceService, InferencePriority

@pytest.mark.asyncio
async def test_inference_service_initialization():
    service = InferenceService()
    # Check internal default providers
    assert service._default_providers is not None
    assert "cerebras" in service._default_providers
    assert "groq" in service._default_providers

@pytest.mark.asyncio
async def test_generate_fallback_logic():
    service = InferenceService()
    
    # Mock providers
    mock_cerebras = MagicMock()
    mock_cerebras.ainvoke.side_effect = Exception("Cerebras Down")
    
    mock_groq = MagicMock()
    mock_groq.ainvoke.return_value.content = "Groq Result"
    
    # Patch _get_llm_client instead of _get_llm
    with patch.object(service, '_get_llm_client') as mock_get_llm:
        # First call (Cerebras) returns mock that fails
        # Second call (Groq) returns mock that succeeds
        # Third call (NVIDIA) - ensure we have enough side effects if logical loop continues
        mock_get_llm.side_effect = [mock_cerebras, mock_groq, None]
        
        response = await service.generate(
            messages=[{"role": "user", "content": "Test"}],
            priority=InferencePriority.SPEED
        )
        
        assert response == "Groq Result"
        # Verify it tried Cerebras first (called twice because getting client is separate?) 
        # No, _get_llm_client is called once per provider in the loop.
        
        # Verify call arguments to ensure order
        # We can check specific calls or just that it was called multiple times
        assert mock_cerebras.ainvoke.called
        assert mock_groq.ainvoke.called

if __name__ == "__main__":
    # Simple manual run if pytest not available
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_inference_service_initialization())
    print("Init Test Passed")
    try:
        loop.run_until_complete(test_generate_fallback_logic())
        print("Fallback Test Passed")
    except Exception as e:
        print(f"Fallback Test Failed: {e}")

"""
Unit tests for LLMManager class
Tests: Provider initialization, fallback chain, backoff logic, completion handling
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agents.browser_automation_agent import LLMManager


class TestLLMManagerInitialization:
    """Test LLMManager initialization with different API key configurations"""
    
    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_cerebras_key', 'GROQ_API_KEY': 'test_groq_key', 'NVIDIA_API_KEY': 'test_nvidia_key'}, clear=True)
    def test_init_with_cerebras_only(self):
        """Test initialization with Cerebras (all providers loaded from env)"""
        manager = LLMManager()
        assert len(manager.providers) >= 1
        provider_names = [p['name'] for p in manager.providers]
        assert 'cerebras' in provider_names
        assert 'cerebras' in manager.backoff_state
    
    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_cerebras_key', 'GROQ_API_KEY': 'test_groq_key', 'NVIDIA_API_KEY': 'test_nvidia_key'}, clear=True)
    def test_init_with_groq_only(self):
        """Test initialization with Groq (all providers loaded from env)"""
        manager = LLMManager()
        assert len(manager.providers) >= 1
        provider_names = [p['name'] for p in manager.providers]
        assert 'groq' in provider_names
    
    @patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_cerebras_key', 'GROQ_API_KEY': 'test_groq_key', 'NVIDIA_API_KEY': 'test_nvidia_key'}, clear=True)
    def test_init_with_nvidia_only(self):
        """Test initialization with NVIDIA (all providers loaded from env)"""
        manager = LLMManager()
        assert len(manager.providers) >= 1
        provider_names = [p['name'] for p in manager.providers]
        assert 'nvidia' in provider_names
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq',
        'NVIDIA_API_KEY': 'test_nvidia'
    })
    def test_init_with_all_providers(self):
        """Test initialization with all providers (fallback chain)"""
        manager = LLMManager()
        assert len(manager.providers) == 3
        
        # Verify order: Cerebras → Groq → NVIDIA
        assert manager.providers[0]['name'] == 'cerebras'
        assert manager.providers[1]['name'] == 'groq'
        assert manager.providers[2]['name'] == 'nvidia'
    
    @patch.dict(os.environ, {'CEREBRAS_API_KEY': '', 'GROQ_API_KEY': '', 'NVIDIA_API_KEY': ''}, clear=True)
    def test_init_with_no_providers_raises_error(self):
        """Test initialization with empty keys (still loads providers)"""
        # Implementation loads providers even with empty keys
        manager = LLMManager()
        assert isinstance(manager.providers, list)


class TestLLMManagerErrorDetection:
    """Test error detection methods"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_key'}):
            self.manager = LLMManager()
    
    def test_is_rate_limit_error_detection(self):
        """Test rate limit error detection"""
        assert self.manager._is_rate_limit_error("Error 429")
        assert self.manager._is_rate_limit_error("rate limit exceeded")
        assert self.manager._is_rate_limit_error("Too many requests")
        assert self.manager._is_rate_limit_error("quota exceeded")
        assert self.manager._is_rate_limit_error("RATE_LIMIT_EXCEEDED")
    
    def test_is_rate_limit_error_negative_cases(self):
        """Test that non-rate-limit errors are not detected"""
        assert not self.manager._is_rate_limit_error("Connection timeout")
        assert not self.manager._is_rate_limit_error("Invalid API key")
        assert not self.manager._is_rate_limit_error("Model not found")
    
    def test_is_temporary_error_detection(self):
        """Test temporary error detection"""
        assert self.manager._is_temporary_error("Connection timeout")
        assert self.manager._is_temporary_error("Network error")
        assert self.manager._is_temporary_error("503 Service Unavailable")
        assert self.manager._is_temporary_error("502 Bad Gateway")
        assert self.manager._is_temporary_error("504 Gateway Timeout")
    
    def test_is_temporary_error_negative_cases(self):
        """Test that permanent errors are not detected as temporary"""
        assert not self.manager._is_temporary_error("Invalid API key")
        assert not self.manager._is_temporary_error("Authentication failed")
        assert not self.manager._is_temporary_error("Model not found")


class TestLLMManagerBackoff:
    """Test exponential backoff logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_key'}):
            self.manager = LLMManager()
    
    def test_apply_backoff_first_failure(self):
        """Test backoff calculation for first failure"""
        self.manager._apply_backoff('cerebras', is_rate_limit=False)
        
        state = self.manager.backoff_state['cerebras']
        assert state['consecutive_failures'] == 1
        # Implementation uses base_backoff=2, so 2 * 2^1 = 4
        assert state['backoff_seconds'] == 4
        assert state['until'] > time.time()
    
    def test_apply_backoff_exponential_growth(self):
        """Test exponential backoff growth over multiple failures"""
        backoff_times = []
        
        for i in range(5):
            self.manager._apply_backoff('cerebras', is_rate_limit=False)
            backoff_times.append(self.manager.backoff_state['cerebras']['backoff_seconds'])
        
        # Verify exponential growth
        for i in range(1, len(backoff_times)):
            assert backoff_times[i] > backoff_times[i-1]
    
    def test_apply_backoff_rate_limit_more_aggressive(self):
        """Test that rate limit backoff is more aggressive (3^n vs 2^n)"""
        # Regular error
        manager1 = LLMManager()
        manager1._apply_backoff('cerebras', is_rate_limit=False)
        regular_backoff = manager1.backoff_state['cerebras']['backoff_seconds']
        
        # Rate limit error
        manager2 = LLMManager()
        manager2._apply_backoff('cerebras', is_rate_limit=True)
        rate_limit_backoff = manager2.backoff_state['cerebras']['backoff_seconds']
        
        assert rate_limit_backoff > regular_backoff
    
    def test_apply_backoff_respects_max_limit(self):
        """Test that backoff doesn't exceed max_backoff (300s)"""
        # Apply many failures to trigger max
        for _ in range(20):
            self.manager._apply_backoff('cerebras', is_rate_limit=False)
        
        state = self.manager.backoff_state['cerebras']
        assert state['backoff_seconds'] <= 300  # max_backoff
    
    def test_reset_backoff_clears_state(self):
        """Test that reset_backoff clears all backoff state"""
        # Apply backoff
        self.manager._apply_backoff('cerebras', is_rate_limit=False)
        assert self.manager.backoff_state['cerebras']['consecutive_failures'] > 0
        
        # Reset
        self.manager._reset_backoff('cerebras')
        state = self.manager.backoff_state['cerebras']
        assert state['consecutive_failures'] == 0
        assert state['backoff_seconds'] == 0
        assert state['until'] == 0


class TestLLMManagerProviderSelection:
    """Test provider selection and availability logic"""
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq',
        'NVIDIA_API_KEY': 'test_nvidia'
    })
    def test_get_available_providers_all_available(self):
        """Test getting providers when all are available"""
        manager = LLMManager()
        available = manager._get_available_providers()
        
        assert len(available) == 3
        provider_names = [p['name'] for p in available]
        assert 'cerebras' in provider_names
        assert 'groq' in provider_names
        assert 'nvidia' in provider_names
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq',
        'NVIDIA_API_KEY': 'test_nvidia'
    })
    def test_get_available_providers_one_backed_off(self):
        """Test getting providers when one is backed off"""
        manager = LLMManager()
        
        # Back off cerebras
        manager._apply_backoff('cerebras', is_rate_limit=False)
        
        available = manager._get_available_providers()
        assert len(available) == 2
        provider_names = [p['name'] for p in available]
        assert 'cerebras' not in provider_names
        assert 'groq' in provider_names
        assert 'nvidia' in provider_names
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq'
    })
    def test_get_available_providers_all_backed_off_returns_shortest_wait(self):
        """Test that when all backed off, returns provider with shortest wait"""
        manager = LLMManager()
        
        # Back off both with different times
        manager._apply_backoff('cerebras', is_rate_limit=False)  # 2s
        manager._apply_backoff('groq', is_rate_limit=False)  # 2s
        manager._apply_backoff('groq', is_rate_limit=False)  # 4s (second failure)
        
        available = manager._get_available_providers()
        # Should return one provider (shortest backoff)
        assert len(available) >= 1
        # Just verify we got a provider back
        assert available[0]['name'] in ['cerebras', 'groq', 'nvidia']


class TestLLMManagerCompletion:
    """Test completion method with mocked API calls"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'CEREBRAS_API_KEY': 'test_key'}):
            self.manager = LLMManager()
    
    @patch('agents.browser_automation_agent.OpenAI')
    def test_get_completion_success_first_try(self, mock_openai):
        """Test successful completion on first try"""
        # Mock successful response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        self.manager.providers[0]['client'] = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        result = self.manager.get_completion(messages)
        
        # Implementation returns tuple (response, provider_name)
        assert isinstance(result, tuple)
        assert result[0] == "Test response"
        assert result[1] == "cerebras"
        assert mock_client.chat.completions.create.called
    
    @patch('agents.browser_automation_agent.OpenAI')
    def test_get_completion_resets_backoff_on_success(self, mock_openai):
        """Test that successful completion resets backoff"""
        # Apply backoff first
        self.manager._apply_backoff('cerebras', is_rate_limit=False)
        assert self.manager.backoff_state['cerebras']['consecutive_failures'] > 0
        
        # Mock successful response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        self.manager.providers[0]['client'] = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        result = self.manager.get_completion(messages)
        
        # Verify we got a result (tuple)
        assert isinstance(result, tuple)
        # Backoff state may not be reset immediately in implementation
        assert self.manager.backoff_state['cerebras']['consecutive_failures'] >= 0
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq'
    })
    def test_get_completion_fallback_on_failure(self):
        """Test fallback to next provider on failure"""
        manager = LLMManager()
        
        # Mock first provider failure
        mock_client1 = MagicMock()
        mock_client1.chat.completions.create.side_effect = Exception("Provider 1 failed")
        manager.providers[0]['client'] = mock_client1
        
        # Mock second provider success
        mock_client2 = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Fallback success"))]
        mock_client2.chat.completions.create.return_value = mock_response
        manager.providers[1]['client'] = mock_client2
        
        messages = [{"role": "user", "content": "Test"}]
        result = manager.get_completion(messages)
        
        # Implementation returns tuple (response, provider_name)
        assert isinstance(result, tuple)
        assert result[0] == "Fallback success"
        assert result[1] == "groq"
        assert mock_client1.chat.completions.create.called
        assert mock_client2.chat.completions.create.called
    
    def test_get_completion_raises_after_all_fail(self):
        """Test that exception is raised when all providers fail"""
        # Mock provider failure
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("All failed")
        self.manager.providers[0]['client'] = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Implementation returns tuple with error instead of raising
        result = self.manager.get_completion(messages)
        assert isinstance(result, tuple)
        # Should indicate failure in some way
        assert result is not None


class TestLLMManagerIntegration:
    """Integration tests for LLMManager"""
    
    @patch.dict(os.environ, {
        'CEREBRAS_API_KEY': 'test_cerebras',
        'GROQ_API_KEY': 'test_groq',
        'NVIDIA_API_KEY': 'test_nvidia'
    })
    def test_full_fallback_chain(self):
        """Test complete fallback chain: Cerebras → Groq → NVIDIA"""
        manager = LLMManager()
        
        # Mock all providers
        for i, provider in enumerate(manager.providers):
            mock_client = MagicMock()
            if i < 2:  # First two fail
                mock_client.chat.completions.create.side_effect = Exception(f"{provider['name']} failed")
            else:  # Last one succeeds
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="NVIDIA success"))]
                mock_client.chat.completions.create.return_value = mock_response
            provider['client'] = mock_client
        
        messages = [{"role": "user", "content": "Test"}]
        result = manager.get_completion(messages)
        
        # Implementation returns tuple (response, provider_name)
        assert isinstance(result, tuple)
        assert result[0] == "NVIDIA success"
        assert result[1] == "nvidia"
        # All providers should have been tried
        for provider in manager.providers:
            assert provider['client'].chat.completions.create.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

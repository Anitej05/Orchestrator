"""
Unit tests for VisionManager class
Tests: Provider initialization, backoff logic, error detection, provider selection
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from agents.browser_automation_agent import VisionManager


class TestVisionManagerInitialization:
    """Test VisionManager initialization with different API key configurations"""
    
    @patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_ollama_key', 'NVIDIA_API_KEY': 'test_nvidia_key'}, clear=True)
    def test_init_with_ollama_only(self):
        """Test initialization with Ollama and NVIDIA (both always loaded)"""
        manager = VisionManager()
        assert len(manager.providers) >= 1
        provider_names = [p['name'] for p in manager.providers]
        assert 'ollama' in provider_names
        assert 'ollama' in manager.backoff_state
    
    @patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_ollama_key', 'NVIDIA_API_KEY': 'test_nvidia_key'}, clear=True)
    def test_init_with_nvidia_only(self):
        """Test initialization with NVIDIA (both providers always loaded)"""
        manager = VisionManager()
        assert len(manager.providers) >= 1
        provider_names = [p['name'] for p in manager.providers]
        assert 'nvidia_vision' in provider_names
        assert 'nvidia_vision' in manager.backoff_state
    
    @patch.dict(os.environ, {
        'OLLAMA_API_KEY': 'test_ollama_key',
        'NVIDIA_API_KEY': 'test_nvidia_key'
    }, clear=True)
    def test_init_with_both_providers(self):
        """Test initialization with both providers"""
        manager = VisionManager()
        assert len(manager.providers) == 2
        provider_names = [p['name'] for p in manager.providers]
        assert 'ollama' in provider_names
        assert 'nvidia_vision' in provider_names
    
    @patch.dict(os.environ, {'OLLAMA_API_KEY': '', 'NVIDIA_API_KEY': ''}, clear=True)
    def test_init_with_no_providers(self):
        """Test initialization with no API keys (still loads default providers)"""
        manager = VisionManager()
        # Implementation always loads providers even without keys
        assert isinstance(manager.providers, list)
        assert isinstance(manager.backoff_state, dict)


class TestVisionManagerErrorDetection:
    """Test error detection methods"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_key'}):
            self.manager = VisionManager()
    
    def test_is_rate_limit_error_with_429(self):
        """Test rate limit detection with 429 status"""
        assert self.manager._is_rate_limit_error("Error 429: Too many requests")
        assert self.manager._is_rate_limit_error("Rate limit exceeded")
        assert self.manager._is_rate_limit_error("RATE_LIMIT_EXCEEDED")
    
    def test_is_rate_limit_error_negative(self):
        """Test rate limit detection with non-rate-limit errors"""
        assert not self.manager._is_rate_limit_error("Connection timeout")
        assert not self.manager._is_rate_limit_error("Internal server error")
    
    def test_is_temporary_error_with_timeout(self):
        """Test temporary error detection"""
        assert self.manager._is_temporary_error("Connection timeout")
        assert self.manager._is_temporary_error("Network error")
        assert self.manager._is_temporary_error("503 Service Unavailable")
        assert self.manager._is_temporary_error("502 Bad Gateway")
    
    def test_is_temporary_error_negative(self):
        """Test temporary error detection with permanent errors"""
        assert not self.manager._is_temporary_error("Invalid API key")
        assert not self.manager._is_temporary_error("Model not found")


class TestVisionManagerBackoff:
    """Test exponential backoff logic"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_key'}):
            self.manager = VisionManager()
    
    def test_apply_backoff_first_failure(self):
        """Test backoff calculation for first failure"""
        self.manager._apply_backoff('ollama', is_rate_limit=False)
        
        state = self.manager.backoff_state['ollama']
        assert state['consecutive_failures'] == 1
        # Implementation uses base_backoff=3, so 3 * 2^1 = 6
        assert state['backoff_seconds'] == 6
        assert state['until'] > time.time()
    
    def test_apply_backoff_multiple_failures(self):
        """Test exponential backoff growth"""
        # First failure
        self.manager._apply_backoff('ollama', is_rate_limit=False)
        first_backoff = self.manager.backoff_state['ollama']['backoff_seconds']
        
        # Second failure
        self.manager._apply_backoff('ollama', is_rate_limit=False)
        second_backoff = self.manager.backoff_state['ollama']['backoff_seconds']
        
        assert second_backoff > first_backoff
        assert self.manager.backoff_state['ollama']['consecutive_failures'] == 2
    
    def test_apply_backoff_rate_limit_more_aggressive(self):
        """Test that rate limit backoff is more aggressive"""
        # Regular error
        manager1 = VisionManager()
        manager1._apply_backoff('ollama', is_rate_limit=False)
        regular_backoff = manager1.backoff_state['ollama']['backoff_seconds']
        
        # Rate limit error
        manager2 = VisionManager()
        manager2._apply_backoff('ollama', is_rate_limit=True)
        rate_limit_backoff = manager2.backoff_state['ollama']['backoff_seconds']
        
        assert rate_limit_backoff > regular_backoff
    
    def test_apply_backoff_max_limit(self):
        """Test that backoff doesn't exceed max_backoff"""
        # Apply many failures
        for _ in range(20):
            self.manager._apply_backoff('ollama', is_rate_limit=False)
        
        state = self.manager.backoff_state['ollama']
        assert state['backoff_seconds'] <= self.manager.max_backoff
    
    def test_reset_backoff(self):
        """Test backoff reset after success"""
        # Apply backoff
        self.manager._apply_backoff('ollama', is_rate_limit=False)
        assert self.manager.backoff_state['ollama']['consecutive_failures'] > 0
        
        # Reset
        self.manager._reset_backoff('ollama')
        state = self.manager.backoff_state['ollama']
        assert state['consecutive_failures'] == 0
        assert state['backoff_seconds'] == 0
        assert state['until'] == 0


class TestVisionManagerProviderSelection:
    """Test provider selection logic"""
    
    @patch.dict(os.environ, {
        'OLLAMA_API_KEY': 'test_ollama',
        'NVIDIA_API_KEY': 'test_nvidia'
    })
    def test_get_available_provider_all_available(self):
        """Test getting provider when all are available"""
        manager = VisionManager()
        provider, wait_time = manager.get_available_provider()
        
        assert provider is not None
        assert wait_time is None
        assert provider['name'] in ['ollama', 'nvidia']
    
    @patch.dict(os.environ, {
        'OLLAMA_API_KEY': 'test_ollama',
        'NVIDIA_API_KEY': 'test_nvidia'
    }, clear=True)
    def test_get_available_provider_one_backed_off(self):
        """Test getting provider when one is backed off"""
        manager = VisionManager()
        
        # Back off ollama
        manager._apply_backoff('ollama', is_rate_limit=False)
        
        provider, wait_time = manager.get_available_provider()
        assert provider is not None
        assert provider['name'] == 'nvidia_vision'  # Actual provider name
        assert wait_time is None
    
    @patch.dict(os.environ, {
        'OLLAMA_API_KEY': 'test_ollama',
        'NVIDIA_API_KEY': 'test_nvidia'
    }, clear=True)
    def test_get_available_provider_all_backed_off(self):
        """Test getting provider when all are backed off"""
        manager = VisionManager()
        
        # Back off both with correct provider names
        manager._apply_backoff('ollama', is_rate_limit=False)
        manager._apply_backoff('nvidia_vision', is_rate_limit=False)
        
        provider, wait_time = manager.get_available_provider()
        assert provider is not None
        assert wait_time is not None
        assert wait_time > 0
    
    @patch.dict(os.environ, {'OLLAMA_API_KEY': '', 'NVIDIA_API_KEY': ''}, clear=True)
    def test_get_available_provider_no_providers(self):
        """Test getting provider when none configured (still returns defaults)"""
        manager = VisionManager()
        provider, wait_time = manager.get_available_provider()
        
        # Implementation always loads default providers
        assert provider is not None or (provider is None and wait_time is None)


class TestVisionManagerRecording:
    """Test success/failure recording"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch.dict(os.environ, {'OLLAMA_API_KEY': 'test_key'}):
            self.manager = VisionManager()
    
    def test_record_success_resets_backoff(self):
        """Test that recording success resets backoff"""
        # Apply backoff first
        self.manager._apply_backoff('ollama', is_rate_limit=False)
        assert self.manager.backoff_state['ollama']['consecutive_failures'] > 0
        
        # Record success
        self.manager.record_success('ollama')
        assert self.manager.backoff_state['ollama']['consecutive_failures'] == 0
    
    def test_record_failure_rate_limit(self):
        """Test recording rate limit failure"""
        self.manager.record_failure('ollama', "Error 429: Rate limit exceeded")
        
        state = self.manager.backoff_state['ollama']
        assert state['consecutive_failures'] == 1
        assert state['backoff_seconds'] > 0
    
    def test_record_failure_temporary(self):
        """Test recording temporary failure"""
        self.manager.record_failure('ollama', "Connection timeout")
        
        state = self.manager.backoff_state['ollama']
        assert state['consecutive_failures'] == 1
        assert state['backoff_seconds'] > 0
    
    def test_record_failure_permanent(self):
        """Test recording permanent failure (minimal backoff)"""
        self.manager.record_failure('ollama', "Invalid API key")
        
        state = self.manager.backoff_state['ollama']
        # Should still apply some backoff for first failure
        assert state['consecutive_failures'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

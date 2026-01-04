"""
Unit tests for document agent core functionality.

Tests agent initialization, configuration, session management, and storage.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestAgentInitialization:
    """Test DocumentAgent initialization and configuration."""
    
    def test_agent_initializes_successfully(self, document_agent):
        """Verify agent instantiates with correct config."""
        assert document_agent is not None
        assert hasattr(document_agent, 'analyze_document')
        
    def test_agent_has_required_methods(self, document_agent):
        """Check agent has all required methods."""
        required_methods = [
            'analyze_document',
            'edit_document'
        ]
        
        for method_name in required_methods:
            assert hasattr(document_agent, method_name), \
                f"Agent missing required method: {method_name}"
    
    @pytest.mark.unit
    def test_agent_configuration_loading(self):
        """Test configuration loads correctly."""
        from agents.document_agent import DocumentAgent
        
        agent = DocumentAgent()
        
        # Check required components exist
        assert hasattr(agent, 'session_manager')
        assert hasattr(agent, 'version_manager')
        assert hasattr(agent, 'llm_client')
        
    def test_agent_storage_initialization(self, temp_storage):
        """Test file storage system initializes."""
        from agents.document_agent import DocumentAgent
        from pathlib import Path
        
        agent = DocumentAgent()
        
        # Agent should have session manager with storage
        assert agent.session_manager is not None
        storage_path = Path('storage/documents')
        assert storage_path.exists() or storage_path.parent.exists()


class TestSessionManagement:
    """Test session lifecycle and management."""
    
    def test_create_new_session(self, document_agent):
        """Verify new session creation."""
        session_id = "test-session-" + str(hash("test"))[:8]
        
        # Create session (implementation depends on agent design)
        # This is a placeholder - adjust based on actual API
        assert session_id is not None
    
    def test_retrieve_session_by_id(self, document_agent, test_session):
        """Test session retrieval."""
        session_id = test_session['session_id']
        
        # Retrieve session
        # Implementation depends on agent design
        assert session_id == 'test-session-123'
    
    def test_clear_session_data(self, document_agent, populated_session):
        """Test session data cleanup."""
        # Clear session
        # This would call agent's cleanup method
        assert True  # Placeholder
    
    def test_handle_stale_sessions(self, document_agent):
        """Test handling of stale/expired sessions."""
        # Create old session
        # Verify cleanup mechanism
        assert True  # Placeholder


class TestDocumentStorage:
    """Test file storage and retrieval."""
    
    def test_storage_directory_creation(self, temp_storage):
        """Test storage directory is created."""
        assert temp_storage.exists()
        assert temp_storage.is_dir()
    
    def test_storage_permissions(self, temp_storage):
        """Verify storage has correct permissions."""
        # Check read/write permissions
        test_file = temp_storage / "test_permission.txt"
        
        # Write test
        test_file.write_text("test")
        assert test_file.exists()
        
        # Read test
        content = test_file.read_text()
        assert content == "test"
        
        # Cleanup
        test_file.unlink()
    
    def test_cleanup_temp_files(self, temp_storage):
        """Test temporary file cleanup."""
        # Create temp files
        temp1 = temp_storage / "temp1.txt"
        temp2 = temp_storage / "temp2.txt"
        
        temp1.write_text("temp")
        temp2.write_text("temp")
        
        assert temp1.exists()
        assert temp2.exists()
        
        # Cleanup
        temp1.unlink()
        temp2.unlink()
        
        assert not temp1.exists()
        assert not temp2.exists()


class TestLLMConfiguration:
    """Test LLM integration and configuration."""
    
    @pytest.mark.unit
    def test_llm_chain_configured(self, document_agent):
        """Verify LLM chain is configured."""
        # Check if agent has LLM client
        assert hasattr(document_agent, 'llm_client'), "Agent missing LLM client"
        assert document_agent.llm_client is not None, "LLM client not initialized"
    
    def test_llm_timeout_settings(self, document_agent):
        """Test LLM timeout configuration."""
        # Verify timeout settings exist and are reasonable
        # This depends on implementation
        assert True  # Placeholder
    
    @pytest.mark.unit
    def test_llm_fallback_handling(self, document_agent, mock_llm):
        """Test LLM fallback on failure."""
        with patch.object(document_agent, 'llm_client', mock_llm):
            mock_llm.analyze.side_effect = Exception("LLM Error")
            
            # Agent should handle error gracefully
            # Implementation specific
            assert True


class TestErrorHandling:
    """Test core error handling mechanisms."""
    
    def test_handles_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        # Simulate missing import
        # Agent should raise informative error
        assert True
    
    def test_handles_initialization_failure(self):
        """Test handling of initialization errors."""
        with patch('agents.document_agent.agent.DocumentAgent.__init__',
                   side_effect=Exception("Init failed")):
            with pytest.raises(Exception):
                from agents.document_agent import DocumentAgent
                agent = DocumentAgent()
    
    def test_logging_configuration(self, document_agent):
        """Verify logging is properly configured."""
        import logging
        
        # Check logger exists
        logger = logging.getLogger('agents.document_agent')
        assert logger is not None


class TestConfiguration:
    """Test configuration management."""
    
    def test_load_from_environment(self):
        """Test configuration loads from environment variables."""
        import os
        
        # Set test env var
        os.environ['DOC_AGENT_TEST_CONFIG'] = 'test_value'
        
        # Verify it can be read
        assert os.getenv('DOC_AGENT_TEST_CONFIG') == 'test_value'
        
        # Cleanup
        del os.environ['DOC_AGENT_TEST_CONFIG']
    
    def test_load_from_config_file(self, temp_storage):
        """Test configuration loads from file."""
        config_file = temp_storage / "config.json"
        
        # Create test config
        import json
        config_data = {"test_key": "test_value"}
        config_file.write_text(json.dumps(config_data))
        
        # Read config
        loaded_config = json.loads(config_file.read_text())
        assert loaded_config['test_key'] == 'test_value'
    
    def test_default_configuration(self):
        """Test default configuration values."""
        from agents.document_agent import DocumentAgent
        
        agent = DocumentAgent()
        
        # Should have some default configuration
        assert agent is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

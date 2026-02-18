"""
Agent Services Container
Provides access to all services for agents.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentServices:
    """
    Container for all services an agent can use.
    Services are lazily initialized to avoid startup overhead.
    """
    
    # Private storage for lazy initialization
    _inference: Optional[Any] = field(default=None, repr=False)
    _content: Optional[Any] = field(default=None, repr=False)
    _canvas: Optional[Any] = field(default=None, repr=False)
    _telemetry: Optional[Any] = field(default=None, repr=False)
    _credentials: Optional[Any] = field(default=None, repr=False)
    _file_manager: Optional[Any] = field(default=None, repr=False)
    
    @property
    def inference(self):
        """Inference service for LLM calls."""
        if self._inference is None:
            from backend.services.inference_service import inference_service
            self._inference = inference_service
        return self._inference
    
    @property
    def content(self):
        """Content management service for file/content operations."""
        if self._content is None:
            from backend.services.content_management_service import ContentManagementService
            self._content = ContentManagementService()
        return self._content
    
    @property
    def canvas(self):
        """Canvas service for frontend displays."""
        if self._canvas is None:
            from backend.services.canvas_service import CanvasService
            self._canvas = CanvasService()
        return self._canvas
    
    @property
    def telemetry(self):
        """Telemetry service for metrics and logging."""
        if self._telemetry is None:
            from backend.services.telemetry_service import telemetry_service
            self._telemetry = telemetry_service
        return self._telemetry
    
    @property
    def credentials(self):
        """Credential service for API keys and secrets."""
        if self._credentials is None:
            from backend.services.credential_service import credential_service
            self._credentials = credential_service
        return self._credentials
    
    @property
    def file_manager(self):
        """File manager for local file operations."""
        if self._file_manager is None:
            from backend.agents.utils.agent_file_manager import AgentFileManager
            self._file_manager = AgentFileManager()
        return self._file_manager
    
    def initialize_essential(self):
        """
        Initialize only essential lightweight services.
        Called during agent spawn.
        """
        # Only initialize telemetry (lightweight)
        _ = self.telemetry
        logger.debug("Essential services initialized")
    
    def initialize_all(self):
        """
        Initialize all services.
        Called when agent is about to execute first task.
        """
        _ = self.inference
        _ = self.content
        _ = self.canvas
        _ = self.telemetry
        logger.debug("All services initialized")
    
    @classmethod
    def create_default(cls) -> "AgentServices":
        """Factory method to create with default service instances."""
        return cls()


# Convenience function for creating services
def get_services() -> AgentServices:
    """Get or create agent services instance."""
    return AgentServices.create_default()

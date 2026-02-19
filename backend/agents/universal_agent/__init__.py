"""
Universal Agent - General Purpose Task Executor

A flexible agent capable of handling any arbitrary task not covered by specialized agents.
Follows the BaseAgent pattern for consistency with other agents.
"""

from backend.agents.base.server import create_agent_server
from .base_agent_impl import UniversalAgent

# Create agent server using the factory function
server = create_agent_server(
    agent_class=UniversalAgent, agent_id="universal_agent", agent_name="Universal Agent"
)

# Export the FastAPI app for uvicorn
app = server.app

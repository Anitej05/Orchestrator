"""
Document Agent Entry Point
Run with: python -m backend.agents.document_agent_lib
"""

from backend.agents.base.server import create_agent_server
from .base_agent_impl import DocumentAgent

if __name__ == "__main__":
    server = create_agent_server(
        agent_class=DocumentAgent,
        agent_id="document_agent",
        agent_name="Document Agent"
    )
    server.run(host="0.0.0.0", port=8050)

"""
Zoho Books Agent Entry Point
Run with: python -m backend.agents.zoho_books
"""

from backend.agents.base.server import create_agent_server
from .base_agent_impl import ZohoBooksAgent

if __name__ == "__main__":
    server = create_agent_server(
        agent_class=ZohoBooksAgent,
        agent_id="zoho_books",
        agent_name="Zoho Books Agent"
    )
    server.run(host="0.0.0.0", port=8060)

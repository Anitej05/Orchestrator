"""
Mail Agent Entry Point
Run with: python -m backend.agents.mail_agent
"""

from backend.agents.base.server import create_agent_server
from .base_agent_impl import MailAgent

if __name__ == "__main__":
    server = create_agent_server(
        agent_class=MailAgent,
        agent_id="mail_agent",
        agent_name="Mail Agent"
    )
    server.run(host="0.0.0.0", port=8040)

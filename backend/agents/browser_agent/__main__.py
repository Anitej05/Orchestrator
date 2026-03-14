"""
Browser Agent Entry Point
Run with: python -m backend.agents.browser_agent
"""

from backend.base_agent.server import create_agent_server
from .base_agent_impl import BrowserAgent

if __name__ == "__main__":
    server = create_agent_server(
        agent_class=BrowserAgent,
        agent_id="browser_agent",
        agent_name="Browser Agent"
    )
    server.run(host="0.0.0.0", port=8090)

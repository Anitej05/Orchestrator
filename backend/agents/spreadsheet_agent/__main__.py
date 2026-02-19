"""
Spreadsheet Agent Entry Point
Run with: python -m backend.agents.spreadsheet_agent
"""

from backend.agents.base.server import create_agent_server
from .base_agent_impl import SpreadsheetAgent

if __name__ == "__main__":
    server = create_agent_server(
        agent_class=SpreadsheetAgent,
        agent_id="spreadsheet_agent",
        agent_name="Spreadsheet Agent"
    )
    server.run(host="0.0.0.0", port=9000)

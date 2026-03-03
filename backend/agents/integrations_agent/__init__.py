"""
Integrations Agent - Universal Composio Integration Handler
BaseAgent-compliant implementation for any Composio-powered app.

Version: 2.0.0 (BaseAgent Migration Complete)
"""

__version__ = "2.0.0"


def run_agent() -> None:
    """Run the Integrations Agent server on port 8085."""
    from backend.agents.base.server import create_agent_server
    from .base_agent_impl import get_agent
    
    agent = get_agent()
    app = create_agent_server(agent)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)


if __name__ == "__main__":
    run_agent()


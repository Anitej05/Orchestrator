# agents/gmail_agent/__init__.py
"""
Gmail Agent - Clean Composio-native implementation
Uses only official Composio Gmail tools (23 tools + 2 triggers)
"""

__version__ = "1.0.0"

def run_agent() -> None:
    import uvicorn
    from .agent import app
    uvicorn.run(app, host="0.0.0.0", port=8003)


if __name__ == "__main__":
    run_agent()

"""
Universal Agent - Entry Point

Run: python -m backend.agents.universal_agent
"""

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("AGENT_PORT", 8070))
    uvicorn.run(
        "backend.agents.universal_agent:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

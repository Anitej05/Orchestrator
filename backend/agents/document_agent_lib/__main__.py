"""
Document Agent Entry Point
Run with: python -m backend.agents.document_agent_lib
"""

from .__init__ import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8050)

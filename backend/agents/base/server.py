"""
Agent HTTP Server
FastAPI wrapper for BaseAgent to expose HTTP endpoints.
"""

import logging
from typing import Type, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .agent import BaseAgent
from .types import AgentRequest, AgentResponse
from .services import AgentServices

logger = logging.getLogger(__name__)


class AgentServer:
    """
    HTTP server wrapper for any BaseAgent.
    Provides /health, /execute, /capabilities, and /metrics endpoints.
    """
    
    def __init__(
        self,
        agent_class: Type[BaseAgent],
        agent_id: str,
        agent_name: str,
        services: Optional[AgentServices] = None
    ):
        self.agent_class = agent_class
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.services = services
        
        self.agent: Optional[BaseAgent] = None
        self.app = FastAPI(
            title=f"{agent_name} API",
            description=f"HTTP API for {agent_name}",
            version="1.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with agent info."""
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            if self.agent is None:
                return {
                    "status": "not_initialized",
                    "agent_id": self.agent_id,
                    "initialized": False
                }
            
            return await self.agent.health_check()
        
        @self.app.post("/execute", response_model=AgentResponse)
        async def execute(request: AgentRequest, raw_request: Request):
            """Execute a task."""
            # Lazy initialization
            if self.agent is None:
                logger.info(f"Lazy initializing agent: {self.agent_name}")
                self.agent = self.agent_class(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    services=self.services
                )
                await self.agent.initialize()

            if not request.user_id:
                header_user_id = raw_request.headers.get("X-User-ID")
                if header_user_id:
                    request.user_id = header_user_id
            
            try:
                result = await self.agent.execute(request)
                return result
            except Exception as e:
                logger.error(f"Error executing task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/capabilities")
        async def capabilities():
            """Get list of available capabilities."""
            if self.agent is None:
                # Return capabilities from class without initializing
                temp_agent = self.agent_class(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    services=self.services
                )
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "capabilities": temp_agent.get_capabilities_info()
                }
            
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "capabilities": self.agent.get_capabilities_info()
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Get agent metrics and telemetry."""
            if self.agent is None:
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "status": "not_initialized",
                    "metrics": {}
                }
            
            return await self.agent.get_metrics()
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, log_level: str = "info"):
        """Run the server."""
        logger.info(f"Starting {self.agent_name} server on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level=log_level
        )
    
    async def shutdown(self):
        """Graceful shutdown."""
        if self.agent:
            await self.agent.terminate()
            logger.info(f"Agent {self.agent_name} shutdown complete")


def create_agent_server(
    agent_class: Type[BaseAgent],
    agent_id: str,
    agent_name: str,
    services: Optional[AgentServices] = None
) -> AgentServer:
    """
    Factory function to create an agent server.
    
    Usage:
        server = create_agent_server(
            agent_class=SpreadsheetAgent,
            agent_id="spreadsheet",
            agent_name="Spreadsheet Agent"
        )
        server.run(port=9000)
    """
    return AgentServer(
        agent_class=agent_class,
        agent_id=agent_id,
        agent_name=agent_name,
        services=services
    )

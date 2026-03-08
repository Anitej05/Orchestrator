"""
Agent HTTP Server
FastAPI wrapper for BaseAgent to expose HTTP endpoints.
"""

import asyncio
import json
import logging
from typing import Type, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
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
        
        @self.app.post("/execute")
        async def execute(request: AgentRequest):
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

            try:
                result = await self.agent.execute(request)
                # AgentResponse is a plain Python @dataclass, NOT a Pydantic BaseModel.
                # FastAPI's default JSONResponse can't serialize dataclasses directly and
                # would raise a TypeError. jsonable_encoder converts the dataclass to a
                # plain dict first, which JSONResponse can safely serialize.
                # Do NOT add a response_model= to this endpoint — that forces Pydantic
                # validation on the return value and breaks dataclass responses.
                return JSONResponse(content=jsonable_encoder(result))
            except Exception as e:
                logger.error(f"Error executing task: {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error_message": str(e),
                        "summary": "Execution failed",
                        "result": None,
                        "data": None,
                    }
                )
        
        @self.app.post("/execute/stream")
        async def execute_stream(request: AgentRequest):
            """
            Execute a task and stream progress events as SSE.

            Each event is a JSON-encoded line in the format:
                data: {"type": "progress", "message": "..."}\n\n
            A final event signals completion:
                data: {"type": "done", "result": {...}}\n\n
            Errors send:
                data: {"type": "error", "message": "..."}\n\n
            """
            # Lazy initialization (same as /execute)
            if self.agent is None:
                logger.info(f"Lazy initializing agent: {self.agent_name}")
                self.agent = self.agent_class(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    services=self.services
                )
                await self.agent.initialize()

            # Bounded queue for progress messages (max 64 items — never blocks)
            progress_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
            self.agent._progress_queue = progress_queue

            async def _event_generator():
                try:
                    execute_task = asyncio.create_task(self.agent.execute(request))

                    while not execute_task.done():
                        try:
                            msg = await asyncio.wait_for(
                                progress_queue.get(), timeout=0.4
                            )
                            yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
                        except asyncio.TimeoutError:
                            # Heartbeat so the client knows we're alive
                            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

                    # Drain any remaining messages
                    while not progress_queue.empty():
                        try:
                            msg = progress_queue.get_nowait()
                            yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
                        except asyncio.QueueEmpty:
                            break

                    result = execute_task.result()
                    # Same dataclass → dict conversion needed here as in /execute above.
                    yield f"data: {json.dumps({'type': 'done', 'result': jsonable_encoder(result)})}\n\n"

                except Exception as e:
                    logger.error(f"Error in execute/stream: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                finally:
                    # Always clear the queue reference so the agent stops emitting
                    if self.agent is not None:
                        self.agent._progress_queue = None

            return StreamingResponse(
                _event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

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

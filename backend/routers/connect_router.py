# backend/routers/connect_router.py
"""
Router for MCP connection management - DEPRECATED

MCP functionality was removed when the AgentEndpoint, EndpointParameter,
and AgentCredential tables were dropped from the schema.

These endpoints return 501 Not Implemented to inform clients that
MCP integration is no longer available.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, Optional
from sqlalchemy.orm import Session
from database import get_db
import json
import os
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/connect", tags=["connections"])


# --- Request Models (kept for API documentation) ---

class ProbeRequest(BaseModel):
    """Request to probe an MCP server URL"""
    url: str = Field(..., description="MCP server URL to probe")


class ConnectRequest(BaseModel):
    """Request to connect and ingest an MCP server"""
    url: str = Field(..., description="MCP server URL")
    credentials: Dict[str, str] = Field(default_factory=dict, description="Authentication headers")
    user_id: str = Field(..., description="User ID for credential storage")
    agent_name: Optional[str] = Field(None, description="Custom name for the agent")
    agent_description: Optional[str] = Field(None, description="Custom description")


# --- Endpoints ---

@router.post("/probe", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def probe_connection(req: ProbeRequest):
    """
    DEPRECATED: MCP server probing is no longer supported.
    
    MCP agent ingestion was removed when the AgentEndpoint, EndpointParameter,
    and AgentCredential tables were dropped from the schema.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MCP server integration is deprecated and no longer supported."
    )


@router.post("/ingest", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def ingest_connection(req: ConnectRequest, db: Session = Depends(get_db)):
    """
    DEPRECATED: MCP agent ingestion is no longer supported.
    
    The AgentEndpoint, EndpointParameter, and AgentCredential tables
    have been dropped from the schema.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MCP agent ingestion is deprecated. The required database tables have been removed."
    )


@router.get("/list", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def list_connections(user_id: str, db: Session = Depends(get_db)):
    """
    DEPRECATED: MCP connection listing is no longer supported.
    
    The AgentCredential table has been dropped.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MCP connection management is deprecated."
    )


@router.delete("/{agent_id}", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def delete_connection(agent_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    DEPRECATED: MCP connection deletion is no longer supported.
    
    The AgentCredential table has been dropped.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="MCP connection management is deprecated."
    )


@router.get("/integrations")
async def get_integrations():
    """
    Get the list of pre-configured integration templates.
    
    Returns a list of known MCP servers with their configuration details,
    making it easy for users to connect to popular services.
    """
    try:
        integrations_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "integrations.json"
        )
        
        if os.path.exists(integrations_path):
            with open(integrations_path, 'r') as f:
                integrations = json.load(f)
            return integrations
        else:
            logger.warning(f"Integrations file not found: {integrations_path}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading integrations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load integrations"
        )

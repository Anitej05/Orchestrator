# backend/routers/connect_router.py
"""
Router for MCP connection management.
Handles probing, ingestion, and management of MCP server connections.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from database import get_db
from services.mcp_service import (
    probe_mcp_url,
    ingest_mcp_agent,
    list_user_connections,
    delete_user_connection
)
import json
import os
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/connect", tags=["connections"])


# --- Request/Response Models ---

class ProbeRequest(BaseModel):
    """Request to probe an MCP server URL"""
    url: str = Field(..., description="MCP server URL to probe")


class ProbeResponse(BaseModel):
    """Response from probing an MCP server"""
    status: str
    message: Optional[str] = None
    type: Optional[str] = None
    header: Optional[str] = None
    details: Optional[str] = None


class ConnectRequest(BaseModel):
    """Request to connect and ingest an MCP server"""
    url: str = Field(..., description="MCP server URL")
    credentials: Dict[str, str] = Field(default_factory=dict, description="Authentication headers")
    user_id: str = Field(..., description="User ID for credential storage")
    agent_name: Optional[str] = Field(None, description="Custom name for the agent")
    agent_description: Optional[str] = Field(None, description="Custom description")


class ConnectResponse(BaseModel):
    """Response from connecting to an MCP server"""
    status: str
    message: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    tool_count: Optional[int] = None
    tools: Optional[List[str]] = None


class ConnectionInfo(BaseModel):
    """Information about a user's MCP connection"""
    agent_id: str
    name: str
    description: Optional[str]
    url: Optional[str]
    tool_count: int
    tools: List[str]
    created_at: Optional[str]


class DeleteResponse(BaseModel):
    """Response from deleting a connection"""
    status: str
    message: str


# --- Endpoints ---

@router.post("/probe", response_model=ProbeResponse)
async def probe_connection(req: ProbeRequest):
    """
    Probe an MCP server URL to detect authentication requirements.
    
    This endpoint checks if the server is accessible and what type of
    authentication it requires (if any).
    """
    try:
        result = await probe_mcp_url(req.url)
        return ProbeResponse(**result)
    except Exception as e:
        logger.error(f"Error probing URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/ingest", response_model=ConnectResponse)
async def ingest_connection(req: ConnectRequest, db: Session = Depends(get_db)):
    """
    Connect to an MCP server, discover its tools, and save the connection.
    
    This endpoint:
    1. Connects to the MCP server using provided credentials
    2. Discovers available tools via MCP protocol
    3. Saves the agent definition to the database
    4. Stores encrypted user credentials
    """
    try:
        result = await ingest_mcp_agent(
            db=db,
            url=req.url,
            user_id=req.user_id,
            credentials=req.credentials,
            agent_name=req.agent_name,
            agent_description=req.agent_description
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to ingest MCP agent")
            )
        
        return ConnectResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting connection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/list", response_model=List[ConnectionInfo])
async def list_connections(user_id: str, db: Session = Depends(get_db)):
    """
    List all MCP connections for a user.
    
    Returns a list of all MCP agents that the user has connected to,
    including their tools and connection details.
    """
    try:
        connections = await list_user_connections(db, user_id)
        return [ConnectionInfo(**conn) for conn in connections]
    except Exception as e:
        logger.error(f"Error listing connections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{agent_id}", response_model=DeleteResponse)
async def delete_connection(agent_id: str, user_id: str, db: Session = Depends(get_db)):
    """
    Delete a user's connection to an MCP agent.
    
    This removes the user's stored credentials for the agent but does not
    delete the agent definition itself (other users may still use it).
    """
    try:
        result = await delete_user_connection(db, user_id, agent_id)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("message", "Connection not found")
            )
        
        return DeleteResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
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

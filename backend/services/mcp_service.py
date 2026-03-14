# backend/services/mcp_service.py
"""
MCP Service - DEPRECATED

MCP agent ingestion was removed when the AgentEndpoint, EndpointParameter,
and AgentCredential tables were dropped from the schema.

These stub functions remain for backward compatibility with the /api/connect/*
endpoints, which now return 501 Not Implemented responses.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger("uvicorn.error")


async def probe_mcp_url(url: str) -> Dict[str, Any]:
    """
    STUB: MCP functionality deprecated.
    
    Probing MCP URLs is no longer supported.
    """
    return {
        "status": "error",
        "message": "MCP server integration is deprecated and no longer supported."
    }


async def ingest_mcp_agent(
    db: Session,
    url: str,
    user_id: str,
    credentials: Dict[str, str],
    agent_name: Optional[str] = None,
    agent_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    STUB: MCP agent ingestion removed.
    
    The AgentEndpoint, EndpointParameter, and AgentCredential tables
    have been dropped from the schema.
    """
    return {
        "status": "error",
        "message": "MCP agent ingestion is deprecated. The required database tables (AgentEndpoint, EndpointParameter, AgentCredential) have been removed."
    }


async def list_user_connections(db: Session, user_id: str) -> list[Dict[str, Any]]:
    """
    STUB: Connection listing removed.
    
    The AgentCredential table has been dropped.
    """
    return []


async def delete_user_connection(db: Session, user_id: str, agent_id: str) -> Dict[str, Any]:
    """
    STUB: Connection deletion removed.
    
    The AgentCredential table has been dropped.
    """
    return {
        "status": "error",
        "message": "MCP connection management is deprecated."
    }

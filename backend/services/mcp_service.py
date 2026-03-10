# backend/services/mcp_service.py
"""
MCP Service: Handles discovery, ingestion, and management of MCP servers.
"""

import httpx
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from models import Agent, AgentType
import uuid

logger = logging.getLogger("uvicorn.error")


async def probe_mcp_url(url: str) -> Dict[str, Any]:
    """
    Probe an MCP URL to determine authentication requirements.
    
    Args:
        url: The MCP server URL to probe
        
    Returns:
        Dictionary with status and auth requirements
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try SSE endpoint first (standard MCP entry point)
            sse_url = f"{url}/sse" if not url.endswith("/sse") else url
            
            try:
                response = await client.get(sse_url)
                
                if response.status_code == 200:
                    return {
                        "status": "open",
                        "message": "No authentication required",
                        "url": url
                    }
                
                if response.status_code == 401:
                    auth_header = response.headers.get("WWW-Authenticate", "")
                    
                    if "OAuth" in auth_header or "oauth" in auth_header.lower():
                        # Parse OAuth details from header
                        return {
                            "status": "auth_required",
                            "type": "oauth2",
                            "details": auth_header,
                            "message": "OAuth2 authentication required"
                        }
                    elif "Bearer" in auth_header or "bearer" in auth_header.lower():
                        return {
                            "status": "auth_required",
                            "type": "api_key",
                            "header": "Authorization",
                            "message": "API key authentication required (Bearer token)"
                        }
                    else:
                        # Generic auth required
                        return {
                            "status": "auth_required",
                            "type": "api_key",
                            "header": "Authorization",
                            "message": "Authentication required"
                        }
                
                return {
                    "status": "unknown",
                    "code": response.status_code,
                    "message": f"Unexpected status code: {response.status_code}"
                }
                
            except httpx.HTTPStatusError as e:
                return {
                    "status": "error",
                    "message": f"HTTP error: {str(e)}"
                }
                
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Connection timeout - server did not respond"
        }
    except Exception as e:
        logger.error(f"Error probing MCP URL {url}: {e}")
        return {
            "status": "error",
            "message": str(e)
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
    STUB: MCP agent ingestion removed (AgentEndpoint/EndpointParameter/AgentCredential tables dropped).
    """
    return {
        "status": "error",
        "message": "MCP agent ingestion is not supported in this version."
    }


async def list_user_connections(db: Session, user_id: str) -> list[Dict[str, Any]]:
    """STUB: AgentCredential table dropped."""
    return []


async def delete_user_connection(db: Session, user_id: str, agent_id: str) -> Dict[str, Any]:
    """STUB: AgentCredential table dropped."""
    return {"status": "error", "message": "Connection management not supported in this version."}

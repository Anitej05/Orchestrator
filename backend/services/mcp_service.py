# backend/services/mcp_service.py
"""
MCP Service: Handles discovery, ingestion, and management of MCP servers.
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from models import Agent, AgentEndpoint, AgentCredential, EndpointParameter, AgentType, AuthType
from backend.utils.encryption import encrypt
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
    Connect to an MCP server, discover tools, and save to database.
    
    Args:
        db: Database session
        url: MCP server URL
        user_id: User ID for credential storage
        credentials: Dictionary of header_name -> value for authentication
        agent_name: Optional custom name for the agent
        agent_description: Optional custom description
        
    Returns:
        Dictionary with ingestion results
    """
    try:
        # Import MCP client (lazy import to avoid startup issues if not installed)
        try:
            from mcp import ClientSession
            import mcp.client.sse
        except ImportError:
            return {
                "status": "error",
                "message": "MCP SDK not installed. Run: pip install mcp httpx-sse"
            }
        
        # Prepare headers for connection
        headers = {}
        for key, value in credentials.items():
            if value:  # Only add non-empty values
                headers[key] = value
        
        logger.info(f"Attempting to connect to MCP server: {url}")
        
        # Connect to MCP server and discover tools
        tools_list = []
        server_info = {}
        
        try:
            # Use SSE transport for MCP HTTP
            sse_url = f"{url}/sse" if not url.endswith("/sse") else url
            
            # Create SSE client with headers (newer MCP SDK version)
            async with mcp.client.sse.sse_client(sse_url, headers=headers, timeout=30.0) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    init_result = await session.initialize()
                    server_info = {
                        "name": init_result.serverInfo.name if hasattr(init_result, 'serverInfo') else "Unknown",
                        "version": init_result.serverInfo.version if hasattr(init_result, 'serverInfo') else "Unknown"
                    }
                    
                    logger.info(f"Connected to MCP server: {server_info}")
                    
                    # List available tools
                    tools_response = await session.list_tools()
                    tools_list = tools_response.tools if hasattr(tools_response, 'tools') else []
                    
                    logger.info(f"Discovered {len(tools_list)} tools from MCP server")
                        
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return {
                "status": "error",
                "message": f"Failed to connect to MCP server: {str(e)}"
            }
        
        if not tools_list:
            return {
                "status": "error",
                "message": "No tools discovered from MCP server"
            }
        
        # Check if agent already exists for this URL
        agent = db.query(Agent).filter(
            Agent.connection_config.contains({"url": url})
        ).first()
        
        if not agent:
            # Create new agent
            agent_id = str(uuid.uuid4())
            agent = Agent(
                id=agent_id,
                owner_id="system",  # System-owned MCP agents
                name=agent_name or server_info.get("name", f"MCP Agent ({url})"),
                description=agent_description or f"MCP Server: {server_info.get('name', url)}",
                agent_type=AgentType.MCP_HTTP,
                connection_config={"url": url, "server_info": server_info},
                capabilities=[tool.name for tool in tools_list],
                price_per_call_usd=0.0,
                public_key_pem=None  # Not needed for MCP agents
            )
            db.add(agent)
            db.flush()  # Get the ID
            logger.info(f"Created new MCP agent: {agent.id}")
        else:
            # Update existing agent
            agent.capabilities = [tool.name for tool in tools_list]
            agent.connection_config = {"url": url, "server_info": server_info}
            if agent_name:
                agent.name = agent_name
            if agent_description:
                agent.description = agent_description
            logger.info(f"Updated existing MCP agent: {agent.id}")
        
        # Delete old endpoints and parameters
        db.query(AgentEndpoint).filter_by(agent_id=agent.id).delete()
        
        # Add new endpoints (one per tool)
        for tool in tools_list:
            endpoint = AgentEndpoint(
                agent_id=agent.id,
                endpoint=tool.name,  # Tool name is the "endpoint"
                http_method="MCP",  # Special marker for MCP tools
                description=tool.description if hasattr(tool, 'description') else ""
            )
            db.add(endpoint)
            db.flush()
            
            # Add parameters from tool's input schema
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                properties = schema.get("properties", {})
                required_fields = schema.get("required", [])
                
                for param_name, param_schema in properties.items():
                    parameter = EndpointParameter(
                        endpoint_id=endpoint.id,
                        name=param_name,
                        description=param_schema.get("description", ""),
                        param_type=param_schema.get("type", "string"),
                        required=param_name in required_fields
                    )
                    db.add(parameter)
        
        # Save or update user credentials
        # Delete existing credentials for this user/agent
        db.query(AgentCredential).filter_by(
            user_id=user_id,
            agent_id=agent.id
        ).delete()
        
        # Add new credentials
        for header_name, value in credentials.items():
            if value:  # Only store non-empty credentials
                credential = AgentCredential(
                    user_id=user_id,
                    agent_id=agent.id,
                    auth_type=AuthType.API_KEY,
                    auth_header_name=header_name,
                    encrypted_access_token=encrypt(value)
                )
                db.add(credential)
        
        db.commit()
        
        return {
            "status": "success",
            "agent_id": agent.id,
            "agent_name": agent.name,
            "tool_count": len(tools_list),
            "tools": [tool.name for tool in tools_list]
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error ingesting MCP agent: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


async def list_user_connections(db: Session, user_id: str) -> list[Dict[str, Any]]:
    """
    List all MCP connections for a user.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        List of connection details
    """
    # Get all agents that have credentials for this user
    credentials = db.query(AgentCredential).filter_by(user_id=user_id).all()
    
    connections = []
    for cred in credentials:
        agent = cred.agent
        if agent.agent_type == AgentType.MCP_HTTP:
            connections.append({
                "agent_id": agent.id,
                "name": agent.name,
                "description": agent.description,
                "url": agent.connection_config.get("url") if agent.connection_config else None,
                "tool_count": len(agent.capabilities) if agent.capabilities else 0,
                "tools": agent.capabilities,
                "created_at": cred.created_at.isoformat() if cred.created_at else None
            })
    
    return connections


async def delete_user_connection(db: Session, user_id: str, agent_id: str) -> Dict[str, Any]:
    """
    Delete a user's connection to an MCP agent.
    
    Args:
        db: Database session
        user_id: User ID
        agent_id: Agent ID
        
    Returns:
        Result dictionary
    """
    try:
        # Delete user's credentials for this agent
        deleted_count = db.query(AgentCredential).filter_by(
            user_id=user_id,
            agent_id=agent_id
        ).delete()
        
        db.commit()
        
        if deleted_count > 0:
            return {
                "status": "success",
                "message": "Connection deleted successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Connection not found"
            }
            
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting connection: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

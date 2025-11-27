"""
Credential Service
Helper functions for managing agent credentials
"""

from sqlalchemy.orm import Session
from models import Agent, AgentCredential
from utils.encryption import encrypt, decrypt
from typing import Dict, Optional
import logging

logger = logging.getLogger("uvicorn.error")

def get_agent_credentials(
    db: Session,
    agent_id: str,
    user_id: str
) -> Dict[str, str]:
    """
    Get decrypted credentials for an agent and user.
    
    Returns:
        Dictionary of credential_name -> decrypted_value
    """
    user_cred = db.query(AgentCredential).filter_by(
        agent_id=agent_id,
        user_id=user_id,
        is_active=True
    ).first()
    
    if not user_cred:
        return {}
    
    credentials = {}
    
    # Use new encrypted_credentials field (supports multiple fields)
    if user_cred.encrypted_credentials:
        for key, encrypted_value in user_cred.encrypted_credentials.items():
            try:
                credentials[key] = decrypt(encrypted_value)
            except Exception as e:
                logger.error(f"Failed to decrypt credential '{key}' for agent {agent_id}: {e}")
    
    # Fallback to legacy fields for backward compatibility
    elif user_cred.encrypted_access_token:
        try:
            credentials['access_token'] = decrypt(user_cred.encrypted_access_token)
            credentials['auth_header_name'] = user_cred.auth_header_name or 'Authorization'
        except Exception as e:
            logger.error(f"Failed to decrypt legacy token for agent {agent_id}: {e}")
    
    return credentials

def save_agent_credentials(
    db: Session,
    agent_id: str,
    user_id: str,
    credentials: Dict[str, str]
) -> bool:
    """
    Save encrypted credentials for an agent and user.
    
    Args:
        db: Database session
        agent_id: Agent ID
        user_id: User ID
        credentials: Dictionary of credential_name -> value
        
    Returns:
        True if successful
    """
    try:
        # Encrypt all credentials
        encrypted_creds = {}
        for key, value in credentials.items():
            if value:  # Only encrypt non-empty values
                encrypted_creds[key] = encrypt(value)
        
        # Check if credentials exist
        existing_cred = db.query(AgentCredential).filter_by(
            agent_id=agent_id,
            user_id=user_id
        ).first()
        
        if existing_cred:
            # Update existing
            existing_cred.encrypted_credentials = encrypted_creds
            existing_cred.is_active = True
            from datetime import datetime
            existing_cred.updated_at = datetime.utcnow()
        else:
            # Create new
            import uuid
            new_cred = AgentCredential(
                id=str(uuid.uuid4()),
                user_id=user_id,
                agent_id=agent_id,
                encrypted_credentials=encrypted_creds,
                is_active=True
            )
            db.add(new_cred)
        
        db.commit()
        logger.info(f"Saved credentials for agent {agent_id}, user {user_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save credentials: {e}", exc_info=True)
        return False

def delete_agent_credentials(
    db: Session,
    agent_id: str,
    user_id: str
) -> bool:
    """
    Delete credentials for an agent and user.
    
    Returns:
        True if deleted, False if not found
    """
    try:
        deleted = db.query(AgentCredential).filter_by(
            agent_id=agent_id,
            user_id=user_id
        ).delete()
        
        db.commit()
        
        if deleted:
            logger.info(f"Deleted credentials for agent {agent_id}, user {user_id}")
            return True
        return False
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete credentials: {e}", exc_info=True)
        return False

def has_valid_credentials(
    db: Session,
    agent_id: str,
    user_id: str
) -> bool:
    """
    Check if user has valid credentials for an agent.
    
    Returns:
        True if credentials exist and are active
    """
    user_cred = db.query(AgentCredential).filter_by(
        agent_id=agent_id,
        user_id=user_id,
        is_active=True
    ).first()
    
    if not user_cred:
        return False
    
    # Check if has any credentials
    if user_cred.encrypted_credentials:
        return bool(user_cred.encrypted_credentials)
    elif user_cred.encrypted_access_token:
        return True
    
    return False

def get_credentials_for_headers(
    db: Session,
    agent_id: str,
    user_id: str,
    agent_type: str = "http_rest"
) -> Dict[str, str]:
    """
    Get credentials formatted as HTTP headers.
    
    Args:
        db: Database session
        agent_id: Agent ID
        user_id: User ID
        agent_type: Agent type (http_rest or mcp_http)
        
    Returns:
        Dictionary of header_name -> value
    """
    credentials = get_agent_credentials(db, agent_id, user_id)
    headers = {}
    
    if agent_type == "mcp_http":
        # MCP agents typically use x-api-key header
        if 'composio_api_key' in credentials:
            headers['x-api-key'] = credentials['composio_api_key']
        elif 'api_key' in credentials:
            headers['x-api-key'] = credentials['api_key']
    else:
        # REST agents typically use Authorization header
        if 'api_key' in credentials:
            headers['Authorization'] = f"Bearer {credentials['api_key']}"
        elif 'access_token' in credentials:
            auth_header = credentials.get('auth_header_name', 'Authorization')
            headers[auth_header] = f"Bearer {credentials['access_token']}"
    
    # Add any other credentials as headers
    for key, value in credentials.items():
        if key not in ['api_key', 'access_token', 'auth_header_name', 'composio_api_key']:
            headers[key] = value
    
    return headers

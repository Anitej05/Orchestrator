"""
Credentials Router
Manages user credentials for agents
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel
from database import SessionLocal
from models import Agent, AgentCredential
from backend.utils.encryption import encrypt, decrypt
from auth import get_current_user_id
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/credentials", tags=["credentials"])

# Pydantic models
class CredentialFieldValue(BaseModel):
    field_name: str
    value: str

class SaveCredentialsRequest(BaseModel):
    agent_id: str
    credentials: List[CredentialFieldValue]

class CredentialStatus(BaseModel):
    agent_id: str
    agent_name: str
    agent_type: str
    requires_credentials: bool
    credential_fields: List[Dict[str, Any]]
    is_configured: bool
    configured_fields: List[str]
    created_at: str = None
    updated_at: str = None

class AgentCredentialResponse(BaseModel):
    agents: List[CredentialStatus]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/status", response_model=AgentCredentialResponse)
async def get_credentials_status(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get credential status for all agents for the current user
    Shows which agents need credentials and which are configured
    """
    try:
        # Get all active agents
        agents = db.query(Agent).filter(Agent.status == "active").all()
        
        agent_statuses = []
        for agent in agents:
            # Check if user has credentials for this agent
            user_cred = db.query(AgentCredential).filter(
                AgentCredential.agent_id == agent.id,
                AgentCredential.user_id == user_id,
                AgentCredential.is_active == True
            ).first()
            
            # Determine configured fields
            configured_fields = []
            if user_cred and user_cred.encrypted_credentials:
                configured_fields = list(user_cred.encrypted_credentials.keys())
            
            agent_statuses.append(CredentialStatus(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_type=agent.agent_type,
                requires_credentials=agent.requires_credentials or False,
                credential_fields=agent.credential_fields or [],
                is_configured=bool(user_cred and configured_fields),
                configured_fields=configured_fields,
                created_at=user_cred.created_at.isoformat() if user_cred else None,
                updated_at=user_cred.updated_at.isoformat() if user_cred else None
            ))
        
        return AgentCredentialResponse(agents=agent_statuses)
        
    except Exception as e:
        logger.error(f"Error getting credentials status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get credentials status: {str(e)}"
        )

@router.get("/{agent_id}")
async def get_agent_credentials(
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get credentials for a specific agent (returns field names only, not values)
    """
    try:
        # Get agent
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get user credentials
        user_cred = db.query(AgentCredential).filter(
            AgentCredential.agent_id == agent_id,
            AgentCredential.user_id == user_id,
            AgentCredential.is_active == True
        ).first()
        
        configured_fields = []
        if user_cred and user_cred.encrypted_credentials:
            configured_fields = list(user_cred.encrypted_credentials.keys())
        
        return {
            "agent_id": agent.id,
            "agent_name": agent.name,
            "agent_type": agent.agent_type,
            "requires_credentials": agent.requires_credentials or False,
            "credential_fields": agent.credential_fields or [],
            "is_configured": bool(configured_fields),
            "configured_fields": configured_fields
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent credentials: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent credentials: {str(e)}"
        )

@router.post("/{agent_id}")
async def save_agent_credentials(
    agent_id: str,
    request: SaveCredentialsRequest,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Save or update credentials for an agent
    """
    try:
        # Verify agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Encrypt all credential values
        encrypted_creds = {}
        for cred in request.credentials:
            if cred.value:  # Only encrypt non-empty values
                encrypted_creds[cred.field_name] = encrypt(cred.value)
        
        # Check if credentials already exist
        existing_cred = db.query(AgentCredential).filter(
            AgentCredential.agent_id == agent_id,
            AgentCredential.user_id == user_id
        ).first()
        
        if existing_cred:
            # Update existing
            existing_cred.encrypted_credentials = encrypted_creds
            existing_cred.is_active = True
            from datetime import datetime
            existing_cred.updated_at = datetime.utcnow()
            logger.info(f"Updated credentials for agent {agent_id}, user {user_id}")
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
            logger.info(f"Created credentials for agent {agent_id}, user {user_id}")
        
        db.commit()
        
        return {
            "success": True,
            "message": "Credentials saved successfully",
            "agent_id": agent_id,
            "configured_fields": list(encrypted_creds.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving credentials: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save credentials: {str(e)}"
        )

@router.delete("/{agent_id}")
async def delete_agent_credentials(
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Delete credentials for an agent
    """
    try:
        # Find and delete credentials
        deleted = db.query(AgentCredential).filter(
            AgentCredential.agent_id == agent_id,
            AgentCredential.user_id == user_id
        ).delete()
        
        db.commit()
        
        if deleted:
            logger.info(f"Deleted credentials for agent {agent_id}, user {user_id}")
            return {
                "success": True,
                "message": "Credentials deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Credentials not found")
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting credentials: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete credentials: {str(e)}"
        )

@router.post("/{agent_id}/test")
async def test_agent_credentials(
    agent_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Test if agent credentials are valid (basic connectivity test)
    """
    try:
        # Get agent and credentials
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        user_cred = db.query(AgentCredential).filter(
            AgentCredential.agent_id == agent_id,
            AgentCredential.user_id == user_id,
            AgentCredential.is_active == True
        ).first()
        
        if not user_cred or not user_cred.encrypted_credentials:
            raise HTTPException(status_code=404, detail="Credentials not configured")
        
        # Decrypt credentials for testing
        decrypted_creds = {}
        for key, encrypted_value in user_cred.encrypted_credentials.items():
            decrypted_creds[key] = decrypt(encrypted_value)
        
        # TODO: Implement actual connectivity tests based on agent type
        # For now, just verify credentials exist
        
        return {
            "success": True,
            "message": "Credentials are configured",
            "agent_id": agent_id,
            "agent_name": agent.name,
            "configured_fields": list(decrypted_creds.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing credentials: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test credentials: {str(e)}"
        )

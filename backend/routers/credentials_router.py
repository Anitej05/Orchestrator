"""
Credentials Router -- STUB
AgentCredential table has been dropped. All endpoints return 410 Gone.
"""

from fastapi import APIRouter, HTTPException, status
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/credentials", tags=["credentials"])


@router.get("/status")
async def get_credentials_status():
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Agent credentials feature removed.")


@router.get("/{agent_id}")
async def get_agent_credentials(agent_id: str):
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Agent credentials feature removed.")


@router.post("/{agent_id}")
async def save_agent_credentials(agent_id: str):
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Agent credentials feature removed.")


@router.delete("/{agent_id}")
async def delete_agent_credentials(agent_id: str):
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Agent credentials feature removed.")


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


# ============================================================================
# SCOPE-BASED ENDPOINTS (New - for tools, system, and generic scopes)
# ============================================================================

class ScopedSaveRequest(BaseModel):
    credentials: List[CredentialFieldValue]

class ScopedCredentialStatus(BaseModel):
    scope: str
    scope_id: str
    is_configured: bool
    configured_fields: List[str]
    created_at: str = None
    updated_at: str = None


@router.get('/scope/{scope}/status')
async def get_scoped_credentials_status(
    scope: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    from models import Credential
    try:
        rows = (
            db.query(Credential)
            .filter_by(scope=scope, user_id=user_id, is_active=True)
            .all()
        )
        statuses = []
        for row in rows:
            configured = list(row.encrypted_credentials.keys()) if row.encrypted_credentials else []
            statuses.append(ScopedCredentialStatus(
                scope=row.scope,
                scope_id=row.scope_id,
                is_configured=bool(configured),
                configured_fields=configured,
                created_at=row.created_at.isoformat() if row.created_at else None,
                updated_at=row.updated_at.isoformat() if row.updated_at else None,
            ))
        return {'scope': scope, 'credentials': statuses}
    except Exception as e:
        logger.error(f'Error getting scoped credentials: {e}', exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post('/scope/{scope}/{scope_id}')
async def save_scoped_credentials(
    scope: str,
    scope_id: str,
    request: ScopedSaveRequest,
    user_id: str = Depends(get_current_user_id),
):
    from backend.services.credential_service import credential_manager
    creds_dict = {c.field_name: c.value for c in request.credentials if c.value}
    success = credential_manager.save(scope, scope_id, creds_dict, user_id)
    if success:
        return {'success': True, 'message': f'Credentials saved for {scope}/{scope_id}'}
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to save credentials')


@router.delete('/scope/{scope}/{scope_id}')
async def delete_scoped_credentials(
    scope: str,
    scope_id: str,
    user_id: str = Depends(get_current_user_id),
):
    from backend.services.credential_service import credential_manager
    deleted = credential_manager.delete(scope, scope_id, user_id)
    if deleted:
        return {'success': True, 'message': f'Credentials deleted for {scope}/{scope_id}'}
    else:
        raise HTTPException(status_code=404, detail='Credentials not found')


@router.get('/scope/{scope}/{scope_id}')
async def get_scoped_credentials(
    scope: str,
    scope_id: str,
    user_id: str = Depends(get_current_user_id),
):
    from backend.services.credential_service import credential_manager
    creds = credential_manager.get_all(scope, scope_id, user_id)
    return {
        'scope': scope,
        'scope_id': scope_id,
        'is_configured': bool(creds),
        'configured_fields': list(creds.keys()),
    }



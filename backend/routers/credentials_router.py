"""
System Credentials Router

API endpoints for managing system-level credentials (LLM API keys, etc.).
Stored encrypted in the database with .env fallback.

Usage Examples:
    # Get stored keys (names only, not values)
    GET /api/credentials/system/llm_providers
    
    # Add LLM API keys
    POST /api/credentials/system/llm_providers
    {
        "credentials": {
            "OLLAMA_API_KEY": "your-key",
            "GROQ_API_KEY": "your-key"
        }
    }
    
    # Delete specific keys
    DELETE /api/credentials/system/llm_providers
    {
        "keys": ["GROQ_API_KEY"]
    }
    
    # Test if a key works
    POST /api/credentials/test/OLLAMA_API_KEY
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("credentials_router")

router = APIRouter(prefix="/api/credentials", tags=["credentials"])


# ============================================================================
# Request/Response Models
# ============================================================================

class CredentialUpdateRequest(BaseModel):
    """Request to add/update system credentials."""
    credentials: Dict[str, str] = Field(
        ..., 
        description="Key-value pairs of credential names and values",
        example={"OLLAMA_API_KEY": "sk-xxx", "GROQ_API_KEY": "gsk_xxx"}
    )
    user_id: Optional[str] = Field("system", description="User ID (defaults to 'system')")


class CredentialKeysResponse(BaseModel):
    """Response with credential metadata (never returns actual values)."""
    scope: str
    scope_id: str
    keys: List[str]  # Only key names, not values
    has_valid: bool


class CredentialDeleteRequest(BaseModel):
    """Request to delete specific credentials."""
    keys: List[str] = Field(..., description="List of credential keys to delete")
    user_id: Optional[str] = Field("system", description="User ID")


class TestCredentialRequest(BaseModel):
    """Request to test a credential."""
    key_name: str
    test_endpoint: Optional[str] = None


class TestCredentialResponse(BaseModel):
    """Response from credential test."""
    valid: bool
    message: str
    key_name: str


# ============================================================================
# Helper Functions
# ============================================================================

def get_credential_manager():
    """Get credential manager instance."""
    from backend.services.credential_service import credential_manager
    return credential_manager


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/system/llm_providers", response_model=CredentialKeysResponse)
async def get_llm_credentials(
    credential_manager = Depends(get_credential_manager)
):
    """
    Get metadata about stored LLM credentials.
    
    Returns key names only (never actual values for security).
    Use this to check which API keys are configured.
    """
    try:
        creds = credential_manager.get_all("system", "llm_providers", user_id="system")
        
        return CredentialKeysResponse(
            scope="system",
            scope_id="llm_providers",
            keys=list(creds.keys()),
            has_valid=len(creds) > 0
        )
        
    except Exception as e:
        logger.error(f"Failed to get LLM credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get credentials: {str(e)}"
        )


@router.post("/system/llm_providers", response_model=CredentialKeysResponse)
async def update_llm_credentials(
    request: CredentialUpdateRequest,
    credential_manager = Depends(get_credential_manager)
):
    """
    Add or update LLM API keys for the system.
    
    This stores credentials encrypted in the database under:
    - scope: "system"
    - scope_id: "llm_providers"
    
    The credentials will be used by inference_service and all agents.
    
    **Supported Keys:**
    - OLLAMA_API_KEY
    - CEREBRAS_API_KEY
    - GROQ_API_KEY
    - NVIDIA_API_KEY
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    
    **Note:** Keys are encrypted in the database. Values are never returned in responses.
    """
    try:
        success = credential_manager.save(
            scope="system",
            scope_id="llm_providers",
            credentials=request.credentials,
            user_id=request.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save credentials"
            )
        
        # Return keys stored (not values)
        creds = credential_manager.get_all("system", "llm_providers", user_id=request.user_id)
        
        return CredentialKeysResponse(
            scope="system",
            scope_id="llm_providers",
            keys=list(creds.keys()),
            has_valid=len(creds) > 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save LLM credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save credentials: {str(e)}"
        )


@router.delete("/system/llm_providers", response_model=CredentialKeysResponse)
async def delete_llm_credentials(
    request: CredentialDeleteRequest,
    credential_manager = Depends(get_credential_manager)
):
    """
    Delete specific LLM API keys.
    
    Use this to rotate keys or remove unused providers.
    """
    try:
        # Get current credentials
        current = credential_manager.get_all("system", "llm_providers", user_id=request.user_id)
        
        # Remove specified keys
        for key in request.keys:
            if key in current:
                del current[key]
        
        # Save updated credentials (without deleted keys)
        success = credential_manager.save(
            scope="system",
            scope_id="llm_providers",
            credentials=current,
            user_id=request.user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update credentials"
            )
        
        return CredentialKeysResponse(
            scope="system",
            scope_id="llm_providers",
            keys=list(current.keys()),
            has_valid=len(current) > 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete LLM credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete credentials: {str(e)}"
        )


@router.post("/test/{key_name}", response_model=TestCredentialResponse)
async def test_credential(
    key_name: str,
    request: Optional[TestCredentialRequest] = None,
    credential_manager = Depends(get_credential_manager)
):
    """
    Test if a credential is valid by making a simple API call.
    
    This helps verify keys before using them in production.
    """
    try:
        # Get the API key
        api_key = credential_manager.get("system", "llm_providers", key_name)
        
        if not api_key:
            return TestCredentialResponse(
                valid=False,
                message=f"Credential '{key_name}' not found in database or .env",
                key_name=key_name
            )
        
        # Test the key based on provider
        import httpx
        
        test_endpoints = {
            "OLLAMA_API_KEY": ("https://ollama.com/v1/models", "Authorization", "Bearer"),
            "CEREBRAS_API_KEY": ("https://api.cerebras.ai/v1/models", "Authorization", "Bearer"),
            "GROQ_API_KEY": ("https://api.groq.com/openai/v1/models", "Authorization", "Bearer"),
            "NVIDIA_API_KEY": ("https://integrate.api.nvidia.com/v1/models", "Authorization", "Bearer"),
            "OPENAI_API_KEY": ("https://api.openai.com/v1/models", "Authorization", "Bearer"),
        }
        
        if key_name not in test_endpoints:
            return TestCredentialResponse(
                valid=False,
                message=f"Unknown credential: {key_name}. Supported: {list(test_endpoints.keys())}",
                key_name=key_name
            )
        
        url, header_name, header_prefix = test_endpoints[key_name]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={header_name: f"{header_prefix} {api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                return TestCredentialResponse(
                    valid=True,
                    message=f"{key_name} is valid and working",
                    key_name=key_name
                )
            else:
                return TestCredentialResponse(
                    valid=False,
                    message=f"Invalid key (HTTP {response.status_code}): {response.text[:200]}",
                    key_name=key_name
                )
                
    except httpx.RequestError as e:
        return TestCredentialResponse(
            valid=False,
            message=f"Connection error: {str(e)}",
            key_name=key_name
        )
    except Exception as e:
        logger.error(f"Failed to test credential {key_name}: {e}")
        return TestCredentialResponse(
            valid=False,
            message=f"Test failed: {str(e)}",
            key_name=key_name
        )

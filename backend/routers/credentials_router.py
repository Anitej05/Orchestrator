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
async def test_agent_credentials(agent_id: str):
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Agent credentials feature removed.")

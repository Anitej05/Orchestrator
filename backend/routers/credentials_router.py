"""
Credentials Router -- STUB
AgentCredential table has been dropped. All endpoints return 410 Gone.
"""

from fastapi import APIRouter, HTTPException, status
import logging

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/credentials", tags=["credentials"])

@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(path: str):
    raise HTTPException(status_code=status.HTTP_410_GONE, detail="Credentials endpoints have been removed in favor of Composio in-chat OAuth.")

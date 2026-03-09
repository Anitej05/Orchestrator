"""
Credential Service -- STUB
AgentCredential table has been dropped. All functions return empty/False.
"""

from sqlalchemy.orm import Session
from typing import Dict
import logging

logger = logging.getLogger("uvicorn.error")


def get_agent_credentials(db: Session, agent_id: str, user_id: str) -> Dict[str, str]:
    """STUB: AgentCredential table dropped."""
    return {}


def save_agent_credentials(db: Session, agent_id: str, user_id: str, credentials: Dict[str, str]) -> bool:
    """STUB: AgentCredential table dropped."""
    logger.warning("save_agent_credentials called but AgentCredential table has been dropped.")
    return False


def delete_agent_credentials(db: Session, agent_id: str, user_id: str) -> bool:
    """STUB: AgentCredential table dropped."""
    return False


def has_valid_credentials(db: Session, agent_id: str, user_id: str) -> bool:
    """STUB: AgentCredential table dropped."""
    return False


def get_credentials_for_headers(db: Session, agent_id: str, user_id: str, agent_type: str = "http_rest") -> Dict[str, str]:
    """STUB: AgentCredential table dropped."""
    return {}

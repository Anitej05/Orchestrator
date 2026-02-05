import logging
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session, joinedload
from models import Agent, AgentEndpoint, AgentCapability, StatusEnum
from database import SessionLocal
import json
from pathlib import Path

logger = logging.getLogger("AgentRegistryService")

class AgentRegistryService:
    """
    Central service for discovering, retrieving, and validating agents.
    Replaces ad-hoc query logic in the orchestrator.
    """
    
    def __init__(self):
        self._agent_cache: Dict[str, Dict] = {}
        self._last_refresh = 0
    
    def list_active_agents(self, db: Session = None) -> List[Dict[str, Any]]:
        """
        List all active agents with their metadata and endpoints.
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True
            
        try:
            query = db.query(Agent).options(
                joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
            ).filter(Agent.status == StatusEnum.active)
            
            agents = query.all()
            
            catalog = []
            for agent in agents:
                catalog.append(self._serialize_agent(agent))
                
            return catalog
        except Exception as e:
            logger.error(f"Failed to list active agents: {e}")
            return []
        finally:
            if should_close_db:
                db.close()

    def get_agent(self, agent_id: str, db: Session = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve details for a specific agent.
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True
            
        try:
            agent = db.query(Agent).options(
                joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
            ).filter(Agent.id == agent_id).first()
            
            if agent:
                return self._serialize_agent(agent)
            return None
        except Exception as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
        finally:
            if should_close_db:
                db.close()

    def validate_capability(self, agent_id: str, task_description: str) -> bool:
        """
        Validate if an agent can perform a task based on its capabilities.
        Currently a placeholder for semantic validation logic.
        """
        # In the future, this could use vector search against AgentCapability
        return True

    def get_request_format(self, agent_id: str, endpoint_path: str) -> Optional[str]:
        """
        Get the request format (json/form) for a specific agent endpoint.
        Falls back to file-based metadata if not in DB.
        """
        # 1. Try DB lookup (if we had request_format in DB model, which we do)
        # For now, implementing the file-based fallback logic extracted from graph.py
        
        try:
            normalized_endpoint_path = (endpoint_path or "").strip()
            if normalized_endpoint_path and not normalized_endpoint_path.startswith('/'):
                normalized_endpoint_path = '/' + normalized_endpoint_path

            # Determine backend directory relative to this service file
            # services/ -> backend/
            backend_dir = Path(__file__).resolve().parents[1]
            entry_path = backend_dir / 'agent_entries' / f'{agent_id}.json'
            
            if not entry_path.exists():
                return None

            data = json.loads(entry_path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                return None

            for ep in data.get('endpoints', []) or []:
                if not isinstance(ep, dict):
                    continue

                ep_path = (ep.get('endpoint') or "").strip()
                if ep_path and not ep_path.startswith('/'):
                    ep_path = '/' + ep_path

                if ep_path == normalized_endpoint_path:
                    rf = ep.get('request_format')
                    if isinstance(rf, str) and rf.strip():
                        return rf.strip()
                    return None
            return None
        except Exception as e:
            logger.error(f"Error resolving request format for {agent_id}: {e}")
            return None

    def _serialize_agent(self, agent: Agent) -> Dict[str, Any]:
        """Convert Agent ORM object to dictionary"""
        endpoints_info = []
        for ep in agent.endpoints:
            endpoints_info.append({
                "endpoint": ep.endpoint,
                "http_method": ep.http_method,
                "description": ep.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.param_type,
                        "description": p.description,
                        "required": p.required
                    } for p in ep.parameters
                ]
            })
            
        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "rating": agent.rating,
            "price_per_call_usd": agent.price_per_call_usd,
            "endpoints": endpoints_info,
            "type": agent.agent_type,
            "connection_config": agent.connection_config
            # Add other fields as needed
        }

# Global singleton
agent_registry = AgentRegistryService()

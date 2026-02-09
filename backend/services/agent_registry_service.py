"""
Agent Registry Service - Unified Agent Protocol (UAP) Edition

Central service for discovering, retrieving, and routing to agents.
Now reads from SKILL.md files instead of verbose JSON configurations.

UAP guarantees: ALL agents expose /execute, /continue, /health endpoints.
"""

import logging
import re
import yaml
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session, joinedload
from models import Agent, AgentEndpoint, AgentCapability, StatusEnum
from database import SessionLocal
import json
from pathlib import Path

logger = logging.getLogger("AgentRegistryService")

# Agent directories to scan for SKILL.md files
AGENT_DIRS = [
    "spreadsheet_agent",
    "mail_agent", 
    "browser_agent",
    "document_agent_lib",
    "zoho_books",
]


def parse_skill_md(skill_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a SKILL.md file and extract agent configuration.
    
    Returns:
        Dict with id, name, port, version, description, and full skill text
    """
    try:
        content = skill_path.read_text(encoding='utf-8')
        
        # Extract YAML frontmatter (between --- markers)
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not frontmatter_match:
            logger.warning(f"No frontmatter found in {skill_path}")
            return None
            
        frontmatter_text = frontmatter_match.group(1)
        body = content[frontmatter_match.end():]
        
        # Parse YAML frontmatter
        config = yaml.safe_load(frontmatter_text)
        if not config:
            return None
            
        # Required fields
        agent_id = config.get('id')
        if not agent_id:
            logger.warning(f"Missing 'id' in {skill_path}")
            return None
            
        # Build agent config
        return {
            "id": agent_id,
            "name": config.get('name', agent_id),
            "port": config.get('port', 8000),
            "version": config.get('version', '1.0.0'),
            "host": config.get('host', 'localhost'),
            "description": _extract_description(body),
            "skill_text": body.strip(),  # Full SKILL.md body for LLM context
        }
        
    except Exception as e:
        logger.error(f"Failed to parse {skill_path}: {e}")
        return None


def _extract_description(body: str) -> str:
    """Extract first paragraph after the title as description."""
    lines = body.strip().split('\n')
    description_lines = []
    in_description = False
    
    for line in lines:
        # Skip the title line
        if line.startswith('# '):
            in_description = True
            continue
        # Stop at next heading
        if line.startswith('#'):
            break
        # Collect non-empty lines
        if in_description and line.strip():
            description_lines.append(line.strip())
        # Stop after first paragraph
        if in_description and not line.strip() and description_lines:
            break
            
    return ' '.join(description_lines) if description_lines else ""


class AgentRegistryService:
    """
    Central service for discovering, retrieving, and validating agents.
    
    UAP Edition: Reads from SKILL.md files instead of JSON configs.
    All agents are guaranteed to have /execute, /continue, /health endpoints.
    """
    
    def __init__(self):
        self._agent_cache: Dict[str, Dict] = {}
        self._skill_cache: Dict[str, Dict] = {}
        self._last_refresh = 0
        
    def _load_skill_configs(self) -> Dict[str, Dict]:
        """Load all SKILL.md configurations."""
        if self._skill_cache:
            return self._skill_cache
            
        backend_dir = Path(__file__).resolve().parents[1]
        agents_dir = backend_dir / 'agents'
        
        for agent_subdir in AGENT_DIRS:
            skill_path = agents_dir / agent_subdir / 'SKILL.md'
            if skill_path.exists():
                config = parse_skill_md(skill_path)
                if config:
                    self._skill_cache[config['id']] = config
                    logger.debug(f"Loaded SKILL.md for {config['id']}")
                    
        return self._skill_cache
    
    def get_agent_skill(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the SKILL.md configuration for an agent.
        
        Returns:
            Dict with id, name, port, version, description, skill_text
        """
        skills = self._load_skill_configs()
        return skills.get(agent_id)
    
    def get_all_skills_context(self) -> str:
        """
        Get a formatted string of all agent skills for LLM context.
        Used by the Brain to understand available agents.
        """
        skills = self._load_skill_configs()
        
        context_parts = []
        for agent_id, config in skills.items():
            context_parts.append(f"## {config['name']} (id: {agent_id})")
            context_parts.append(config.get('skill_text', config.get('description', '')))
            context_parts.append("")  # Empty line between agents
            
        return '\n'.join(context_parts)
    
    def list_active_agents(self, db: Session = None) -> List[Dict[str, Any]]:
        """
        List all active agents with their metadata.
        
        UAP Edition: Also includes agents from SKILL.md files.
        """
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True
            
        try:
            # Get DB agents
            query = db.query(Agent).options(
                joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
            ).filter(Agent.status == StatusEnum.active)
            
            agents = query.all()
            
            catalog = []
            seen_ids = set()
            
            # Add DB agents
            for agent in agents:
                serialized = self._serialize_agent(agent)
                catalog.append(serialized)
                seen_ids.add(agent.id)
            
            # Add SKILL.md agents not in DB
            skills = self._load_skill_configs()
            for agent_id, skill_config in skills.items():
                if agent_id not in seen_ids:
                    catalog.append({
                        "id": agent_id,
                        "name": skill_config['name'],
                        "description": skill_config['description'],
                        "capabilities": [],
                        "rating": None,
                        "price_per_call_usd": None,
                        "endpoints": [],  # UAP: endpoints are standardized, no need to list
                        "type": "http_rest",
                        "connection_config": {
                            "base_url": f"http://{skill_config['host']}:{skill_config['port']}"
                        }
                    })
                    
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
            
            # Fallback: check SKILL.md
            skill = self.get_agent_skill(agent_id)
            if skill:
                return {
                    "id": agent_id,
                    "name": skill['name'],
                    "description": skill['description'],
                    "capabilities": [],
                    "rating": None,
                    "price_per_call_usd": None,
                    "endpoints": [],
                    "type": "http_rest",
                    "connection_config": {
                        "base_url": f"http://{skill['host']}:{skill['port']}"
                    }
                }
                
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

    def get_agent_url(self, agent_id: str, agent_name: str = None, db: Session = None) -> Optional[str]:
        """
        Get the base URL for an agent.
        
        UAP Edition: Checks SKILL.md first, then DB, then legacy JSON.
        
        Args:
            agent_id: The agent's unique ID
            agent_name: Optional agent name for fallback lookup
            db: Optional database session
            
        Returns:
            Base URL string or None if not found
        """
        # 1. Check SKILL.md first (preferred for UAP)
        skill = self.get_agent_skill(agent_id)
        if skill:
            return f"http://{skill['host']}:{skill['port']}"
        
        should_close_db = False
        if db is None:
            db = SessionLocal()
            should_close_db = True
            
        try:
            # 2. Try DB lookup
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if agent and agent.connection_config:
                config = agent.connection_config
                if isinstance(config, dict):
                    base_url = config.get('base_url') or config.get('url')
                    if base_url:
                        return base_url
            
            logger.warning(f"No base URL found for agent {agent_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting agent URL for {agent_id}: {e}")
            return None
        finally:
            if should_close_db:
                db.close()

    def get_request_format(self, agent_id: str, endpoint_path: str) -> Optional[str]:
        """
        Get the request format for an endpoint.
        
        UAP Edition: Always returns 'json' for standard endpoints.
        """
        # UAP: Standard endpoints always use JSON
        return 'json'

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
        }


# Global singleton
agent_registry = AgentRegistryService()

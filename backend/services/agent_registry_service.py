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
from models import Agent, AgentEndpoint, StatusEnum, AgentType
from database import SessionLocal, Base, engine
from pathlib import Path

logger = logging.getLogger("AgentRegistryService")

# Agent directories to scan for SKILL.md files
AGENT_DIRS = [
    "spreadsheet_agent",
    "mail_agent",
    "gmail_agent",
    "browser_agent",
    "document_agent_lib",
    "zoho_books",
    "universal_agent",
    "coding_agent",
]

# Standardized agent aliases mapping for consistent naming across the orchestrator
# Maps common aliases/short names to canonical SKILL.md names
AGENT_ALIASES = {
    # Browser agent aliases
    "browser": "Browser Automation Agent",
    "browser_agent": "Browser Automation Agent",
    "web_agent": "Browser Automation Agent",
    "web_browser": "Browser Automation Agent",
    # Spreadsheet agent aliases
    "spreadsheet": "Spreadsheet Agent",
    "spreadsheet_agent": "Spreadsheet Agent",
    "excel": "Spreadsheet Agent",
    "csv": "Spreadsheet Agent",
    "data_agent": "Spreadsheet Agent",
    # Mail agent aliases
    "mail": "Gmail Agent",
    "mail_agent": "Gmail Agent",
    "email": "Gmail Agent",
    "gmail": "Gmail Agent",
    # Document agent aliases
    "document": "Document Agent",
    "document_agent": "Document Agent",
    "pdf": "Document Agent",
    "word": "Document Agent",
    "docx": "Document Agent",
    # Zoho agent aliases
    "zoho": "Zoho Books Agent",
    "zoho_agent": "Zoho Books Agent",
    "zoho_books": "Zoho Books Agent",
    "accounting": "Zoho Books Agent",
    "invoice": "Zoho Books Agent",
    # Universal agent aliases
    "universal": "Universal Agent",
    "universal_agent": "Universal Agent",
    "general": "Universal Agent",
    "general_agent": "Universal Agent",
    # Coding agent aliases
    "coding": "Coding Agent",
    "coding_agent": "Coding Agent",
    "code": "Coding Agent",
    "developer": "Coding Agent",
    "opencode": "Coding Agent",
    "coder": "Coding Agent",
}


def normalize_agent_name(name: str) -> str:
    """
    Normalize an agent name using the centralized alias mapping.

    Args:
        name: Agent name or alias (case-insensitive)

    Returns:
        Canonical agent name from SKILL.md files
    """
    if not name:
        return None

    normalized = name.strip().lower()
    return AGENT_ALIASES.get(normalized, name)


def parse_skill_md(skill_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a SKILL.md file and extract agent configuration.

    Returns:
        Dict with id, name, port, version, description, and full skill text
    """
    try:
        content = skill_path.read_text(encoding="utf-8")

        # Extract YAML frontmatter (between --- markers)
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not frontmatter_match:
            logger.warning(f"No frontmatter found in {skill_path}")
            return None

        frontmatter_text = frontmatter_match.group(1)
        body = content[frontmatter_match.end() :]

        # Parse YAML frontmatter
        config = yaml.safe_load(frontmatter_text)
        if not config:
            return None

        # Required fields
        agent_id = config.get("id")
        if not agent_id:
            logger.warning(f"Missing 'id' in {skill_path}")
            return None

        # Build agent config
        return {
            "id": agent_id,
            "name": config.get("name", agent_id),
            "port": config.get("port", 8000),
            "version": config.get("version", "1.0.0"),
            "host": config.get("host", "localhost"),
            "description": _extract_description(body),
            "skill_text": body.strip(),  # Full SKILL.md body for LLM context
        }

    except Exception as e:
        logger.error(f"Failed to parse {skill_path}: {e}")
        return None


def _extract_description(body: str) -> str:
    """Extract first paragraph after the title as description."""
    lines = body.strip().split("\n")
    description_lines = []
    in_description = False

    for line in lines:
        # Skip the title line
        if line.startswith("# "):
            in_description = True
            continue
        # Stop at next heading
        if line.startswith("#"):
            break
        # Collect non-empty lines
        if in_description and line.strip():
            description_lines.append(line.strip())
        # Stop after first paragraph
        if in_description and not line.strip() and description_lines:
            break

    return " ".join(description_lines) if description_lines else ""


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
        agents_dir = backend_dir / "agents"

        for agent_subdir in AGENT_DIRS:
            skill_path = agents_dir / agent_subdir / "SKILL.md"
            if skill_path.exists():
                config = parse_skill_md(skill_path)
                if config:
                    self._skill_cache[config["id"]] = config
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

    def find_agent(
        self, name_or_id: str, active_agents: List[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find an agent by name or ID using standardized normalization.

        This is the centralized method for agent lookup across the orchestrator.
        Uses AGENT_ALIASES for consistent name matching.

        Args:
            name_or_id: Agent name, ID, or alias (e.g., "browser", "Browser Agent", "browser_automation_agent")
            active_agents: Optional list of active agents (defaults to list_active_agents())

        Returns:
            Agent dict if found, None otherwise
        """
        if not name_or_id:
            return None

        if active_agents is None:
            active_agents = self.list_active_agents()

        # Try exact match first (case-insensitive on name, exact on id)
        for agent in active_agents:
            if agent["name"].lower() == name_or_id.lower() or agent["id"] == name_or_id:
                return agent

        # Try normalized alias match
        normalized_name = normalize_agent_name(name_or_id)
        if normalized_name:
            for agent in active_agents:
                if agent["name"].lower() == normalized_name.lower():
                    return agent

        return None

    def get_canonical_name(self, name_or_id: str) -> Optional[str]:
        """
        Get the canonical agent name from SKILL.md files.

        Args:
            name_or_id: Agent name, ID, or alias

        Returns:
            Canonical agent name or None if not found
        """
        agent = self.find_agent(name_or_id)
        return agent["name"] if agent else None

    def get_all_skills_context(self) -> str:
        """
        Get a formatted string of all agent skills for LLM context.
        Used by the Brain to understand available agents.
        """
        skills = self._load_skill_configs()

        context_parts = []
        for agent_id, config in skills.items():
            context_parts.append(f"## {config['name']} (id: {agent_id})")
            context_parts.append(
                config.get("skill_text", config.get("description", ""))
            )
            context_parts.append("")  # Empty line between agents

        return "\n".join(context_parts)

    def list_active_agents(self, db: Session = None) -> List[Dict[str, Any]]:
        """
        List all active agents with their metadata.

        UAP Edition: Always returns SKILL.md agents even if DB is unreachable.
        DB agents take priority; SKILL.md fills in any gaps.
        """
        catalog = []
        seen_ids = set()

        # ── 1. Try DB (best-effort — never crash if DB is down) ──────────────
        should_close_db = False
        _db = db
        try:
            if _db is None:
                _db = SessionLocal()
                should_close_db = True

            query = (
                _db.query(Agent)
                .options(
                    joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
                )
                .filter(Agent.status == StatusEnum.active)
            )
            for agent in query.all():
                serialized = self._serialize_agent(agent)
                catalog.append(serialized)
                seen_ids.add(agent.id)

        except Exception as e:
            logger.warning(
                f"DB unavailable — agent list will be built from SKILL.md only. "
                f"Error: {e}"
            )
        finally:
            if should_close_db and _db is not None:
                try:
                    _db.close()
                except Exception:
                    pass

        # ── 2. Always add SKILL.md agents not already in the catalog ─────────
        try:
            skills = self._load_skill_configs()
            for agent_id, skill_config in skills.items():
                if agent_id not in seen_ids:
                    catalog.append({
                        "id": agent_id,
                        "name": skill_config["name"],
                        "description": skill_config["description"],
                        "capabilities": [],
                        "price_per_call_usd": None,
                        "endpoints": [],  # UAP: endpoints are standardized
                        "type": "http_rest",
                        "connection_config": {
                            "base_url": f"http://{skill_config['host']}:{skill_config['port']}"
                        },
                    })
        except Exception as e:
            logger.error(f"Failed to load SKILL.md agents: {e}")

        if not catalog:
            logger.warning("No agents found (DB down and no SKILL.md files loaded)")

        return catalog

    def get_agent(self, agent_id: str, db: Session = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve details for a specific agent.
        """
        agent_dict = None
        should_close_db = False
        _db = db
        try:
            if _db is None:
                _db = SessionLocal()
                should_close_db = True

            agent = (
                _db.query(Agent)
                .options(
                    joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
                )
                .filter(Agent.id == agent_id)
                .first()
            )

            if agent:
                agent_dict = self._serialize_agent(agent)
        except Exception as e:
            logger.warning(f"DB unavailable or failed to get agent {agent_id}: {e}")
        finally:
            if should_close_db and _db is not None:
                try:
                    _db.close()
                except Exception:
                    pass

        if agent_dict:
            return agent_dict

        # Fallback: check SKILL.md
        skill = self.get_agent_skill(agent_id)
        if skill:
            return {
                "id": agent_id,
                "name": skill["name"],
                "description": skill["description"],
                "capabilities": [],
                "price_per_call_usd": None,
                "endpoints": [],
                "type": "http_rest",
                "connection_config": {
                    "base_url": f"http://{skill['host']}:{skill['port']}"
                },
            }

        return None

    def validate_capability(self, agent_id: str, task_description: str) -> bool:
        """
        Validate if an agent can perform a task based on its capabilities.
        Currently a placeholder for semantic validation logic.
        """
        # In the future, this could use vector search against AgentCapability
        return True

    def get_agent_url(
        self, agent_id: str, agent_name: str = None, db: Session = None
    ) -> Optional[str]:
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
                    base_url = config.get("base_url") or config.get("url")
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
        return "json"

    def _serialize_agent(self, agent: Agent) -> Dict[str, Any]:
        """Convert Agent ORM object to dictionary"""
        endpoints_info = []
        for ep in agent.endpoints:
            endpoints_info.append(
                {
                    "endpoint": ep.endpoint,
                    "http_method": ep.http_method,
                    "description": ep.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.param_type,
                            "description": p.description,
                            "required": p.required,
                        }
                        for p in ep.parameters
                    ],
                }
            )

        return {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "price_per_call_usd": agent.price_per_call_usd,
            "endpoints": endpoints_info,
            "type": agent.agent_type,
            "connection_config": agent.connection_config,
        }

    # =================================================================
    # DB SYNC (consolidated from manage.py)
    # =================================================================

    @staticmethod
    def create_tables():
        """Create all database tables if they don't exist."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully")

    def sync_skill_entries(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Sync all SKILL.md files to the agents table.
        Creates minimal agent entries (UAP uses fixed endpoints).

        Returns:
            dict with added, updated, unchanged counts and errors list
        """
        if verbose:
            logger.info("=" * 60)
            logger.info("UAP Skill Sync: Starting...")
            logger.info("=" * 60)

        self.create_tables()

        added_count = 0
        updated_count = 0
        unchanged_count = 0
        errors = []

        backend_dir = Path(__file__).resolve().parents[1]
        agents_dir = backend_dir / "agents"

        for agent_subdir in AGENT_DIRS:
            skill_path = agents_dir / agent_subdir / "SKILL.md"

            if not skill_path.exists():
                if verbose:
                    logger.warning(f"⚠️  No SKILL.md found in {agent_subdir}")
                continue

            try:
                skill_config = parse_skill_md(skill_path)
                if not skill_config:
                    errors.append(f"Failed to parse {skill_path}")
                    continue

                agent_id = skill_config["id"]
                agent_name = skill_config["name"]

                with SessionLocal() as db:
                    db_agent = db.query(Agent).get(agent_id)

                    if db_agent is None:
                        if verbose:
                            logger.info(f"➕ Adding from SKILL.md: {agent_name}")

                        db_agent = Agent(
                            id=agent_id,
                            owner_id="orbimesh",
                            name=agent_name,
                            description=skill_config["description"],
                            capabilities=[],
                            price_per_call_usd=0.0,
                            status=StatusEnum.active,
                            agent_type=AgentType.HTTP_REST.value,
                            connection_config={
                                "base_url": f"http://{skill_config['host']}:{skill_config['port']}"
                            },
                            requires_credentials=False,
                        )
                        db.add(db_agent)
                        db.commit()
                        added_count += 1
                    else:
                        new_url = f"http://{skill_config['host']}:{skill_config['port']}"
                        current_url = (db_agent.connection_config or {}).get("base_url", "")

                        if (
                            db_agent.name != agent_name
                            or db_agent.description != skill_config["description"]
                            or current_url != new_url
                        ):
                            if verbose:
                                logger.info(f"🔄 Updating from SKILL.md: {agent_name}")
                            db_agent.name = agent_name
                            db_agent.description = skill_config["description"]
                            db_agent.connection_config = {"base_url": new_url}
                            db.commit()
                            updated_count += 1
                        else:
                            if verbose:
                                logger.info(f"✅ Up-to-date (SKILL.md): {agent_name}")
                            unchanged_count += 1

            except Exception as e:
                error_msg = f"Failed to sync {agent_subdir}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                errors.append(error_msg)

        if verbose:
            logger.info("=" * 60)
            logger.info("UAP Skill Sync: Complete")
            logger.info("=" * 60)
            logger.info(f"➕ New agents added: {added_count}")
            logger.info(f"🔄 Agents updated: {updated_count}")
            logger.info(f"✅ Agents unchanged: {unchanged_count}")
            if errors:
                logger.error(f"❌ Errors: {len(errors)}")
            logger.info("=" * 60)

        # Invalidate cache after sync so runtime picks up new agents
        self._skill_cache = {}

        return {
            "added": added_count,
            "updated": updated_count,
            "unchanged": unchanged_count,
            "errors": errors,
        }


# Global singleton
agent_registry = AgentRegistryService()


# Module-level convenience functions for clean imports
def create_tables():
    """Create all database tables."""
    AgentRegistryService.create_tables()


def sync_skill_entries(verbose: bool = True) -> Dict[str, Any]:
    """Sync SKILL.md files to DB. Delegates to the global singleton."""
    return agent_registry.sync_skill_entries(verbose=verbose)

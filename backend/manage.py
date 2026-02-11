#!/usr/bin/env python3
"""
Agent Management Script

UAP Edition: Syncs agent definitions from SKILL.md files (preferred) or 
legacy agent_entries/*.json to the database.
"""

import sys
import json
import os
import re
import yaml
from pathlib import Path
from sqlalchemy.orm import Session, joinedload
from database import SessionLocal, engine, Base
from models import Agent, AgentEndpoint, AgentCapability, EndpointParameter, StatusEnum, AgentType
from backend.schemas import AgentCard
import logging

# Import schema validation
try:
    from agent_schemas import validate_agent_schema, validate_agent_file
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
AGENT_ENTRIES_DIR = Path(__file__).parent / "agent_entries"
AGENTS_DIR = Path(__file__).parent / "agents"

# UAP: Agent subdirectories that contain SKILL.md files
AGENT_DIRS = [
    "spreadsheet_agent",
    "mail_agent", 
    "browser_agent",
    "document_agent_lib",
    "zoho_books",
]

# Lazy load embedding model
_embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-mpnet-base-v2')
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.warning("Continuing without embeddings - agent will still register but semantic search may not work")
            _embedding_model = None
    return _embedding_model


# =============================================================================
# UAP: SKILL.md PARSING AND SYNC
# =============================================================================

def parse_skill_md(skill_path: Path) -> dict:
    """
    Parse a SKILL.md file and extract agent configuration.
    
    Returns:
        Dict with id, name, port, version, host, description
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
            
        agent_id = config.get('id')
        if not agent_id:
            logger.warning(f"Missing 'id' in {skill_path}")
            return None
        
        # Extract first paragraph as description
        lines = body.strip().split('\n')
        description_lines = []
        in_description = False
        
        for line in lines:
            if line.startswith('# '):
                in_description = True
                continue
            if line.startswith('#'):
                break
            if in_description and line.strip():
                description_lines.append(line.strip())
            if in_description and not line.strip() and description_lines:
                break
        
        description = ' '.join(description_lines) if description_lines else config.get('name', agent_id)
            
        return {
            "id": agent_id,
            "name": config.get('name', agent_id),
            "port": config.get('port', 8000),
            "version": config.get('version', '1.0.0'),
            "host": config.get('host', 'localhost'),
            "description": description,
            "skill_text": body.strip(),
        }
        
    except Exception as e:
        logger.error(f"Failed to parse {skill_path}: {e}")
        return None


def sync_skill_entries(verbose: bool = True):
    """
    UAP: Sync all SKILL.md files to the agents table.
    
    This creates minimal agent entries without endpoints (UAP uses fixed endpoints).
    
    Returns:
        dict: Summary of sync operation
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("UAP Skill Sync: Starting...")
        logger.info("=" * 60)
    
    # Ensure tables exist
    create_tables()
    
    added_count = 0
    updated_count = 0
    unchanged_count = 0
    errors = []
    
    for agent_subdir in AGENT_DIRS:
        skill_path = AGENTS_DIR / agent_subdir / 'SKILL.md'
        
        if not skill_path.exists():
            if verbose:
                logger.warning(f"⚠️  No SKILL.md found in {agent_subdir}")
            continue
        
        try:
            skill_config = parse_skill_md(skill_path)
            if not skill_config:
                errors.append(f"Failed to parse {skill_path}")
                continue
            
            agent_id = skill_config['id']
            agent_name = skill_config['name']
            
            with SessionLocal() as db:
                db_agent = db.query(Agent).get(agent_id)
                
                if db_agent is None:
                    # Create new agent
                    if verbose:
                        logger.info(f"➕ Adding from SKILL.md: {agent_name}")
                    
                    db_agent = Agent(
                        id=agent_id,
                        owner_id="orbimesh",
                        name=agent_name,
                        description=skill_config['description'],
                        capabilities=[],  # UAP: capabilities in SKILL.md text
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
                    # Check if update needed
                    new_url = f"http://{skill_config['host']}:{skill_config['port']}"
                    current_url = (db_agent.connection_config or {}).get('base_url', '')
                    
                    if (db_agent.name != agent_name or 
                        db_agent.description != skill_config['description'] or
                        current_url != new_url):
                        
                        if verbose:
                            logger.info(f"🔄 Updating from SKILL.md: {agent_name}")
                        
                        db_agent.name = agent_name
                        db_agent.description = skill_config['description']
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
    
    return {
        "added": added_count,
        "updated": updated_count,
        "unchanged": unchanged_count,
        "errors": errors
    }


def create_tables():
    """Create all database tables if they don't exist."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ All tables created successfully")

def is_agent_changed(db_agent: Agent, file_agent: AgentCard) -> bool:
    """
    Compare database Agent with AgentCard from file to detect changes.
    Returns True if update is needed.
    """
    # Compare simple attributes
    if (db_agent.name != file_agent.name or
        db_agent.description != file_agent.description or
        db_agent.owner_id != file_agent.owner_id or
        db_agent.status.value != file_agent.status or
        db_agent.price_per_call_usd != file_agent.price_per_call_usd or
        db_agent.public_key_pem != file_agent.public_key_pem):
        return True
    
    # Compare agent_type
    file_agent_type = getattr(file_agent, 'agent_type', 'http_rest')
    if db_agent.agent_type != file_agent_type:
        return True
    
    # Compare connection_config
    file_connection_config = getattr(file_agent, 'connection_config', None)
    if db_agent.connection_config != file_connection_config:
        return True

    # Compare capabilities
    db_capabilities = {cap.capability_text for cap in db_agent.capability_vectors}
    file_capabilities = set(file_agent.capabilities)
    if db_capabilities != file_capabilities:
        return True

    # Compare endpoints and parameters (including request_format)
    file_endpoints_set = set()
    for ep in file_agent.endpoints:
        params_tuple = tuple(
            sorted(
                (p.name, p.param_type, p.required, p.default_value if hasattr(p, 'default_value') else None) 
                for p in ep.parameters
            )
        )
        # Include request_format in comparison
        request_format = getattr(ep, 'request_format', None)
        file_endpoints_set.add((str(ep.endpoint), ep.http_method, ep.description, request_format, params_tuple))

    db_endpoints_set = set()
    for ep in db_agent.endpoints:
        params_tuple = tuple(
            sorted(
                (p.name, p.param_type, p.required, p.default_value) 
                for p in ep.parameters
            )
        )
        # Include request_format in comparison
        db_endpoints_set.add((ep.endpoint, ep.http_method, ep.description, ep.request_format, params_tuple))

    if db_endpoints_set != file_endpoints_set:
        return True

    return False

def sync_agent_to_db(db: Session, agent_data: dict, is_new: bool = False):
    """
    Sync a single agent to the database.
    
    Args:
        db: Database session
        agent_data: Agent data from JSON file
        is_new: Whether this is a new agent or an update
    """
    agent_id = agent_data['id']
    
    # Get or create agent
    agent = db.query(Agent).get(agent_id)
    
    if agent is None:
        agent = Agent(id=agent_id)
        db.add(agent)
        is_new = True
    
    # Update agent fields
    agent.owner_id = agent_data['owner_id']
    agent.name = agent_data['name']
    agent.description = agent_data.get('description', '')
    agent.capabilities = agent_data.get('capabilities', [])
    agent.price_per_call_usd = agent_data.get('price_per_call_usd', 0.0)
    agent.status = StatusEnum(agent_data.get('status', 'active'))
    agent.public_key_pem = agent_data.get('public_key_pem')
    agent.agent_type = agent_data.get('agent_type', AgentType.HTTP_REST.value)
    agent.connection_config = agent_data.get('connection_config')
    agent.requires_credentials = agent_data.get('requires_credentials', False)
    agent.credential_fields = agent_data.get('credential_fields', [])
    
    db.flush()
    
    # Delete old capabilities, endpoints, and parameters
    # Must delete parameters first due to foreign key constraints - using subquery for robustness
    db.query(EndpointParameter).filter(
        EndpointParameter.endpoint_id.in_(
            db.query(AgentEndpoint.id).filter(AgentEndpoint.agent_id == agent_id)
        )
    ).delete(synchronize_session=False)

    # Now delete endpoints
    db.query(AgentEndpoint).filter_by(agent_id=agent_id).delete(synchronize_session=False)

    db.query(AgentCapability).filter_by(agent_id=agent_id).delete(synchronize_session=False)
    
    # SKIP: Capabilities system temporarily disabled
    # The capabilities field exists in JSON as empty dict for future use
    # but is not synced to database to avoid validation issues
    # TODO: Re-enable when capabilities redesign is complete
    
    # Add endpoints and parameters
    for endpoint_data in agent_data.get('endpoints', []):
        endpoint = AgentEndpoint(
            agent_id=agent_id,
            endpoint=endpoint_data['endpoint'],
            http_method=endpoint_data['http_method'],
            description=endpoint_data.get('description', ''),
            request_format=endpoint_data.get('request_format')  # 'json' or 'form'
        )
        db.add(endpoint)
        db.flush()
        
        # Add parameters
        for param_data in endpoint_data.get('parameters', []):
            parameter = EndpointParameter(
                endpoint_id=endpoint.id,
                name=param_data['name'],
                description=param_data.get('description', ''),
                param_type=param_data['param_type'],
                required=param_data.get('required', True),
                default_value=param_data.get('default_value')
            )
            db.add(parameter)
    
    db.commit()
    return is_new

def sync_agent_entries(verbose: bool = True):
    """
    Sync all agent entries from agent_entries/*.json to the database.
    Non-destructive: only adds new agents or updates changed ones.
    
    Returns:
        dict: Summary of sync operation
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Agent Sync: Starting...")
        logger.info("=" * 60)
    
    # Ensure tables exist
    create_tables()
    
    # Check if agent_entries directory exists
    if not AGENT_ENTRIES_DIR.is_dir():
        logger.error(f"❌ Directory not found: {AGENT_ENTRIES_DIR}")
        return {"error": "agent_entries directory not found"}
    
    # Find all JSON files
    agent_files = list(AGENT_ENTRIES_DIR.glob("*.json"))
    
    if not agent_files:
        if verbose:
            logger.warning(f"⚠️  No agent JSON files found in {AGENT_ENTRIES_DIR}")
        return {"added": 0, "updated": 0, "unchanged": 0}
    
    if verbose:
        logger.info(f"Found {len(agent_files)} agent definition(s)")
    
    added_count = 0
    updated_count = 0
    unchanged_count = 0
    errors = []
    
    for json_file in agent_files:
        try:
            # Validate agent schema first
            if SCHEMA_VALIDATION_AVAILABLE and verbose:
                is_valid, error_msg, validated_schema = validate_agent_file(str(json_file))
                if not is_valid:
                    logger.warning(f"⚠️  Schema validation warning for {json_file.name}: {error_msg}")
                    logger.warning("    Proceeding with sync anyway (backward compatibility)")
            
            # Load agent data
            with open(json_file, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)
            
            agent_id = agent_data.get('id')
            agent_name = agent_data.get('name', agent_id)
            
            # Create session for this agent
            with SessionLocal() as db:
                # Check if agent exists
                db_agent = db.query(Agent).options(
                    joinedload(Agent.capability_vectors),
                    joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
                ).get(agent_id)
                
                if db_agent:
                    # Check if update needed
                    file_agent = AgentCard(**agent_data)
                    if is_agent_changed(db_agent, file_agent):
                        if verbose:
                            logger.info(f"🔄 Updating: {agent_name}")
                        sync_agent_to_db(db, agent_data, is_new=False)
                        updated_count += 1
                    else:
                        if verbose:
                            logger.info(f"✅ Up-to-date: {agent_name}")
                        unchanged_count += 1
                else:
                    # New agent
                    if verbose:
                        logger.info(f"➕ Adding: {agent_name}")
                    sync_agent_to_db(db, agent_data, is_new=True)
                    added_count += 1
                    
        except Exception as e:
            error_msg = f"Failed to sync {json_file.name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            errors.append(error_msg)
    
    # Summary
    if verbose:
        logger.info("=" * 60)
        logger.info("Agent Sync: Complete")
        logger.info("=" * 60)
        logger.info(f"➕ New agents added: {added_count}")
        logger.info(f"🔄 Agents updated: {updated_count}")
        logger.info(f"✅ Agents unchanged: {unchanged_count}")
        if errors:
            logger.error(f"❌ Errors: {len(errors)}")
            for error in errors:
                logger.error(f"   - {error}")
        logger.info("=" * 60)
    
    return {
        "added": added_count,
        "updated": updated_count,
        "unchanged": unchanged_count,
        "errors": errors
    }

def validate_agent(agent_id: str, verbose: bool = True):
    """
    Validate a single agent schema.
    
    Args:
        agent_id: Agent ID (filename without .json)
        verbose: Print detailed output
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not SCHEMA_VALIDATION_AVAILABLE:
        logger.error("❌ Schema validation not available (agent_schemas module not found)")
        return False
    
    agent_file = AGENT_ENTRIES_DIR / f"{agent_id}.json"
    
    if not agent_file.exists():
        logger.error(f"❌ Agent file not found: {agent_file}")
        return False
    
    if verbose:
        logger.info("=" * 60)
        logger.info(f"Validating agent: {agent_id}")
        logger.info("=" * 60)
    
    is_valid, error_msg, validated_schema = validate_agent_file(str(agent_file))
    
    if is_valid:
        if verbose:
            logger.info(f"✅ Agent schema is valid!")
            logger.info(f"   Name: {validated_schema.name}")
            logger.info(f"   Type: {validated_schema.agent_type.value}")
            logger.info(f"   Version: {validated_schema.version}")
            logger.info(f"   Categories: {len(validated_schema.capabilities.categories)}")
            logger.info("=" * 60)
        return True
    else:
        logger.error(f"❌ Validation failed:")
        logger.error(f"   {error_msg}")
        logger.info("=" * 60)
        return False


def validate_all_agents(verbose: bool = True):
    """
    Validate all agent schemas.
    
    Args:
        verbose: Print detailed output
        
    Returns:
        dict: Summary of validation
    """
    if not SCHEMA_VALIDATION_AVAILABLE:
        logger.error("❌ Schema validation not available (agent_schemas module not found)")
        return {"valid": 0, "invalid": 0, "errors": ["Schema validation module not found"]}
    
    if verbose:
        logger.info("=" * 60)
        logger.info("Validating all agents...")
        logger.info("=" * 60)
    
    # Find all JSON files (excluding templates)
    agent_files = [f for f in AGENT_ENTRIES_DIR.glob("*.json") if not f.parent.name == "templates"]
    
    if not agent_files:
        logger.warning(f"⚠️  No agent JSON files found in {AGENT_ENTRIES_DIR}")
        return {"valid": 0, "invalid": 0, "errors": []}
    
    valid_count = 0
    invalid_count = 0
    errors = []
    
    for json_file in agent_files:
        is_valid, error_msg, validated_schema = validate_agent_file(str(json_file))
        
        if is_valid:
            if verbose:
                logger.info(f"✅ {json_file.name}")
            valid_count += 1
        else:
            if verbose:
                logger.error(f"❌ {json_file.name}")
                logger.error(f"   {error_msg}")
            invalid_count += 1
            errors.append(f"{json_file.name}: {error_msg}")
    
    # Summary
    if verbose:
        logger.info("=" * 60)
        logger.info("Validation Summary")
        logger.info("=" * 60)
        logger.info(f"✅ Valid: {valid_count}")
        logger.info(f"❌ Invalid: {invalid_count}")
        if errors:
            logger.info("\nErrors:")
            for error in errors:
                logger.error(f"   - {error}")
        logger.info("=" * 60)
    
    return {
        "valid": valid_count,
        "invalid": invalid_count,
        "errors": errors
    }


def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage agent database synchronization and validation")
    parser.add_argument(
        'action',
        choices=['sync', 'sync-skills', 'sync-all', 'create-tables', 'validate', 'validate-all'],
        nargs='?',
        default='sync-all',
        help='Action to perform. sync-all (default) syncs both SKILL.md and JSON files.'
    )
    parser.add_argument(
        'agent_id',
        nargs='?',
        help='Agent ID for validate action (e.g., mail_agent)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    if args.action == 'sync':
        # Legacy: sync JSON files only
        result = sync_agent_entries(verbose=not args.quiet)
        if result.get('errors'):
            sys.exit(1)
    elif args.action == 'sync-skills':
        # UAP: sync SKILL.md files only
        result = sync_skill_entries(verbose=not args.quiet)
        if result.get('errors'):
            sys.exit(1)
    elif args.action == 'sync-all':
        # Sync both SKILL.md (preferred) and JSON files
        if not args.quiet:
            logger.info("Syncing SKILL.md files (UAP)...")
        result1 = sync_skill_entries(verbose=not args.quiet)
        if not args.quiet:
            logger.info("\nSyncing legacy JSON files...")
        result2 = sync_agent_entries(verbose=not args.quiet)
        if result1.get('errors') or result2.get('errors'):
            sys.exit(1)
    elif args.action == 'create-tables':
        create_tables()
    elif args.action == 'validate':
        if not args.agent_id:
            logger.error("❌ Agent ID required for validate action")
            logger.error("Usage: python manage.py validate <agent_id>")
            sys.exit(1)
        is_valid = validate_agent(args.agent_id, verbose=not args.quiet)
        sys.exit(0 if is_valid else 1)
    elif args.action == 'validate-all':
        result = validate_all_agents(verbose=not args.quiet)
        sys.exit(0 if result['invalid'] == 0 else 1)

if __name__ == "__main__":
    main()

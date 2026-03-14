"""
Skill Registry Service

Manages agent skill definitions from SKILL.md files.
Progressive disclosure: only frontmatter loaded at startup,
full body loaded lazily on first dispatch.

Matching uses hybrid scoring:
  1. MPNet 768-dim cosine similarity (semantic)
  2. Trigger keyword matching (exact)
  3. not_for penalty

Replaces agent_registry_service.get_all_skills_context() with a
token-efficient alternative.
"""

import re
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import yaml
import numpy as np

logger = logging.getLogger("SkillRegistry")


# ──────────────────────────────────────────────────────────────────────────
# SHARED EMBEDDING MODEL (same as artifact_store — lazy loaded once)
# ──────────────────────────────────────────────────────────────────────────

_embed_lock = threading.Lock()
_embed_model = None


def _get_embed_model():
    """Lazily load the sentence-transformer model exactly once (thread-safe)."""
    global _embed_model
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _embed_model = SentenceTransformer("all-mpnet-base-v2")
                    logger.info("SkillRegistry: Loaded embedding model all-mpnet-base-v2")
                except ImportError:
                    logger.warning("sentence_transformers not installed -- semantic matching disabled")
    return _embed_model


def _embed_text(text: str) -> Optional[np.ndarray]:
    """Embed text into a 768-dim normalized vector. Returns None if unavailable."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        return model.encode(text, normalize_embeddings=True)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None

# Agent directories to scan
AGENTS_DIR = Path(__file__).resolve().parents[1] / "agents"


@dataclass
class SkillConfig:
    """Parsed SKILL.md frontmatter — lightweight metadata for routing."""
    id: str
    name: str
    description: str
    port: int = 8000
    version: str = "1.0.0"
    model: str = "cerebras/llama-3.3-70b"
    context_strategy: str = "standard"        # minimal | standard | full
    requires_auth: bool = False
    composio_app_slug: Optional[str] = None   # Composio toolkit slug e.g. "gmail", "zohobooks"
    deprecated: bool = False
    prefer: Optional[str] = None              # preferred replacement agent
    triggers: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    not_for: List[str] = field(default_factory=list)
    # Internal
    _skill_path: Optional[Path] = field(default=None, repr=False)


@dataclass
class SkillMatch:
    """Result of matching a user prompt against skills."""
    skill_id: str
    skill_name: str
    score: float           # 0.0–1.0 relevance score
    match_reasons: List[str]   # which triggers / description matched


class SkillRegistry:
    """
    Manages agent skill definitions from SKILL.md files.
    
    Design principles (inspired by Claude Code):
    - Only frontmatter loaded at startup (~50 bytes per agent)  
    - Full body text lazy-loaded on first dispatch
    - Compact summary for Brain context (~200 tokens for all agents)
    - Trigger-based matching for fast agent selection
    """

    def __init__(self, agents_dir: Optional[Path] = None):
        self._agents_dir = agents_dir or AGENTS_DIR
        self._skills: Dict[str, SkillConfig] = {}       # id -> config (frontmatter)
        self._body_cache: Dict[str, str] = {}            # id -> full body (lazy)
        self._embeddings: Dict[str, np.ndarray] = {}     # id -> 768-dim vector
        self._initialized = False

    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Scan all SKILL.md files and parse frontmatter only."""
        if self._initialized:
            return

        if not self._agents_dir.exists():
            logger.warning(f"Agents directory not found: {self._agents_dir}")
            self._initialized = True
            return

        count = 0
        for agent_dir in sorted(self._agents_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            skill_path = agent_dir / "SKILL.md"
            if skill_path.exists():
                config = self._parse_frontmatter(skill_path)
                if config:
                    self._skills[config.id] = config
                    count += 1
                    logger.debug(f"Loaded skill: {config.id} ({len(config.triggers)} triggers)")

        self._initialized = True
        logger.info(f"SkillRegistry initialized: {count} skills loaded")

        # Pre-compute embeddings for all skills (runs in background)
        self._precompute_embeddings()

    def _parse_frontmatter(self, skill_path: Path) -> Optional[SkillConfig]:
        """Parse only the YAML frontmatter from a SKILL.md file."""
        try:
            content = skill_path.read_text(encoding="utf-8")

            # Extract YAML frontmatter between --- markers
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if not match:
                logger.warning(f"No frontmatter in {skill_path}")
                return None

            data = yaml.safe_load(match.group(1))
            if not data or not data.get("id"):
                return None

            # Parse description — handle YAML multi-line strings
            description = data.get("description", "")
            if isinstance(description, str):
                description = " ".join(description.split())  # normalize whitespace

            return SkillConfig(
                id=data["id"],
                name=data.get("name", data["id"]),
                description=description,
                port=data.get("port", 8000),
                version=data.get("version", "1.0.0"),
                model=data.get("model", "cerebras/llama-3.3-70b"),
                context_strategy=data.get("context_strategy", "standard"),
                requires_auth=data.get("requires_auth", False),
                composio_app_slug=data.get("composio_app_slug"),
                deprecated=data.get("deprecated", False),
                prefer=data.get("prefer"),
                triggers=data.get("triggers", []),
                capabilities=data.get("capabilities", []),
                not_for=data.get("not_for", []),
                _skill_path=skill_path,
            )

        except Exception as e:
            logger.error(f"Failed to parse {skill_path}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # EMBEDDING PRE-COMPUTATION
    # ──────────────────────────────────────────────────────────────────────

    def _precompute_embeddings(self) -> None:
        """Pre-compute 768-dim embeddings for all skills (runs at init)."""
        model = _get_embed_model()
        if model is None:
            logger.info("SkillRegistry: No embedding model -- using keyword-only matching")
            return

        texts = []
        ids = []
        for skill_id, config in self._skills.items():
            if config.deprecated:
                continue
            # Build rich text from description + triggers + capabilities
            rich_text = " ".join([
                config.description,
                " ".join(config.triggers),
                " ".join(config.capabilities),
                config.name,
            ])
            texts.append(rich_text)
            ids.append(skill_id)

        if not texts:
            return

        try:
            # Batch encode all skills at once (much faster than one-by-one)
            vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
            for skill_id, vec in zip(ids, vectors):
                self._embeddings[skill_id] = vec
            logger.info(f"SkillRegistry: Pre-computed embeddings for {len(self._embeddings)} skills")
        except Exception as e:
            logger.warning(f"SkillRegistry: Embedding pre-computation failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # MATCHING — Hybrid semantic + keyword scoring
    # ──────────────────────────────────────────────────────────────────────

    def match_skills(self, prompt: str, top_k: int = 3) -> List[SkillMatch]:
        """
        Match a user prompt to skills using hybrid scoring:
          1. MPNet cosine similarity (0.6 weight) -- semantic understanding
          2. Trigger keyword matching (0.3 weight) -- exact intent signals
          3. Capability overlap (0.1 weight) -- feature matching
          4. not_for penalty (-0.5) -- exclusion signals

        Falls back to keyword-only if embeddings unavailable.
        Returns ranked list, highest score first.
        """
        self.initialize()

        prompt_lower = prompt.lower()
        prompt_words = set(prompt_lower.split())
        matches: List[SkillMatch] = []

        # Embed the user prompt once
        prompt_vec = _embed_text(prompt)
        has_embeddings = prompt_vec is not None and len(self._embeddings) > 0

        for skill_id, config in self._skills.items():
            # Skip deprecated agents -- redirect to preferred
            if config.deprecated and config.prefer:
                continue

            score = 0.0
            reasons: List[str] = []

            # 1. SEMANTIC SIMILARITY (0.6 weight)
            if has_embeddings and skill_id in self._embeddings:
                cos_sim = float(np.dot(prompt_vec, self._embeddings[skill_id]))
                # cos_sim is in [-1, 1] for normalized vectors; clamp to [0, 1]
                cos_sim = max(0.0, cos_sim)
                semantic_score = cos_sim * 0.6
                score += semantic_score
                if cos_sim > 0.2:
                    reasons.append(f"semantic: {cos_sim:.2f}")
            else:
                # Fallback: description word overlap (if no embeddings)
                desc_words = set(config.description.lower().split())
                filler = {"a", "an", "the", "for", "and", "or", "to", "with", "of", "in", "on", "is", "are"}
                meaningful = (prompt_words & desc_words) - filler
                if meaningful:
                    score += min(len(meaningful) / 5.0, 1.0) * 0.3
                    reasons.append(f"keywords: {list(meaningful)[:3]}")

            # 2. TRIGGER MATCHING (0.3 weight -- exact intent)
            trigger_hits = 0
            for trigger in config.triggers:
                if trigger.lower() in prompt_lower:
                    trigger_hits += 1
                    reasons.append(f"trigger: '{trigger}'")
            if config.triggers:
                trigger_score = min(trigger_hits / max(len(config.triggers), 1), 1.0) * 0.3
                score += trigger_score

            # 3. CAPABILITY OVERLAP (0.1 weight)
            for cap in config.capabilities:
                cap_words = set(cap.lower().replace("_", " ").split())
                if cap_words & prompt_words:
                    score += 0.1
                    reasons.append(f"capability: '{cap}'")
                    break

            # 4. NOT-FOR PENALTY
            for nf in config.not_for:
                if nf.lower() in prompt_lower:
                    score -= 0.5
                    reasons.append(f"not_for: '{nf}'")

            if score > 0.01:
                matches.append(SkillMatch(
                    skill_id=skill_id,
                    skill_name=config.name,
                    score=round(score, 3),
                    match_reasons=reasons,
                ))

        # Sort by score descending, take top_k
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    # ──────────────────────────────────────────────────────────────────────
    # CONTEXT GENERATION — For Brain prompt
    # ──────────────────────────────────────────────────────────────────────

    def get_skill_summary(self) -> str:
        """
        Compact summary for Brain context (~200 tokens for all agents).
        Only includes: name, one-line description, trigger keywords.
        
        This replaces agent_registry.get_all_skills_context() which dumped
        the full SKILL.md body for every agent (~5000+ tokens).
        """
        self.initialize()

        lines = []
        for skill_id, config in self._skills.items():
            if config.deprecated:
                continue  # Don't show deprecated agents to Brain

            # One compact line per agent
            triggers_str = ", ".join(config.triggers[:5]) if config.triggers else "general"
            lines.append(
                f"- **{config.name}** (id: `{skill_id}`): "
                f"{config.description[:100]}... "
                f"[triggers: {triggers_str}]"
            )

        return "\n".join(lines) if lines else "No agents available."

    def get_skill_context(self, agent_id: str) -> str:
        """
        Full SKILL.md body for a specific agent (lazy-loaded).
        Used when Brain has decided to dispatch to this agent and needs 
        the full instructions for context.
        """
        self.initialize()

        # Return from cache if available
        if agent_id in self._body_cache:
            return self._body_cache[agent_id]

        config = self._skills.get(agent_id)
        if not config or not config._skill_path:
            return ""

        try:
            content = config._skill_path.read_text(encoding="utf-8")
            # Extract body (everything after frontmatter)
            match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
            body = content[match.end():] if match else content
            self._body_cache[agent_id] = body.strip()
            return self._body_cache[agent_id]
        except Exception as e:
            logger.error(f"Failed to load skill body for {agent_id}: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────────
    # CONFIG ACCESS — For Hands/pipeline
    # ──────────────────────────────────────────────────────────────────────

    def get_skill_config(self, agent_id: str) -> Optional[SkillConfig]:
        """Get the full SkillConfig for an agent."""
        self.initialize()
        return self._skills.get(agent_id)

    def list_skills(self) -> List[SkillConfig]:
        """List all non-deprecated skill configs."""
        self.initialize()
        return [c for c in self._skills.values() if not c.deprecated]

    def get_all_configs(self) -> Dict[str, SkillConfig]:
        """Get all skill configs (including deprecated)."""
        self.initialize()
        return dict(self._skills)

    def reload(self) -> None:
        """Force reload all skills and re-compute embeddings."""
        self._skills.clear()
        self._body_cache.clear()
        self._embeddings.clear()
        self._initialized = False
        self.initialize()


# =============================================================================
# STARTUP PRE-LOADING (Background initialization)
# =============================================================================

def _preload_embedding_model_async():
    """
    Pre-load embedding model in background after server starts.
    
    Thread-safe: Uses existing double-checked locking in _get_embed_model().
    Benefit: First skill matching request is fast if sent >14s after startup.
    """
    def _load():
        try:
            _get_embed_model()
            logger.info("✅ SkillRegistry: Embedding model pre-loaded (background)")
        except Exception as e:
            logger.warning(f"⚠️ SkillRegistry: Embedding model pre-load failed: {e}")
    
    # Daemon thread: won't block server shutdown
    threading.Thread(target=_load, daemon=True, name="SkillEmbeddingPreload").start()
    logger.info("🔄 SkillRegistry: Embedding model pre-load started (background)")


# Trigger pre-loading at module import time (server startup)
_preload_embedding_model_async()


# ══════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════

skill_registry = SkillRegistry()

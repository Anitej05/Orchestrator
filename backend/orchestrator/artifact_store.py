"""
Artifact Store — Persistent Knowledge & Experience System

Captures, stores, retrieves, and distills knowledge from orchestrator interactions.
Gets smarter over time as more tasks are completed.

Artifact Types:
    - task_result: Completed task outputs with metadata
    - knowledge: Distilled learnings (agent patterns, error recovery, tips)
    - playbook: Reusable multi-step workflows
    - profile: User preferences and behavioral patterns

Storage:
    storage/artifacts/{user_id}/
    ├── manifest.json       # Local index (backup/cache)
    ├── tasks/              # Task result markdown files
    ├── knowledge/          # Knowledge markdown files
    ├── playbooks/          # Playbook markdown files
    └── profile.json        # User preference accumulation

Retrieval:
    Hybrid semantic search via pgvector (cosine similarity on 768-dim embeddings)
    + keyword boost + recency decay.  O(log n) with IVFFlat index.
"""

import json
import logging
import re
import time
import hashlib
import threading
import numpy as np
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.mega_logger import setup_mega_logger

logger = setup_mega_logger("ArtifactStore")

# Centralized storage paths
from backend.storage_config import STORAGE_ROOT as STORAGE_BASE, ARTIFACTS_DIR as ARTIFACTS_BASE

# ── Lazy-loaded embedding model (shared across all stores) ───────────────────
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
                    logger.info("🧠 Loaded embedding model: all-mpnet-base-v2 (768-dim)")
                except ImportError:
                    logger.warning("sentence_transformers not installed — semantic search disabled")
    return _embed_model


def _embed_text(text: str) -> Optional[List[float]]:
    """Embed a text string into a 768-dim vector. Returns None if model unavailable."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


# =============================================================================
# MANIFEST ENTRY  (local cache — still useful as a quick backup/offline fallback)
# =============================================================================

class ArtifactEntry:
    """Metadata for a single artifact in the manifest."""

    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,
        tags: List[str],
        summary: str,
        file_path: str,
        created: str = None,
        last_used: str = None,
        use_count: int = 0,
        relevance_decay: float = 0.95,
        source_objective: str = "",
        source_agent: str = "",
    ):
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.tags = tags
        self.summary = summary
        self.file_path = file_path
        self.created = created or datetime.now().isoformat()
        self.last_used = last_used or self.created
        self.use_count = use_count
        self.relevance_decay = relevance_decay
        self.source_objective = source_objective
        self.source_agent = source_agent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "tags": self.tags,
            "summary": self.summary,
            "file_path": self.file_path,
            "created": self.created,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "relevance_decay": self.relevance_decay,
            "source_objective": self.source_objective,
            "source_agent": self.source_agent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactEntry":
        return cls(**data)


# =============================================================================
# KEYWORD HELPERS (kept for hybrid boost)
# =============================================================================

def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text (simple tokenizer)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "while", "where", "how", "what",
        "which", "who", "whom", "this", "that", "these", "those", "am",
        "it", "its", "my", "me", "we", "us", "you", "your", "he", "she",
        "they", "them", "his", "her", "all", "also", "about", "up",
        "i", "create", "make", "get", "use", "write", "please",
    }
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in stopwords]


# =============================================================================
# ARTIFACT STORE
# =============================================================================

class ArtifactStore:
    """
    Persistent artifact storage for a single user.

    Captures learnings (task results, knowledge, playbooks) and retrieves them
    via hybrid semantic + keyword search backed by pgvector.

    Usage:
        store = ArtifactStore("user123")
        await store.capture_from_task(history_entry, state)
        relevant = store.retrieve_relevant("analyze csv data", top_k=3)
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.base_path = ARTIFACTS_BASE / user_id
        self.manifest_path = self.base_path / "manifest.json"
        self.profile_path = self.base_path / "profile.json"

        # Ensure directories
        for subdir in ["tasks", "knowledge", "playbooks"]:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)

        # Load local manifest + profile
        self.manifest: List[ArtifactEntry] = self._load_manifest()
        self.profile: Dict[str, Any] = self._load_profile()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> List[ArtifactEntry]:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                return [ArtifactEntry.from_dict(e) for e in data]
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return []

    def _save_manifest(self):
        try:
            self.manifest_path.write_text(
                json.dumps(
                    [e.to_dict() for e in self.manifest],
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def _load_profile(self) -> Dict[str, Any]:
        if self.profile_path.exists():
            try:
                return json.loads(self.profile_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "user_id": self.user_id,
            "preferred_canvas_types": [],
            "preferred_agents": {},
            "behavioral_notes": [],
            "total_tasks_completed": 0,
            "total_conversations": 0,
        }

    def _save_profile(self):
        try:
            self.profile_path.write_text(
                json.dumps(self.profile, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    # ── Artifact ID Generation ───────────────────────────────────────────────

    def _make_id(self, content: str, prefix: str = "art") -> str:
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}_{h}"

    # ── Vector Embedding Upsert ──────────────────────────────────────────────

    def _upsert_embedding(self, entry: ArtifactEntry) -> None:
        """
        Embed the artifact summary+tags+objective and upsert into PostgreSQL
        via the ArtifactEmbedding table.  Safe: silently skips if DB or model
        is unavailable.
        """
        # Build a rich text blob that captures the artifact's semantics
        embed_text = " ".join([
            entry.summary,
            " ".join(entry.tags),
            entry.source_objective or "",
            entry.source_agent or "",
        ]).strip()

        vec = _embed_text(embed_text)
        if vec is None:
            return  # Model/library not available — degrade gracefully

        db = None
        try:
            from database import SessionLocal
            from models import ArtifactEmbedding

            db = SessionLocal()
            existing = db.query(ArtifactEmbedding).filter_by(
                artifact_id=entry.artifact_id
            ).first()

            if existing:
                existing.summary = entry.summary
                existing.embedding = vec
                existing.tags = entry.tags
                existing.source_objective = entry.source_objective or ""
                existing.source_agent = entry.source_agent or ""
                existing.file_path = entry.file_path
                existing.last_used_at = datetime.now(timezone.utc).replace(tzinfo=None)
            else:
                db.add(ArtifactEmbedding(
                    user_id=self.user_id,
                    artifact_id=entry.artifact_id,
                    artifact_type=entry.artifact_type,
                    summary=entry.summary,
                    embedding=vec,
                    tags=entry.tags,
                    source_objective=entry.source_objective or "",
                    source_agent=entry.source_agent or "",
                    file_path=entry.file_path,
                    use_count=0,
                ))
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to upsert artifact embedding: {e}")
            if db:
                try:
                    db.rollback()
                except Exception:
                    pass
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass

    # ── Capture Methods ──────────────────────────────────────────────────────

    async def capture_from_task(
        self,
        action_entry: Dict[str, Any],
        state: Dict[str, Any],
        objective: str = "",
    ):
        """
        Auto-capture artifacts from a completed task execution.
        Decides what's worth storing based on quality thresholds.
        """
        action_type = action_entry.get("action_type", "")
        resource_id = action_entry.get("resource_id", "")
        success = action_entry.get("success", False)
        result_summary = action_entry.get("result_summary", "")
        instruction = action_entry.get("instruction", "")
        exec_time = action_entry.get("execution_time_ms", 0)

        # Skip trivial results
        if len(result_summary) < 30 and action_type != "agent":
            return

        # 1. Capture task result (if substantial)
        if success and len(result_summary) > 50:
            await self._capture_task_result(
                action_type=action_type,
                resource_id=resource_id,
                instruction=instruction,
                result_summary=result_summary,
                objective=objective,
                exec_time=exec_time,
            )

        # 2. Capture routing knowledge (which resource worked)
        if success and action_type in ("agent", "tool"):
            self._capture_routing_knowledge(
                action_type=action_type,
                resource_id=resource_id,
                instruction=instruction,
                objective=objective,
            )

        # 3. Capture error patterns (what went wrong)
        if not success:
            error_msg = action_entry.get("error_message", "")
            self._capture_error_pattern(
                action_type=action_type,
                resource_id=resource_id,
                instruction=instruction,
                error_msg=error_msg,
                objective=objective,
            )

    async def _capture_task_result(
        self,
        action_type: str,
        resource_id: str,
        instruction: str,
        result_summary: str,
        objective: str,
        exec_time: float,
    ):
        """Store a meaningful task result as a markdown artifact."""
        tags = _extract_keywords(instruction)[:8]
        if resource_id:
            tags.append(resource_id.lower().replace(" ", "_"))
        tags.append(action_type)

        artifact_id = self._make_id(instruction + result_summary, "task")
        filename = f"{artifact_id}.md"
        file_path = self.base_path / "tasks" / filename

        content = f"""---
type: task_result
tags: {json.dumps(tags)}
created: {datetime.now().isoformat()}
action_type: {action_type}
resource: {resource_id or 'direct'}
execution_time_ms: {exec_time:.0f}
---
# Task Result: {instruction[:80]}

**Objective:** {objective[:200]}

**Action:** {action_type} → {resource_id or 'direct'}

**Result:**
{result_summary[:1500]}
"""
        file_path.write_text(content, encoding="utf-8")

        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type="task_result",
            tags=tags,
            summary=f"{action_type}({resource_id}): {result_summary[:100]}",
            file_path=str(file_path),
            source_objective=objective[:200],
            source_agent=resource_id,
        )
        self.manifest.append(entry)
        self._save_manifest()
        self._upsert_embedding(entry)

        logger.info(f"📦 Captured task result: {artifact_id}")

    def _capture_routing_knowledge(
        self,
        action_type: str,
        resource_id: str,
        instruction: str,
        objective: str,
    ):
        """Record which resource successfully handled which task type."""
        routing_id = "routing_patterns"
        routing_path = self.base_path / "knowledge" / "routing_patterns.md"

        # Load existing patterns
        patterns: Dict[str, Dict[str, int]] = {}
        if routing_path.exists():
            try:
                text = routing_path.read_text(encoding="utf-8")
                match = re.search(r"```json\n(.+?)\n```", text, re.DOTALL)
                if match:
                    patterns = json.loads(match.group(1))
            except Exception:
                pass

        key = f"{action_type}:{resource_id}" if resource_id else action_type
        task_keywords = _extract_keywords(instruction)[:5]
        task_label = " ".join(task_keywords) if task_keywords else instruction[:50]

        if key not in patterns:
            patterns[key] = {}
        patterns[key][task_label] = patterns[key].get(task_label, 0) + 1

        content = f"""---
type: knowledge
tags: [routing, agent_selection, resource_selection]
created: {datetime.now().isoformat()}
---
# Agent/Resource Routing Patterns

Accumulated patterns of which resources successfully handle which task types.

```json
{json.dumps(patterns, indent=2)}
```

## Summary
"""
        for resource, tasks in patterns.items():
            content += f"\n### {resource}\n"
            for task_desc, count in sorted(tasks.items(), key=lambda x: -x[1])[:5]:
                content += f"- {task_desc} ({count}x)\n"

        routing_path.write_text(content, encoding="utf-8")

        # Ensure manifest entry + embedding
        existing = next(
            (e for e in self.manifest if e.artifact_id == routing_id), None
        )
        if existing:
            existing.last_used = datetime.now().isoformat()
            existing.use_count += 1
            existing.summary = f"Routing patterns: {len(patterns)} resources tracked"
            self._upsert_embedding(existing)
        else:
            entry = ArtifactEntry(
                artifact_id=routing_id,
                artifact_type="knowledge",
                tags=["routing", "agent_selection", "resource_selection"],
                summary="Which resources work best for which task types",
                file_path=str(routing_path),
            )
            self.manifest.append(entry)
            self._upsert_embedding(entry)
        self._save_manifest()

    def _capture_error_pattern(
        self,
        action_type: str,
        resource_id: str,
        instruction: str,
        error_msg: str,
        objective: str,
    ):
        """Record error patterns for avoidance in future tasks."""
        error_path = self.base_path / "knowledge" / "error_patterns.md"

        errors: List[Dict] = []
        if error_path.exists():
            try:
                text = error_path.read_text(encoding="utf-8")
                match = re.search(r"```json\n(.+?)\n```", text, re.DOTALL)
                if match:
                    errors = json.loads(match.group(1))
            except Exception:
                pass

        errors.append({
            "action": f"{action_type}:{resource_id}",
            "instruction": instruction[:100],
            "error": error_msg[:200],
            "when": datetime.now().isoformat(),
        })
        errors = errors[-50:]

        content = f"""---
type: knowledge
tags: [errors, debugging, avoidance]
created: {datetime.now().isoformat()}
---
# Error Patterns (Avoid These)

```json
{json.dumps(errors, indent=2)}
```
"""
        error_path.write_text(content, encoding="utf-8")

        existing = next(
            (e for e in self.manifest if e.artifact_id == "error_patterns"), None
        )
        if existing:
            existing.last_used = datetime.now().isoformat()
            existing.use_count += 1
            existing.summary = f"Error patterns: {len(errors)} entries"
            self._upsert_embedding(existing)
        else:
            entry = ArtifactEntry(
                artifact_id="error_patterns",
                artifact_type="knowledge",
                tags=["errors", "debugging", "avoidance"],
                summary="Known error patterns to avoid",
                file_path=str(error_path),
            )
            self.manifest.append(entry)
            self._upsert_embedding(entry)
        self._save_manifest()

    def capture_knowledge(
        self,
        key: str,
        value: str,
        tags: List[str] = None,
        objective: str = "",
    ):
        """Manually store a knowledge artifact."""
        artifact_id = self._make_id(key + value, "know")
        filename = f"{artifact_id}.md"
        file_path = self.base_path / "knowledge" / filename

        all_tags = list(set((tags or []) + _extract_keywords(key)[:5]))

        content = f"""---
type: knowledge
tags: {json.dumps(all_tags)}
created: {datetime.now().isoformat()}
---
# {key}

{value}
"""
        file_path.write_text(content, encoding="utf-8")

        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type="knowledge",
            tags=all_tags,
            summary=f"{key}: {value[:100]}",
            file_path=str(file_path),
            source_objective=objective[:200],
        )
        self.manifest.append(entry)
        self._save_manifest()
        self._upsert_embedding(entry)
        logger.info(f"📦 Captured knowledge: {artifact_id}")

    def capture_playbook(
        self,
        name: str,
        objective: str,
        steps: List[Dict[str, Any]],
        outcome: str = "",
    ):
        """Store a reusable multi-step workflow pattern."""
        artifact_id = self._make_id(name + objective, "play")
        filename = f"{artifact_id}.md"
        file_path = self.base_path / "playbooks" / filename

        tags = _extract_keywords(objective)[:8] + ["workflow", "playbook"]

        content = f"""---
type: playbook
tags: {json.dumps(tags)}
created: {datetime.now().isoformat()}
steps: {len(steps)}
---
# Playbook: {name}

**Objective:** {objective[:300]}

**Outcome:** {outcome[:200]}

## Steps

"""
        for i, step in enumerate(steps, 1):
            action = step.get("action_type", "unknown")
            resource = step.get("resource_id", "")
            instr = step.get("instruction", "")
            success = "ok" if step.get("success") else "err"
            content += f"{i}. `{action}` → `{resource}` [{success}]\n   {instr[:100]}\n\n"

        file_path.write_text(content, encoding="utf-8")

        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type="playbook",
            tags=tags,
            summary=f"Playbook: {name} ({len(steps)} steps)",
            file_path=str(file_path),
            source_objective=objective[:200],
        )
        self.manifest.append(entry)
        self._save_manifest()
        self._upsert_embedding(entry)
        logger.info(f"📦 Captured playbook: {artifact_id} ({len(steps)} steps)")

    # ── Retrieval (HYBRID: pgvector semantic + keyword boost) ────────────────

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 2000,
        artifact_type: str = None,
    ) -> str:
        """
        Retrieve the most relevant artifacts for a query.

        Strategy:
          1. Embed the query → 768-dim vector
          2. pgvector cosine similarity search (O(log n) with index)
          3. Apply keyword boost + recency decay
          4. Return formatted string ready for Brain prompt injection

        Falls back to manifest-based keyword search if DB/embeddings unavailable.
        """
        if not query or not query.strip():
            return ""

        # Try semantic search first
        result = self._semantic_retrieve(query, top_k, max_tokens, artifact_type)
        if result:
            return result

        # Fallback to keyword-based manifest search
        return self._keyword_retrieve(query, top_k, max_tokens, artifact_type)

    def _semantic_retrieve(
        self,
        query: str,
        top_k: int,
        max_tokens: int,
        artifact_type: Optional[str],
    ) -> str:
        """Retrieve via pgvector cosine similarity."""
        query_vec = _embed_text(query)
        if query_vec is None:
            return ""  # Model unavailable

        db = None
        try:
            from database import SessionLocal
            from models import ArtifactEmbedding

            db = SessionLocal()

            # pgvector cosine distance: embedding <=> query_vec  (lower = more similar)
            # We ORDER BY distance ASC to get most similar first
            q = db.query(ArtifactEmbedding).filter(
                ArtifactEmbedding.user_id == self.user_id,
                ArtifactEmbedding.embedding.isnot(None),
            )
            if artifact_type:
                q = q.filter(ArtifactEmbedding.artifact_type == artifact_type)

            # Use pgvector's cosine distance operator <=>
            results = q.order_by(
                ArtifactEmbedding.embedding.cosine_distance(query_vec)
            ).limit(top_k * 2).all()  # Fetch 2x for re-ranking

            if not results:
                return ""

            # Re-rank with hybrid scoring (semantic + keyword + recency)
            query_keywords = set(_extract_keywords(query))
            now = time.time()
            scored = []

            for row in results:
                # Cosine similarity = 1 - cosine_distance
                # pgvector stores distance; we compute similarity
                try:
                    row_vec = np.array(row.embedding) if row.embedding else None
                    q_vec = np.array(query_vec)
                    if row_vec is not None:
                        cos_sim = float(np.dot(row_vec, q_vec) / (
                            np.linalg.norm(row_vec) * np.linalg.norm(q_vec) + 1e-8
                        ))
                    else:
                        cos_sim = 0.0
                except Exception:
                    cos_sim = 0.0

                # Keyword boost
                summary_kw = set(_extract_keywords(row.summary or ""))
                tag_kw = set(t.lower() for t in (row.tags or []))
                kw_boost = len(summary_kw & query_keywords) * 0.1 + len(tag_kw & query_keywords) * 0.15

                # Recency boost
                try:
                    age_days = (now - row.last_used_at.timestamp()) / 86400
                except Exception:
                    age_days = 30
                recency = max(0, 1.0 - (age_days / 90))

                # Use count bonus (diminishing)
                use_bonus = min((row.use_count or 0) * 0.05, 0.3)

                # Combined score
                final_score = cos_sim * 5.0 + kw_boost + recency * 0.5 + use_bonus

                if final_score > 0.5:  # Minimum threshold
                    scored.append((final_score, row))

            scored.sort(key=lambda x: -x[0])
            top = scored[:top_k]

            if not top:
                return ""

            # Build context string
            parts = []
            tokens_used = 0
            char_budget = max_tokens * 4

            for score, row in top:
                # Read artifact file content
                try:
                    content = Path(row.file_path).read_text(encoding="utf-8")
                    content = re.sub(r"^---\n.+?\n---\n", "", content, flags=re.DOTALL).strip()
                except Exception:
                    content = row.summary

                remaining = char_budget - tokens_used * 4
                if len(content) > remaining:
                    content = content[:remaining] + "..."

                type_label = {
                    "task_result": "Past Result",
                    "knowledge": "Learning",
                    "playbook": "Playbook",
                }.get(row.artifact_type, row.artifact_type)

                parts.append(
                    f"### [{type_label}] {row.summary[:80]} (relevance: {score:.1f})\n{content}"
                )
                tokens_used += len(content) // 4

                # Update usage stats in DB
                row.last_used_at = datetime.now(timezone.utc).replace(tzinfo=None)
                row.use_count = (row.use_count or 0) + 1

                if tokens_used >= max_tokens:
                    break

            db.commit()
            return "\n\n".join(parts) if parts else ""

        except Exception as e:
            logger.warning(f"Semantic retrieval failed, falling back to keywords: {e}")
            if db:
                try:
                    db.rollback()
                except Exception:
                    pass
            return ""
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass

    def _keyword_retrieve(
        self,
        query: str,
        top_k: int,
        max_tokens: int,
        artifact_type: Optional[str],
    ) -> str:
        """Fallback keyword-based retrieval from local manifest."""
        if not self.manifest:
            return ""

        query_keywords = _extract_keywords(query)
        if not query_keywords:
            return ""

        now = time.time()
        scored: List[Tuple[float, ArtifactEntry]] = []

        for entry in self.manifest:
            if artifact_type and entry.artifact_type != artifact_type:
                continue

            tag_set = set(t.lower() for t in entry.tags)
            query_set = set(query_keywords)
            tag_overlap = len(tag_set & query_set)
            summary_kw = set(_extract_keywords(entry.summary))
            keyword_match = len(summary_kw & query_set)

            try:
                last_used_ts = datetime.fromisoformat(entry.last_used).timestamp()
            except Exception:
                last_used_ts = now - 86400 * 30
            age_days = (now - last_used_ts) / 86400
            recency = max(0, 1.0 - (age_days / 90))

            s = tag_overlap * 3.0 + keyword_match * 2.0 + recency * 1.0
            if s > 0.5:
                scored.append((s, entry))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        if not top:
            return ""

        parts = []
        tokens_used = 0
        char_budget = max_tokens * 4

        for score, entry in top:
            try:
                content = Path(entry.file_path).read_text(encoding="utf-8")
                content = re.sub(r"^---\n.+?\n---\n", "", content, flags=re.DOTALL).strip()
            except Exception:
                content = entry.summary

            remaining = char_budget - tokens_used * 4
            if len(content) > remaining:
                content = content[:remaining] + "..."

            type_label = {
                "task_result": "Past Result",
                "knowledge": "Learning",
                "playbook": "Playbook",
            }.get(entry.artifact_type, entry.artifact_type)

            parts.append(
                f"### [{type_label}] {entry.summary[:80]} (score: {score:.1f})\n{content}"
            )
            tokens_used += len(content) // 4

            entry.last_used = datetime.now().isoformat()
            entry.use_count += 1

            if tokens_used >= max_tokens:
                break

        self._save_manifest()
        return "\n\n".join(parts) if parts else ""

    def get_user_profile_prompt(self) -> str:
        """Get user profile as a prompt-friendly string."""
        if not self.profile:
            return "No profile data yet."

        parts = []
        if self.profile.get("preferred_agents"):
            agents = self.profile["preferred_agents"]
            top_agents = sorted(agents.items(), key=lambda x: -x[1])[:5]
            parts.append(
                "**Preferred agents:** "
                + ", ".join(f"{a} ({c}x)" for a, c in top_agents)
            )

        if self.profile.get("behavioral_notes"):
            notes = self.profile["behavioral_notes"][-3:]
            parts.append("**Notes:** " + "; ".join(notes))

        stats = f"Tasks completed: {self.profile.get('total_tasks_completed', 0)}"
        parts.append(stats)

        return "\n".join(parts) if parts else "New user — no history."

    # ── Distillation ─────────────────────────────────────────────────────────

    async def distill_conversation(
        self,
        action_history: List[Dict],
        insights: Dict[str, str],
        objective: str,
        success: bool,
    ):
        """
        Called at conversation end. Distills the session into reusable artifacts.

        1. Multi-step workflows → playbooks
        2. Insights → knowledge
        3. Agent usage → profile update
        """
        if not action_history:
            return

        # 1. Create playbook if multi-step (3+ actions)
        if len(action_history) >= 3 and success:
            name = objective[:60].strip()
            self.capture_playbook(
                name=name,
                objective=objective,
                steps=action_history,
                outcome="Completed successfully",
            )
            logger.info(f"📖 Distilled playbook from {len(action_history)}-step conversation")

        # 2. Store non-trivial insights as knowledge
        for key, value in insights.items():
            if len(value) > 20:
                self.capture_knowledge(
                    key=key,
                    value=value,
                    tags=_extract_keywords(value)[:5],
                    objective=objective,
                )

        # 3. Update user profile
        self.profile["total_tasks_completed"] = self.profile.get(
            "total_tasks_completed", 0
        ) + len(action_history)
        self.profile["total_conversations"] = self.profile.get(
            "total_conversations", 0
        ) + 1

        agent_counts = self.profile.get("preferred_agents", {})
        for entry in action_history:
            if entry.get("action_type") == "agent":
                agent = entry.get("resource_id", "")
                if agent:
                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
        self.profile["preferred_agents"] = agent_counts

        self._save_profile()
        logger.info("🧠 Conversation distilled into artifacts")

    # ── Backfill ─────────────────────────────────────────────────────────────

    def backfill_embeddings(self) -> int:
        """
        One-time backfill: embed all existing manifest entries that don't
        yet have a vector in PostgreSQL.  Returns count of items backfilled.
        """
        count = 0
        for entry in self.manifest:
            self._upsert_embedding(entry)
            count += 1
        logger.info(f"📦 Backfilled {count} artifact embeddings")
        return count

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        type_counts = Counter(e.artifact_type for e in self.manifest)
        return {
            "total_artifacts": len(self.manifest),
            "by_type": dict(type_counts),
            "total_tasks_completed": self.profile.get("total_tasks_completed", 0),
            "total_conversations": self.profile.get("total_conversations", 0),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_stores: Dict[str, ArtifactStore] = {}


def get_artifact_store(user_id: str = "default") -> ArtifactStore:
    """Get or create an artifact store for a user."""
    if user_id not in _stores:
        _stores[user_id] = ArtifactStore(user_id)
    return _stores[user_id]


# =============================================================================
# STARTUP PRE-LOADING (Background initialization)
# =============================================================================

def _preload_embedding_model_async():
    """
    Pre-load embedding model in background after server starts.
    
    Thread-safe: Uses existing double-checked locking in _get_embed_model().
    Benefit: First user message is fast (~200-800ms) if sent >14s after startup.
    
    If a request arrives before pre-loading completes, it will wait for the
    model to finish loading (graceful degradation to lazy-loading behavior).
    """
    def _load():
        try:
            _get_embed_model()
            logger.info("✅ ArtifactStore: Embedding model pre-loaded (background)")
        except Exception as e:
            logger.warning(f"⚠️ ArtifactStore: Embedding model pre-load failed: {e}")
    
    # Daemon thread: won't block server shutdown
    threading.Thread(target=_load, daemon=True, name="ArtifactEmbeddingPreload").start()
    logger.info("🔄 ArtifactStore: Embedding model pre-load started (background)")


# Trigger pre-loading at module import time (server startup)
_preload_embedding_model_async()

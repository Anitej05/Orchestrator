"""
Browser Agent - Persistent Memory

Cross-session memory storage for credentials, learnings, and important facts.
This data persists across browser agent runs and can be used to remember:
- Login credentials (securely encoded)
- Site-specific learnings (navigation patterns, element locations)
- User preferences and instructions
- Important facts and observations
"""

import json
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Storage location
PERSISTENT_MEMORY_DIR = Path("storage/browser_agent/memory")
MEMORY_FILE = PERSISTENT_MEMORY_DIR / "persistent_memory.json"
CREDENTIALS_FILE = PERSISTENT_MEMORY_DIR / "credentials.enc.json"


class Credential(BaseModel):
    """Stored credential for a site"""
    site: str  # e.g., "amazon.in", "github.com"
    username: str
    password_encoded: str  # Base64 encoded (not secure encryption, just obfuscation)
    last_used: Optional[float] = None
    notes: Optional[str] = None


class Learning(BaseModel):
    """Something the agent learned about a site or task"""
    category: str  # e.g., "site_navigation", "element_pattern", "user_preference"
    key: str
    value: str
    source: Optional[str] = None  # URL or context where learned
    confidence: float = 1.0
    created_at: float = Field(default_factory=time.time)
    last_accessed: Optional[float] = None


class PersistentMemory:
    """
    Cross-session persistent memory for the browser agent.
    
    Stores and retrieves:
    - Credentials (username/password for sites)
    - Learnings (patterns, preferences, instructions)
    - Site-specific info (where to find elements, navigation paths)
    """
    
    def __init__(self):
        self.credentials: Dict[str, Credential] = {}
        self.learnings: List[Learning] = []
        self.site_info: Dict[str, Dict[str, Any]] = {}  # site -> {key: value}
        self.user_preferences: Dict[str, Any] = {}
        
        # Ensure storage directory exists
        PERSISTENT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load persistent memory from disk"""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.learnings = [Learning(**l) for l in data.get('learnings', [])]
                self.site_info = data.get('site_info', {})
                self.user_preferences = data.get('user_preferences', {})
                logger.info(f"📚 Loaded persistent memory: {len(self.learnings)} learnings, {len(self.site_info)} sites")
            
            if CREDENTIALS_FILE.exists():
                with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                    cred_data = json.load(f)
                self.credentials = {k: Credential(**v) for k, v in cred_data.items()}
                logger.info(f"🔐 Loaded {len(self.credentials)} stored credentials")
                
        except Exception as e:
            logger.warning(f"Failed to load persistent memory: {e}")
    
    def _save(self):
        """Save persistent memory to disk"""
        try:
            # Save main memory
            data = {
                'learnings': [l.model_dump() for l in self.learnings],
                'site_info': self.site_info,
                'user_preferences': self.user_preferences,
                'last_updated': time.time()
            }
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save credentials separately
            if self.credentials:
                cred_data = {k: v.model_dump() for k, v in self.credentials.items()}
                with open(CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cred_data, f, indent=2)
                    
            logger.debug("💾 Persistent memory saved")
        except Exception as e:
            logger.error(f"Failed to save persistent memory: {e}")
    
    # ============== CREDENTIALS ==============
    
    def save_credential(self, site: str, username: str, password: str, notes: str = None):
        """Save login credentials for a site"""
        # Simple base64 encoding (not secure, just obfuscation)
        encoded_pw = base64.b64encode(password.encode()).decode()
        
        self.credentials[site.lower()] = Credential(
            site=site.lower(),
            username=username,
            password_encoded=encoded_pw,
            notes=notes
        )
        self._save()
        logger.info(f"🔐 Saved credentials for {site}")
    
    def get_credential(self, site: str) -> Optional[Dict[str, str]]:
        """Get stored credentials for a site"""
        site_lower = site.lower()
        
        # Try exact match first
        if site_lower in self.credentials:
            cred = self.credentials[site_lower]
            cred.last_used = time.time()
            self._save()
            return {
                'username': cred.username,
                'password': base64.b64decode(cred.password_encoded).decode(),
                'notes': cred.notes
            }
        
        # Try partial match (e.g., "amazon" matches "amazon.in")
        for stored_site, cred in self.credentials.items():
            if site_lower in stored_site or stored_site in site_lower:
                cred.last_used = time.time()
                self._save()
                return {
                    'username': cred.username,
                    'password': base64.b64decode(cred.password_encoded).decode(),
                    'notes': cred.notes
                }
        
        return None
    
    def has_credential(self, site: str) -> bool:
        """Check if we have credentials for a site"""
        return self.get_credential(site) is not None
    
    # ============== LEARNINGS ==============
    
    def add_learning(self, category: str, key: str, value: str, source: str = None, confidence: float = 1.0):
        """Add a new learning/fact to memory"""
        # Check if we already have this learning
        for learning in self.learnings:
            if learning.category == category and learning.key == key:
                # Update existing
                learning.value = value
                learning.confidence = max(learning.confidence, confidence)
                learning.source = source or learning.source
                self._save()
                logger.info(f"📝 Updated learning: [{category}] {key}")
                return
        
        # Add new learning
        self.learnings.append(Learning(
            category=category,
            key=key,
            value=value,
            source=source,
            confidence=confidence
        ))
        self._save()
        logger.info(f"📝 New learning: [{category}] {key} = {value[:50]}...")
    
    def get_learnings(self, category: str = None, key_contains: str = None) -> List[Learning]:
        """Get learnings, optionally filtered"""
        results = self.learnings
        
        if category:
            results = [l for l in results if l.category == category]
        
        if key_contains:
            results = [l for l in results if key_contains.lower() in l.key.lower()]
        
        # Update last accessed
        for l in results:
            l.last_accessed = time.time()
        
        return results
    
    def get_learning_value(self, category: str, key: str) -> Optional[str]:
        """Get a specific learning value"""
        for learning in self.learnings:
            if learning.category == category and learning.key == key:
                learning.last_accessed = time.time()
                return learning.value
        return None
    
    # ============== SITE INFO ==============
    
    def save_site_info(self, site: str, key: str, value: Any):
        """Save site-specific information"""
        site_lower = site.lower()
        if site_lower not in self.site_info:
            self.site_info[site_lower] = {}
        
        self.site_info[site_lower][key] = value
        self._save()
        logger.info(f"🌐 Saved site info: {site}.{key}")
    
    def get_site_info(self, site: str, key: str = None) -> Any:
        """Get site-specific information"""
        site_lower = site.lower()
        
        # Try exact match
        if site_lower in self.site_info:
            if key:
                return self.site_info[site_lower].get(key)
            return self.site_info[site_lower]
        
        # Try partial match
        for stored_site, info in self.site_info.items():
            if site_lower in stored_site or stored_site in site_lower:
                if key:
                    return info.get(key)
                return info
        
        return None
    
    # ============== USER PREFERENCES ==============
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference"""
        self.user_preferences[key] = value
        self._save()
        logger.info(f"⚙️ Preference saved: {key}")
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self.user_preferences.get(key, default)
    
    # ============== CONTEXT FOR LLM ==============
    
    def to_prompt_context(self, current_site: str = None, task_keywords: List[str] = None, task_description: str = None) -> str:
        """Format persistent memory for LLM prompt - HYBRID SEMANTIC + KEYWORD RETRIEVAL.
        
        Uses both:
        1. Semantic similarity (embeddings) to find conceptually related memories
        2. Keyword matching as fallback/boost for exact matches
        
        This ensures the right data is always retrieved regardless of phrasing.
        """
        sections = []
        
        # 1. CREDENTIALS - Always show for current site, plus summary of others
        if self.credentials:
            current_site_creds = []
            other_sites = []
            
            for site in self.credentials.keys():
                if current_site and (current_site.lower() in site or site in current_site.lower()):
                    current_site_creds.append(site)
                else:
                    other_sites.append(site)
            
            if current_site_creds:
                sections.append(f"🔐 CREDENTIALS AVAILABLE for current site: {', '.join(current_site_creds)}")
                sections.append("   → Use get_credential action to auto-login")
            
            if other_sites:
                sections.append(f"🔐 Other stored logins: {', '.join(other_sites[:5])}" + 
                               (f" (+{len(other_sites)-5} more)" if len(other_sites) > 5 else ""))
        
        # 2. SITE INFO - Only for current site
        if current_site:
            site_data = self.get_site_info(current_site)
            if site_data:
                sections.append(f"🌐 KNOWN INFO for {current_site}:")
                for k, v in list(site_data.items())[:5]:
                    sections.append(f"   - {k}: {str(v)[:100]}")
        
        # 3. LEARNINGS - HYBRID SEMANTIC + KEYWORD RETRIEVAL
        if self.learnings:
            relevant_learnings = self._retrieve_relevant_learnings(
                task_description or " ".join(task_keywords or []),
                current_site,
                max_results=5
            )
            
            if relevant_learnings:
                sections.append("📚 RELEVANT REMEMBERED FACTS:")
                for l, score in relevant_learnings:
                    sections.append(f"   - [{l.category}] {l.key}: {l.value[:80]}")
        
        # 4. USER PREFERENCES - Always include (usually small)
        if self.user_preferences:
            sections.append("⚙️ USER PREFERENCES:")
            for k, v in list(self.user_preferences.items())[:3]:
                sections.append(f"   - {k}: {v}")
        
        if not sections:
            return "📦 PERSISTENT MEMORY: Empty. Use save_credential/save_learning to remember info."
        
        return "\n".join(sections)
    
    def _retrieve_relevant_learnings(self, query: str, current_site: str = None, max_results: int = 5) -> List[tuple]:
        """Hybrid retrieval using semantic similarity + keyword matching.
        
        Returns list of (Learning, score) tuples sorted by relevance.
        """
        if not self.learnings:
            return []
        
        if not query or len(query.strip()) < 3:
            # No query, return most recent + always-include categories
            results = []
            for l in self.learnings:
                if l.category in ['user_preference', 'instruction']:
                    results.append((l, 1.0))
            
            # Add recent if we have room
            recent = sorted(self.learnings, key=lambda x: x.last_accessed or x.created_at, reverse=True)
            for l in recent[:max_results - len(results)]:
                if (l, 1.0) not in results:
                    results.append((l, 0.5))
            
            return results[:max_results]
        
        scored_learnings = []
        
        # Try semantic similarity first
        try:
            query_embedding = self._get_embedding(query)
            
            for learning in self.learnings:
                # Compute semantic similarity
                learning_text = f"{learning.category} {learning.key} {learning.value}"
                learning_embedding = self._get_embedding(learning_text)
                
                semantic_score = self._cosine_similarity(query_embedding, learning_embedding)
                
                # Keyword boost
                keyword_boost = 0.0
                query_lower = query.lower()
                if learning.key.lower() in query_lower or query_lower in learning.key.lower():
                    keyword_boost += 0.3
                if any(word in learning.value.lower() for word in query_lower.split() if len(word) > 3):
                    keyword_boost += 0.2
                
                # Category boost (always relevant)
                category_boost = 0.0
                if learning.category in ['user_preference', 'instruction']:
                    category_boost = 0.2
                
                # Site relevance boost
                site_boost = 0.0
                if current_site and learning.source and current_site.lower() in learning.source.lower():
                    site_boost = 0.15
                
                total_score = semantic_score + keyword_boost + category_boost + site_boost
                scored_learnings.append((learning, total_score))
                
        except Exception as e:
            logger.warning(f"Semantic retrieval failed, falling back to keyword: {e}")
            # Fallback to keyword-only matching
            for learning in self.learnings:
                score = 0.0
                learning_text = f"{learning.key} {learning.value}".lower()
                query_lower = query.lower()
                
                # Exact key match
                if learning.key.lower() in query_lower:
                    score += 0.5
                
                # Word overlap
                query_words = set(w for w in query_lower.split() if len(w) > 3)
                learning_words = set(w for w in learning_text.split() if len(w) > 3)
                overlap = len(query_words & learning_words)
                if overlap > 0:
                    score += 0.1 * overlap
                
                # Always include preferences/instructions
                if learning.category in ['user_preference', 'instruction']:
                    score += 0.3
                
                scored_learnings.append((learning, score))
        
        # Sort by score and return top results
        scored_learnings.sort(key=lambda x: x[1], reverse=True)
        
        # Filter to only return meaningful matches (score > 0.3) + always-include categories
        results = []
        for learning, score in scored_learnings:
            if score > 0.3 or learning.category in ['user_preference', 'instruction']:
                results.append((learning, score))
                if len(results) >= max_results:
                    break
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using sentence transformers."""
        global _embedding_model
        
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer('all-mpnet-base-v2')
                logger.info("📊 Loaded embedding model for semantic retrieval")
            except ImportError:
                logger.warning("sentence_transformers not available, using fallback")
                raise
        
        return _embedding_model.encode(text).tolist()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_summary(self) -> str:
        """Get a brief summary of stored memory"""
        return f"Memory: {len(self.credentials)} credentials, {len(self.learnings)} learnings, {len(self.site_info)} sites"


# Global embedding model (lazy loaded)
_embedding_model = None

# Singleton instance
_persistent_memory: Optional[PersistentMemory] = None

def get_persistent_memory() -> PersistentMemory:
    """Get the singleton persistent memory instance"""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemory()
    return _persistent_memory

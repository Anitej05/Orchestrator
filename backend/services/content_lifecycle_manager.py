"""
Content Lifecycle Manager

Manages content retention, cleanup, and archival policies.

Features:
1. Automatic cleanup of expired content
2. Session cleanup on conversation end
3. Archival of old content
4. Storage quota management
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil

from services.unified_content_service import (
    UnifiedContentService,
    UnifiedContentMetadata,
    ContentType,
    ContentSource,
    ContentPriority,
    RetentionPolicy,
    get_content_service
)

logger = logging.getLogger("ContentLifecycleManager")


class ContentLifecycleManager:
    """
    Manages the lifecycle of content in the system.
    
    Responsibilities:
    - Automatic cleanup of expired content
    - Session cleanup when conversations end
    - Archival of old but important content
    - Storage quota enforcement
    """
    
    # Default retention periods (in hours)
    DEFAULT_RETENTION = {
        ContentPriority.CRITICAL: None,      # Never expires
        ContentPriority.HIGH: 30 * 24,       # 30 days
        ContentPriority.MEDIUM: 7 * 24,      # 7 days
        ContentPriority.LOW: 24,             # 24 hours
        ContentPriority.EPHEMERAL: 1,        # 1 hour
    }
    
    # Storage quota (in bytes) - 1GB default
    DEFAULT_QUOTA = 1024 * 1024 * 1024
    
    def __init__(
        self,
        content_service: Optional[UnifiedContentService] = None,
        archive_dir: str = "storage/archive",
        quota_bytes: int = DEFAULT_QUOTA
    ):
        self.service = content_service or get_content_service()
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.quota_bytes = quota_bytes
        
        logger.info(f"ContentLifecycleManager initialized with {quota_bytes / (1024*1024):.1f}MB quota")
    
    # =========================================================================
    # CLEANUP OPERATIONS
    # =========================================================================
    
    def cleanup_expired(self) -> Dict[str, Any]:
        """
        Remove all expired content.
        
        Returns:
            {
                "removed_count": int,
                "freed_bytes": int,
                "removed_ids": List[str]
            }
        """
        now = datetime.utcnow()
        expired_ids = []
        freed_bytes = 0
        
        for content_id, metadata in list(self.service._registry.items()):
            if metadata.expires_at:
                expires_at = datetime.fromisoformat(metadata.expires_at)
                if now > expires_at:
                    freed_bytes += metadata.size_bytes
                    expired_ids.append(content_id)
        
        for content_id in expired_ids:
            self.service.delete_content(content_id)
        
        result = {
            "removed_count": len(expired_ids),
            "freed_bytes": freed_bytes,
            "removed_ids": expired_ids
        }
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired items, freed {freed_bytes / 1024:.1f}KB")
        
        return result
    
    def cleanup_session(self, thread_id: str) -> Dict[str, Any]:
        """
        Remove session-only content for a specific thread.
        
        Args:
            thread_id: The conversation thread ID
        
        Returns:
            {
                "removed_count": int,
                "freed_bytes": int
            }
        """
        session_content = [
            m for m in self.service._registry.values()
            if m.thread_id == thread_id and m.retention_policy == RetentionPolicy.SESSION
        ]
        
        freed_bytes = sum(m.size_bytes for m in session_content)
        
        for metadata in session_content:
            self.service.delete_content(metadata.id)
        
        result = {
            "removed_count": len(session_content),
            "freed_bytes": freed_bytes
        }
        
        if session_content:
            logger.info(f"Cleaned up {len(session_content)} session items for thread {thread_id}")
        
        return result
    
    def cleanup_by_priority(self, priority: ContentPriority) -> Dict[str, Any]:
        """
        Remove all content of a specific priority level.
        
        Args:
            priority: The priority level to clean up
        
        Returns:
            {
                "removed_count": int,
                "freed_bytes": int
            }
        """
        target_content = [
            m for m in self.service._registry.values()
            if m.priority == priority
        ]
        
        freed_bytes = sum(m.size_bytes for m in target_content)
        
        for metadata in target_content:
            self.service.delete_content(metadata.id)
        
        return {
            "removed_count": len(target_content),
            "freed_bytes": freed_bytes
        }
    
    def cleanup_old_content(self, days: int = 30) -> Dict[str, Any]:
        """
        Remove content older than specified days (except CRITICAL priority).
        
        Args:
            days: Age threshold in days
        
        Returns:
            {
                "removed_count": int,
                "freed_bytes": int
            }
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        old_content = []
        
        for metadata in self.service._registry.values():
            if metadata.priority == ContentPriority.CRITICAL:
                continue
            
            created_at = datetime.fromisoformat(metadata.created_at)
            if created_at < cutoff:
                old_content.append(metadata)
        
        freed_bytes = sum(m.size_bytes for m in old_content)
        
        for metadata in old_content:
            self.service.delete_content(metadata.id)
        
        result = {
            "removed_count": len(old_content),
            "freed_bytes": freed_bytes
        }
        
        if old_content:
            logger.info(f"Cleaned up {len(old_content)} items older than {days} days")
        
        return result
    
    # =========================================================================
    # ARCHIVAL OPERATIONS
    # =========================================================================
    
    def archive_content(self, content_id: str) -> Optional[str]:
        """
        Archive content to the archive directory.
        
        Args:
            content_id: The content ID to archive
        
        Returns:
            Archive path if successful, None otherwise
        """
        metadata = self.service.get_metadata(content_id)
        if not metadata:
            return None
        
        # Create archive subdirectory by date
        archive_date = datetime.utcnow().strftime("%Y-%m")
        archive_subdir = self.archive_dir / archive_date
        archive_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to archive
        source_path = Path(metadata.storage_path)
        if not source_path.exists():
            return None
        
        archive_path = archive_subdir / f"{content_id}_{metadata.name}"
        shutil.copy2(source_path, archive_path)
        
        # Save metadata alongside
        metadata_path = archive_subdir / f"{content_id}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        logger.info(f"Archived content {content_id} to {archive_path}")
        return str(archive_path)
    
    def archive_old_content(self, days: int = 30) -> Dict[str, Any]:
        """
        Archive content older than specified days before deletion.
        
        Args:
            days: Age threshold in days
        
        Returns:
            {
                "archived_count": int,
                "archived_bytes": int,
                "archive_paths": List[str]
            }
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        archived = []
        archived_bytes = 0
        
        for metadata in list(self.service._registry.values()):
            # Only archive HIGH and MEDIUM priority
            if metadata.priority not in [ContentPriority.HIGH, ContentPriority.MEDIUM]:
                continue
            
            created_at = datetime.fromisoformat(metadata.created_at)
            if created_at < cutoff:
                archive_path = self.archive_content(metadata.id)
                if archive_path:
                    archived.append(archive_path)
                    archived_bytes += metadata.size_bytes
                    self.service.delete_content(metadata.id)
        
        result = {
            "archived_count": len(archived),
            "archived_bytes": archived_bytes,
            "archive_paths": archived
        }
        
        if archived:
            logger.info(f"Archived {len(archived)} items ({archived_bytes / 1024:.1f}KB)")
        
        return result
    
    # =========================================================================
    # QUOTA MANAGEMENT
    # =========================================================================
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get current storage usage.
        
        Returns:
            {
                "total_bytes": int,
                "quota_bytes": int,
                "usage_percent": float,
                "items_count": int
            }
        """
        total_bytes = sum(m.size_bytes for m in self.service._registry.values())
        
        return {
            "total_bytes": total_bytes,
            "quota_bytes": self.quota_bytes,
            "usage_percent": (total_bytes / self.quota_bytes) * 100 if self.quota_bytes > 0 else 0,
            "items_count": len(self.service._registry)
        }
    
    def enforce_quota(self) -> Dict[str, Any]:
        """
        Enforce storage quota by removing low-priority content.
        
        Returns:
            {
                "removed_count": int,
                "freed_bytes": int,
                "new_usage_percent": float
            }
        """
        usage = self.get_storage_usage()
        
        if usage["usage_percent"] < 90:
            return {
                "removed_count": 0,
                "freed_bytes": 0,
                "new_usage_percent": usage["usage_percent"]
            }
        
        logger.warning(f"Storage at {usage['usage_percent']:.1f}%, enforcing quota")
        
        removed_count = 0
        freed_bytes = 0
        
        # Remove in priority order: EPHEMERAL -> LOW -> MEDIUM
        for priority in [ContentPriority.EPHEMERAL, ContentPriority.LOW, ContentPriority.MEDIUM]:
            if usage["total_bytes"] - freed_bytes < self.quota_bytes * 0.8:
                break
            
            result = self.cleanup_by_priority(priority)
            removed_count += result["removed_count"]
            freed_bytes += result["freed_bytes"]
        
        new_usage = self.get_storage_usage()
        
        return {
            "removed_count": removed_count,
            "freed_bytes": freed_bytes,
            "new_usage_percent": new_usage["usage_percent"]
        }
    
    # =========================================================================
    # RETENTION POLICY MANAGEMENT
    # =========================================================================
    
    def set_retention_policy(
        self,
        content_id: str,
        policy: RetentionPolicy,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        Set retention policy for specific content.
        
        Args:
            content_id: The content ID
            policy: New retention policy
            ttl_hours: Optional TTL in hours (for TTL policy)
        
        Returns:
            True if successful
        """
        metadata = self.service.get_metadata(content_id)
        if not metadata:
            return False
        
        metadata.retention_policy = policy
        
        if policy == RetentionPolicy.TTL and ttl_hours:
            metadata.expires_at = (datetime.utcnow() + timedelta(hours=ttl_hours)).isoformat()
        elif policy == RetentionPolicy.PERMANENT:
            metadata.expires_at = None
        
        self.service._save_registry()
        
        logger.info(f"Set retention policy for {content_id}: {policy.value}")
        return True
    
    def set_priority(self, content_id: str, priority: ContentPriority) -> bool:
        """
        Set priority for specific content.
        
        Args:
            content_id: The content ID
            priority: New priority level
        
        Returns:
            True if successful
        """
        metadata = self.service.get_metadata(content_id)
        if not metadata:
            return False
        
        metadata.priority = priority
        
        # Update expiration based on new priority
        if priority in self.DEFAULT_RETENTION:
            hours = self.DEFAULT_RETENTION[priority]
            if hours:
                metadata.expires_at = (datetime.utcnow() + timedelta(hours=hours)).isoformat()
            else:
                metadata.expires_at = None
        
        self.service._save_registry()
        
        logger.info(f"Set priority for {content_id}: {priority.value}")
        return True
    
    # =========================================================================
    # SCHEDULED CLEANUP
    # =========================================================================
    
    async def run_scheduled_cleanup(self):
        """
        Run all scheduled cleanup tasks.
        
        This should be called periodically (e.g., every hour).
        """
        logger.info("Running scheduled cleanup...")
        
        # 1. Clean up expired content
        expired_result = self.cleanup_expired()
        
        # 2. Enforce quota if needed
        quota_result = self.enforce_quota()
        
        # 3. Archive old content (weekly)
        # This could be triggered by a separate scheduler
        
        return {
            "expired": expired_result,
            "quota": quota_result,
            "timestamp": datetime.utcnow().isoformat()
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_lifecycle_manager: Optional[ContentLifecycleManager] = None


def get_lifecycle_manager() -> ContentLifecycleManager:
    """Get the global lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ContentLifecycleManager()
    return _lifecycle_manager

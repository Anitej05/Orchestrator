"""
Comprehensive File Management Service for Orchestrator

This service handles:
1. File uploads from users
2. File distribution to agents (pre-upload to agents that require it)
3. File storage from agent responses
4. File downloads for users
5. File metadata tracking
6. Cleanup and lifecycle management
"""

import os
import uuid
import json
import httpx
import logging
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Use the AgentOrchestrator logger so file management logs appear in orchestrator_temp.log
logger = logging.getLogger("AgentOrchestrator")

# Storage directories
STORAGE_BASE = Path("storage")
USER_UPLOADS_DIR = STORAGE_BASE / "uploads"
AGENT_FILES_DIR = STORAGE_BASE / "agent_files"
TEMP_DIR = STORAGE_BASE / "temp"
DOCUMENTS_DIR = STORAGE_BASE / "documents"
IMAGES_DIR = STORAGE_BASE / "images"
SPREADSHEETS_DIR = STORAGE_BASE / "spreadsheets"

# Ensure directories exist
for dir_path in [USER_UPLOADS_DIR, AGENT_FILES_DIR, TEMP_DIR, DOCUMENTS_DIR, IMAGES_DIR, SPREADSHEETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class FileCategory(str, Enum):
    """Categories of files based on their purpose and type"""
    IMAGE = "image"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    DATA = "data"
    CODE = "code"
    ARCHIVE = "archive"
    OTHER = "other"


class FileSource(str, Enum):
    """Source of the file"""
    USER_UPLOAD = "user_upload"
    AGENT_OUTPUT = "agent_output"
    SYSTEM_GENERATED = "system_generated"


@dataclass
class AgentFileMapping:
    """Tracks file IDs across different agents"""
    orchestrator_file_id: str
    agent_id: str
    agent_file_id: str
    agent_endpoint: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ManagedFile:
    """Comprehensive file metadata"""
    file_id: str
    original_name: str
    stored_path: str
    file_category: FileCategory
    file_source: FileSource
    mime_type: str
    file_size: int
    checksum: str
    
    # Ownership
    user_id: str
    thread_id: Optional[str] = None
    
    # Agent mappings - tracks file_ids on different agents
    agent_mappings: Dict[str, AgentFileMapping] = field(default_factory=dict)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    is_temporary: bool = False
    
    # Content info
    content_summary: Optional[str] = None
    vector_store_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['file_category'] = self.file_category.value
        result['file_source'] = self.file_source.value
        result['agent_mappings'] = {
            k: asdict(v) for k, v in self.agent_mappings.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagedFile':
        """Create from dictionary"""
        data['file_category'] = FileCategory(data['file_category'])
        data['file_source'] = FileSource(data['file_source'])
        data['agent_mappings'] = {
            k: AgentFileMapping(**v) for k, v in data.get('agent_mappings', {}).items()
        }
        return cls(**data)


class FileManagementService:
    """
    Central service for managing files across the orchestrator and agents.
    
    Key Features:
    - Unified file tracking with unique IDs
    - Automatic file distribution to agents
    - Agent file ID mapping (orchestrator ID <-> agent ID)
    - File lifecycle management
    - Download support
    """
    
    def __init__(self):
        self._file_registry: Dict[str, ManagedFile] = {}
        self._registry_path = STORAGE_BASE / "file_registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load file registry from disk"""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r') as f:
                    data = json.load(f)
                    self._file_registry = {
                        k: ManagedFile.from_dict(v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._file_registry)} files from registry")
            except Exception as e:
                logger.error(f"Failed to load file registry: {e}")
                self._file_registry = {}
    
    def _save_registry(self):
        """Persist file registry to disk"""
        try:
            with open(self._registry_path, 'w') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._file_registry.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save file registry: {e}")
    
    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _determine_category(self, filename: str, mime_type: str) -> FileCategory:
        """Determine file category based on extension and mime type"""
        ext = Path(filename).suffix.lower()
        
        # Spreadsheets
        if ext in ['.csv', '.xlsx', '.xls', '.ods']:
            return FileCategory.SPREADSHEET
        
        # Images
        if mime_type and mime_type.startswith('image/'):
            return FileCategory.IMAGE
        
        # Documents
        if ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
            return FileCategory.DOCUMENT
        
        # Code
        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs']:
            return FileCategory.CODE
        
        # Data
        if ext in ['.json', '.xml', '.yaml', '.yml']:
            return FileCategory.DATA
        
        # Archives
        if ext in ['.zip', '.tar', '.gz', '.rar', '.7z']:
            return FileCategory.ARCHIVE
        
        return FileCategory.OTHER
    
    def _get_storage_dir(self, category: FileCategory) -> Path:
        """Get appropriate storage directory for file category"""
        mapping = {
            FileCategory.IMAGE: IMAGES_DIR,
            FileCategory.DOCUMENT: DOCUMENTS_DIR,
            FileCategory.SPREADSHEET: SPREADSHEETS_DIR,
            FileCategory.DATA: DOCUMENTS_DIR,
            FileCategory.CODE: DOCUMENTS_DIR,
            FileCategory.ARCHIVE: DOCUMENTS_DIR,
            FileCategory.OTHER: DOCUMENTS_DIR,
        }
        return mapping.get(category, DOCUMENTS_DIR)


    async def register_user_upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        thread_id: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> ManagedFile:
        """
        Register a file uploaded by a user.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            user_id: ID of the uploading user
            thread_id: Optional conversation thread ID
            mime_type: Optional MIME type (will be guessed if not provided)
        
        Returns:
            ManagedFile object with all metadata
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Determine MIME type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or 'application/octet-stream'
        
        # Determine category and storage location
        category = self._determine_category(filename, mime_type)
        storage_dir = self._get_storage_dir(category)
        
        # Create unique stored filename
        ext = Path(filename).suffix
        stored_filename = f"{file_id}{ext}"
        stored_path = storage_dir / stored_filename
        
        # Write file to disk
        with open(stored_path, 'wb') as f:
            f.write(file_content)
        
        # Compute checksum
        checksum = self._compute_checksum(str(stored_path))
        
        # Create managed file record
        managed_file = ManagedFile(
            file_id=file_id,
            original_name=filename,
            stored_path=str(stored_path),
            file_category=category,
            file_source=FileSource.USER_UPLOAD,
            mime_type=mime_type,
            file_size=len(file_content),
            checksum=checksum,
            user_id=user_id,
            thread_id=thread_id
        )
        
        # Register in memory and persist
        self._file_registry[file_id] = managed_file
        self._save_registry()
        
        logger.info(f"Registered user upload: {filename} -> {file_id} ({category.value})")
        return managed_file
    
    async def upload_file_to_agent(
        self,
        file_id: str,
        agent_id: str,
        agent_base_url: str,
        upload_endpoint: str = "/upload",
        file_param_name: str = "file",
        force_reupload: bool = False
    ) -> Optional[str]:
        """
        Upload a managed file to a specific agent and track the mapping.
        
        This is the KEY function that solves the spreadsheet agent issue.
        It uploads the file to the agent's upload endpoint and stores
        the returned file_id for use in subsequent calls.
        
        Args:
            file_id: Orchestrator's file ID
            agent_id: ID of the target agent
            agent_base_url: Base URL of the agent
            upload_endpoint: Agent's upload endpoint (default: /upload)
            file_param_name: Name of the file parameter (default: file)
            force_reupload: If True, always re-upload even if mapping exists
        
        Returns:
            Agent's file_id if successful, None otherwise
        """
        managed_file = self._file_registry.get(file_id)
        if not managed_file:
            logger.error(f"File not found in registry: {file_id}")
            return None
        
        # Check if already uploaded to this agent
        if agent_id in managed_file.agent_mappings and not force_reupload:
            existing_mapping = managed_file.agent_mappings[agent_id]
            logger.info(f"[FILE_UPLOAD] Found existing mapping for {file_id} -> {existing_mapping.agent_file_id}")
            
            # Verify the file still exists on the agent by making a lightweight check
            # Try to call get_summary with the file_id to verify it exists
            verify_url = f"{agent_base_url.rstrip('/')}/get_summary"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Use form data for verification (spreadsheet agent uses form)
                    response = await client.post(verify_url, data={"file_id": existing_mapping.agent_file_id})
                    if response.status_code == 200:
                        logger.info(f"[FILE_UPLOAD] ✅ Verified file {existing_mapping.agent_file_id} still exists on agent {agent_id}")
                        return existing_mapping.agent_file_id
                    elif response.status_code == 404:
                        logger.warning(f"[FILE_UPLOAD] ⚠️ File {existing_mapping.agent_file_id} no longer exists on agent {agent_id}, will re-upload")
                        # Remove stale mapping
                        del managed_file.agent_mappings[agent_id]
                        self._save_registry()
                    else:
                        logger.warning(f"[FILE_UPLOAD] ⚠️ Verification returned {response.status_code}, will re-upload to be safe")
                        del managed_file.agent_mappings[agent_id]
                        self._save_registry()
            except Exception as e:
                logger.warning(f"[FILE_UPLOAD] ⚠️ Could not verify file on agent: {e}, will re-upload")
                # Remove potentially stale mapping
                del managed_file.agent_mappings[agent_id]
                self._save_registry()
        
        # Read file content
        try:
            with open(managed_file.stored_path, 'rb') as f:
                file_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_id}: {e}")
            return None
        
        # Upload to agent
        upload_url = f"{agent_base_url.rstrip('/')}{upload_endpoint}"
        
        logger.info(f"[FILE_UPLOAD] Uploading file {file_id} to agent {agent_id} at {upload_url}")
        logger.info(f"[FILE_UPLOAD] File name: {managed_file.original_name}, size: {len(file_content)} bytes")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {file_param_name: (managed_file.original_name, file_content, managed_file.mime_type)}
                response = await client.post(upload_url, files=files)
                
                logger.info(f"[FILE_UPLOAD] Agent response status: {response.status_code}")
                logger.info(f"[FILE_UPLOAD] Agent response body: {response.text[:500]}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract agent's file_id from response
                    # Handle different response formats
                    agent_file_id = None
                    if isinstance(result, dict):
                        # Try common patterns
                        if 'result' in result and isinstance(result['result'], dict):
                            agent_file_id = result['result'].get('file_id')
                        elif 'file_id' in result:
                            agent_file_id = result['file_id']
                        elif 'id' in result:
                            agent_file_id = result['id']
                    
                    logger.info(f"[FILE_UPLOAD] Extracted agent_file_id: {agent_file_id}")
                    
                    if agent_file_id:
                        # Create mapping
                        mapping = AgentFileMapping(
                            orchestrator_file_id=file_id,
                            agent_id=agent_id,
                            agent_file_id=agent_file_id,
                            agent_endpoint=upload_endpoint
                        )
                        managed_file.agent_mappings[agent_id] = mapping
                        self._save_registry()
                        
                        logger.info(f"[FILE_UPLOAD] ✅ Successfully uploaded file {file_id} to agent {agent_id}: agent_file_id={agent_file_id}")
                        return agent_file_id
                    else:
                        logger.warning(f"[FILE_UPLOAD] ⚠️ Agent {agent_id} returned success but no file_id: {result}")
                        return None
                else:
                    logger.error(f"[FILE_UPLOAD] ❌ Failed to upload to agent {agent_id}: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"[FILE_UPLOAD] ❌ Error uploading file to agent {agent_id}: {e}", exc_info=True)
            return None
    
    def get_agent_file_id(self, file_id: str, agent_id: str) -> Optional[str]:
        """
        Get the agent-specific file ID for a managed file.
        
        Args:
            file_id: Orchestrator's file ID
            agent_id: Target agent ID
        
        Returns:
            Agent's file_id if mapping exists, None otherwise
        """
        managed_file = self._file_registry.get(file_id)
        if not managed_file:
            return None
        
        mapping = managed_file.agent_mappings.get(agent_id)
        return mapping.agent_file_id if mapping else None
    
    async def register_agent_output(
        self,
        file_content: bytes,
        filename: str,
        agent_id: str,
        user_id: str,
        thread_id: Optional[str] = None,
        mime_type: Optional[str] = None,
        is_temporary: bool = False,
        expires_hours: int = 24
    ) -> ManagedFile:
        """
        Register a file generated by an agent.
        
        Args:
            file_content: Raw file bytes
            filename: Filename for the output
            agent_id: ID of the agent that generated the file
            user_id: ID of the user who owns this file
            thread_id: Optional conversation thread ID
            mime_type: Optional MIME type
            is_temporary: Whether file should be auto-deleted
            expires_hours: Hours until expiration (if temporary)
        
        Returns:
            ManagedFile object
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Determine MIME type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or 'application/octet-stream'
        
        # Store in agent files directory
        ext = Path(filename).suffix
        stored_filename = f"{file_id}{ext}"
        stored_path = AGENT_FILES_DIR / agent_id / stored_filename
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(stored_path, 'wb') as f:
            f.write(file_content)
        
        # Compute checksum
        checksum = self._compute_checksum(str(stored_path))
        
        # Calculate expiration
        expires_at = None
        if is_temporary:
            expires_at = (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
        
        # Create managed file record
        managed_file = ManagedFile(
            file_id=file_id,
            original_name=filename,
            stored_path=str(stored_path),
            file_category=self._determine_category(filename, mime_type),
            file_source=FileSource.AGENT_OUTPUT,
            mime_type=mime_type,
            file_size=len(file_content),
            checksum=checksum,
            user_id=user_id,
            thread_id=thread_id,
            is_temporary=is_temporary,
            expires_at=expires_at
        )
        
        # Register
        self._file_registry[file_id] = managed_file
        self._save_registry()
        
        logger.info(f"Registered agent output from {agent_id}: {filename} -> {file_id}")
        return managed_file
    
    def get_file(self, file_id: str) -> Optional[ManagedFile]:
        """Get a managed file by ID"""
        return self._file_registry.get(file_id)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the stored path for a file"""
        managed_file = self._file_registry.get(file_id)
        return managed_file.stored_path if managed_file else None
    
    def get_file_content(self, file_id: str) -> Optional[bytes]:
        """Read and return file content"""
        managed_file = self._file_registry.get(file_id)
        if not managed_file:
            return None
        
        try:
            with open(managed_file.stored_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_id}: {e}")
            return None
    
    def get_files_for_thread(self, thread_id: str) -> List[ManagedFile]:
        """Get all files associated with a conversation thread"""
        return [
            f for f in self._file_registry.values()
            if f.thread_id == thread_id
        ]
    
    def get_files_for_user(self, user_id: str) -> List[ManagedFile]:
        """Get all files owned by a user"""
        return [
            f for f in self._file_registry.values()
            if f.user_id == user_id
        ]
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a managed file"""
        managed_file = self._file_registry.get(file_id)
        if not managed_file:
            return False
        
        # Delete physical file
        try:
            if os.path.exists(managed_file.stored_path):
                os.remove(managed_file.stored_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
        
        # Remove from registry
        del self._file_registry[file_id]
        self._save_registry()
        
        logger.info(f"Deleted file: {file_id}")
        return True
    
    def cleanup_expired_files(self) -> int:
        """Remove expired temporary files"""
        now = datetime.utcnow()
        expired_ids = []
        
        for file_id, managed_file in self._file_registry.items():
            if managed_file.is_temporary and managed_file.expires_at:
                expires_at = datetime.fromisoformat(managed_file.expires_at)
                if now > expires_at:
                    expired_ids.append(file_id)
        
        for file_id in expired_ids:
            self.delete_file(file_id)
        
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired files")
        
        return len(expired_ids)


# Global instance
file_manager = FileManagementService()


# Helper functions for orchestrator integration
async def prepare_files_for_agent(
    files: List[Dict[str, Any]],
    agent_id: str,
    agent_config: Dict[str, Any],
    user_id: str
) -> Dict[str, str]:
    """
    Prepare files for an agent by uploading them if needed.
    
    This function checks if the agent has an upload endpoint and
    uploads files to get agent-specific file IDs.
    
    Args:
        files: List of file objects from state
        agent_id: Target agent ID
        agent_config: Agent's connection config
        user_id: User ID
    
    Returns:
        Dict mapping orchestrator file_ids to agent file_ids
    """
    logger.info(f"[PREPARE_FILES] Called for agent {agent_id} with {len(files)} files")
    logger.info(f"[PREPARE_FILES] agent_config: {agent_config}")
    logger.info(f"[PREPARE_FILES] files: {files}")
    
    file_id_mapping = {}
    
    # Get agent's base URL
    base_url = agent_config.get('base_url', '') if agent_config else ''
    
    logger.info(f"[PREPARE_FILES] base_url: {base_url}")
    
    if not base_url:
        logger.warning(f"[PREPARE_FILES] No base_url for agent {agent_id}, skipping file preparation")
        return file_id_mapping
    
    for file_obj in files:
        # Get or create orchestrator file_id
        file_path = file_obj.get('file_path') or file_obj.get('stored_path')
        file_name = file_obj.get('file_name') or file_obj.get('original_name')
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        # Check if file is already registered
        existing_file = None
        for f in file_manager._file_registry.values():
            if f.stored_path == file_path or f.original_name == file_name:
                existing_file = f
                break
        
        if existing_file:
            orchestrator_file_id = existing_file.file_id
            logger.info(f"Found existing file in registry: {orchestrator_file_id}")
        else:
            # Register the file from its current location
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                managed_file = await file_manager.register_user_upload(
                    file_content=content,
                    filename=file_name,
                    user_id=user_id
                )
                orchestrator_file_id = managed_file.file_id
                logger.info(f"Registered new file: {file_name} -> {orchestrator_file_id}")
            except FileNotFoundError:
                logger.error(f"File not found at path: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error registering file {file_name}: {e}")
                continue
        
        # Upload to agent
        logger.info(f"[PREPARE_FILES] Uploading file {orchestrator_file_id} ({file_name}) to agent {agent_id}")
        agent_file_id = await file_manager.upload_file_to_agent(
            file_id=orchestrator_file_id,
            agent_id=agent_id,
            agent_base_url=base_url
        )
        
        logger.info(f"[PREPARE_FILES] Upload result: agent_file_id={agent_file_id}")
        
        if agent_file_id:
            file_id_mapping[orchestrator_file_id] = agent_file_id
            # Also map by original filename for easier lookup
            file_id_mapping[file_name] = agent_file_id
            logger.info(f"[PREPARE_FILES] Added mapping: {orchestrator_file_id} -> {agent_file_id}, {file_name} -> {agent_file_id}")
    
    logger.info(f"[PREPARE_FILES] Final file_id_mapping: {file_id_mapping}")
    return file_id_mapping


def get_download_url(file_id: str) -> Optional[str]:
    """Generate a download URL for a file"""
    managed_file = file_manager.get_file(file_id)
    if not managed_file:
        return None
    return f"/api/files/download/{file_id}"

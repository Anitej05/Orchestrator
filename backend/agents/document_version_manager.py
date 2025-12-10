# agents/document_version_manager.py
# Version control system for document editing with undo/redo capabilities

import os
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DocumentVersionManager:
    """Manages document versions for undo/redo functionality."""
    
    def __init__(self, base_dir: str = "backend/storage/document_versions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.version_index_file = os.path.join(base_dir, "version_index.json")
        self._load_index()
    
    def _load_index(self):
        """Load the version index from disk."""
        if os.path.exists(self.version_index_file):
            with open(self.version_index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save the version index to disk."""
        with open(self.version_index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_document_key(self, file_path: str) -> str:
        """Generate a unique key for a document."""
        return os.path.normpath(file_path).replace(os.sep, '_')
    
    def save_version(self, file_path: str, description: str = "Edit") -> str:
        """
        Save a version of the document.
        Returns the version ID.
        """
        doc_key = self._get_document_key(file_path)
        
        # Initialize document history if not exists
        if doc_key not in self.index:
            self.index[doc_key] = {
                "file_path": file_path,
                "versions": [],
                "current_version": -1
            }
        
        # Create version directory
        version_id = f"v{int(time.time() * 1000)}"
        version_dir = os.path.join(self.base_dir, doc_key, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy the file
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            version_file = os.path.join(version_dir, file_name)
            shutil.copy2(file_path, version_file)
            
            # Save metadata
            metadata = {
                "version_id": version_id,
                "timestamp": time.time(),
                "description": description,
                "file_path": version_file,
                "original_path": file_path
            }
            
            metadata_file = os.path.join(version_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update index
            doc_history = self.index[doc_key]
            
            # If we're not at the latest version, remove future versions
            if doc_history["current_version"] < len(doc_history["versions"]) - 1:
                doc_history["versions"] = doc_history["versions"][:doc_history["current_version"] + 1]
            
            # Add new version
            doc_history["versions"].append(metadata)
            doc_history["current_version"] = len(doc_history["versions"]) - 1
            
            self._save_index()
            
            logger.info(f"Saved version {version_id} for {file_path}: {description}")
            return version_id
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def undo(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Undo the last edit by restoring the previous version.
        Returns metadata of the restored version, or None if no previous version.
        """
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            logger.warning(f"No version history for {file_path}")
            return None
        
        doc_history = self.index[doc_key]
        current_idx = doc_history["current_version"]
        
        if current_idx <= 0:
            logger.warning(f"Already at oldest version for {file_path}")
            return None
        
        # Move to previous version
        prev_idx = current_idx - 1
        prev_version = doc_history["versions"][prev_idx]
        
        # Restore the file
        version_file = prev_version["file_path"]
        if os.path.exists(version_file):
            shutil.copy2(version_file, file_path)
            doc_history["current_version"] = prev_idx
            self._save_index()
            
            logger.info(f"Undone to version {prev_version['version_id']} for {file_path}")
            return prev_version
        else:
            logger.error(f"Version file not found: {version_file}")
            return None
    
    def redo(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Redo the last undone edit.
        Returns metadata of the restored version, or None if no next version.
        """
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            logger.warning(f"No version history for {file_path}")
            return None
        
        doc_history = self.index[doc_key]
        current_idx = doc_history["current_version"]
        
        if current_idx >= len(doc_history["versions"]) - 1:
            logger.warning(f"Already at newest version for {file_path}")
            return None
        
        # Move to next version
        next_idx = current_idx + 1
        next_version = doc_history["versions"][next_idx]
        
        # Restore the file
        version_file = next_version["file_path"]
        if os.path.exists(version_file):
            shutil.copy2(version_file, file_path)
            doc_history["current_version"] = next_idx
            self._save_index()
            
            logger.info(f"Redone to version {next_version['version_id']} for {file_path}")
            return next_version
        else:
            logger.error(f"Version file not found: {version_file}")
            return None
    
    def get_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get the version history for a document."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return []
        
        doc_history = self.index[doc_key]
        return doc_history["versions"]
    
    def can_undo(self, file_path: str) -> bool:
        """Check if undo is available."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return False
        
        return self.index[doc_key]["current_version"] > 0
    
    def can_redo(self, file_path: str) -> bool:
        """Check if redo is available."""
        doc_key = self._get_document_key(file_path)
        
        if doc_key not in self.index:
            return False
        
        doc_history = self.index[doc_key]
        return doc_history["current_version"] < len(doc_history["versions"]) - 1

# Global instance
version_manager = DocumentVersionManager()

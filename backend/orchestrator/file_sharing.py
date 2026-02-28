"""
Smart File Sharing System

The orchestrator intelligently decides when to share files to persistent storage,
based on context and user intent. Users don't need to explicitly say "share this".
"""

import re
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SharingIntentDetector:
    """
    Detects when user wants a file to be persisted/shared.
    
    The orchestrator uses this to intelligently decide when to copy files
    from thread workspace to shared workspace.
    """
    
    # Patterns that indicate sharing intent
    SHARING_KEYWORDS = [
        # Explicit save commands
        r"save (?:this|it|that|the file)",
        r"keep (?:this|it|that)",
        r"store (?:this|it|that)",
        r"remember (?:this|it|that)",
        r"don't delete",
        r"preserve",
        
        # Template/Resource creation
        r"(?:create|make) (?:a |an )?(?:template|blueprint|sample|example)",
        r"(?:save|store) (?:as |for )?(?:template|later|future|next time)",
        
        # Important files
        r"important",
        r"permanent",
        r"persistent",
        r"keep for",
        r"reuse",
        r"use (?:this|it) (?:again|later|next time)",
        
        # Report/Document saving
        r"(?:final |completed |finished )?(?:report|document|analysis)",
        r"(?:save|store) (?:the |this )?(?:report|result|output)",
    ]
    
    # File types that are typically shared
    SHAREABLE_FILE_TYPES = {
        '.template': True,
        '.config': True,
        '.json': True,  # Often config/data files
        '.yaml': True,
        '.yml': True,
        '.md': True,    # Documentation
        '.txt': True,   # Often notes
        '.pdf': True,   # Reports
        '.docx': True,  # Documents
        '.xlsx': True,  # Spreadsheets
    }
    
    # File types that are typically temporary
    TEMPORARY_FILE_TYPES = {
        '.tmp': True,
        '.temp': True,
        '.cache': True,
        '.log': True,   # Unless explicitly requested
    }
    
    @classmethod
    def detect_sharing_intent(cls, user_prompt: str, file_name: str = "") -> tuple[bool, float, str]:
        """
        Analyze user prompt to detect if they want the file shared.
        
        Returns:
            (should_share: bool, confidence: float, reason: str)
        """
        user_prompt_lower = user_prompt.lower()
        
        # Check for explicit sharing keywords
        for pattern in cls.SHARING_KEYWORDS:
            if re.search(pattern, user_prompt_lower):
                return True, 0.9, f"Detected sharing intent: '{pattern}'"
        
        # Check file extension
        if file_name:
            ext = Path(file_name).suffix.lower()
            
            if ext in cls.SHAREABLE_FILE_TYPES:
                # If creating a template/config file, likely want to share
                if any(x in file_name.lower() for x in ['template', 'config', 'setting', 'preference']):
                    return True, 0.7, f"File appears to be a template/config: {file_name}"
            
            if ext in cls.TEMPORARY_FILE_TYPES:
                return False, 0.8, f"File type suggests temporary: {ext}"
        
        # Check for template-related language
        template_indicators = [
            'template', 'blueprint', 'reusable', 'standard', 'format'
        ]
        if any(indicator in user_prompt_lower for indicator in template_indicators):
            return True, 0.6, "Detected template-related language"
        
        # Default: don't share (temporary by default)
        return False, 0.5, "No sharing intent detected - treating as temporary"
    
    @classmethod
    def get_sharing_guidance(cls) -> str:
        """Get guidance for the Brain on when to share files."""
        return """
## INTELLIGENT FILE SHARING

By default, files are saved to the PRIVATE thread workspace (temporary).
Share files to the SHARED workspace when:

**SHARE when user says:**
- "Save this for later"
- "Keep this" / "Don't delete this"
- "Create a template"
- "This is important"
- "I want to reuse this"
- "Remember this"

**SHARE when file is:**
- A template (email template, report template, etc.)
- Configuration/settings
- Important results user references
- User explicitly marks as important

**DON'T SHARE when:**
- Temporary analysis/charts
- Downloaded files for one-time use
- Scratch files
- User doesn't indicate importance
- File type suggests temporary (.tmp, .cache, .log)

**How to share:**
After creating a file, use Python to copy it to shared workspace:
```python
import shutil
shutil.copy('filename', '../shared/{user_id}/filename')
```

Then tell the user: "I've saved this to your shared workspace so it's available in all your conversations."
"""


class AgentWorkspaceInterface:
    """
    Interface for sharing files with agents.
    
    Agents don't directly access shared workspace (by design - isolation),
    but the orchestrator can share files with them when needed.
    """
    
    @staticmethod
    def prepare_file_for_agent(
        file_path: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Prepare a shared file for an agent to use.
        
        Strategy:
        1. Copy from shared workspace to agent's workspace (if needed)
        2. Return file metadata for the agent
        """
        from backend.services.agent_registry_service import agent_registry
        
        source = Path(file_path)
        if not source.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Get agent's workspace
        agent_details = agent_registry.get_agent(agent_id)
        if not agent_details:
            return {
                "success": False,
                "error": f"Agent not found: {agent_id}"
            }
        
        # Agent workspaces are in backend/storage/{agent_id}/
        agent_workspace = Path(f"backend/storage/{agent_id}")
        agent_workspace.mkdir(parents=True, exist_ok=True)
        
        # Copy file to agent workspace
        import shutil
        dest = agent_workspace / source.name
        shutil.copy2(source, dest)
        
        logger.info(f"Shared file with agent {agent_id}: {source.name}")
        
        return {
            "success": True,
            "file_path": str(dest),
            "file_name": source.name,
            "message": f"File shared with {agent_id}"
        }
    
    @staticmethod
    def get_shared_files_for_agent_prompt(shared_files: List[Dict]) -> str:
        """
        Format shared files for inclusion in agent prompt.
        Agents don't have direct access, but orchestrator can tell them about files.
        """
        if not shared_files:
            return ""
        
        lines = ["\n## SHARED FILES (Available from your previous work)"]
        for f in shared_files[:5]:  # Limit to 5 to not overwhelm
            name = f.get("file_name", "Unknown")
            desc = f.get("description", "")
            lines.append(f"- {name}")
            if desc:
                lines.append(f"  ({desc})")
        
        if len(shared_files) > 5:
            lines.append(f"... and {len(shared_files) - 5} more")
        
        return "\n".join(lines)


# Convenience function
def should_share_file(user_prompt: str, file_name: str = "") -> tuple[bool, str]:
    """
    Quick check if a file should be shared.
    
    Usage in Brain:
        should_share, reason = should_share_file(user_prompt, "report.pdf")
        if should_share:
            # Copy to shared workspace
    """
    should_share, confidence, reason = SharingIntentDetector.detect_sharing_intent(
        user_prompt, file_name
    )
    return should_share, f"{reason} (confidence: {confidence:.0%})"

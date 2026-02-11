"""
Content Orchestrator - Manages content flow between orchestrator and agents

This module provides:
1. Automatic content preparation for agents
2. Content ID injection into payloads
3. Agent output capture and registration
4. Lifecycle management hooks
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from backend.services.content_management_service import (
    ContentManagementService,
    UnifiedContentMetadata,
    ContentType,
    ContentSource,
    ContentPriority,
    ContentReference
)

# Helper to get service instance
_content_service = None

def get_content_service():
    global _content_service
    if _content_service is None:
        _content_service = ContentManagementService()
    return _content_service

logger = logging.getLogger("ContentOrchestrator")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ContentOrchestratorConfig:
    """Configuration for content orchestration"""
    
    # Enable/disable automatic content management
    enabled: bool = True
    
    # Size thresholds for automatic artifact creation
    artifact_thresholds: Dict[str, int] = None
    
    # Maximum context tokens for LLM calls
    max_context_tokens: int = 8000
    
    # Auto-upload files to agents that require them
    auto_upload_to_agents: bool = True
    
    # Capture agent output files automatically
    capture_agent_outputs: bool = True
    
    # Verbose logging
    verbose: bool = True
    
    def __post_init__(self):
        if self.artifact_thresholds is None:
            self.artifact_thresholds = {
                "task_result": 2000,
                "canvas_content": 500,
                "screenshot": 100,
                "conversation": 5000,
            }


# Global config
config = ContentOrchestratorConfig()


# =============================================================================
# AGENT FILE INTERFACE DETECTION
# =============================================================================

def agent_requires_file_upload(agent_details: Any, endpoint_path: str) -> bool:
    """
    Check if an agent requires files to be uploaded before operations.
    
    Logic:
    - Agent has an /upload endpoint
    - Current endpoint requires a file_id parameter
    """
    # Check for upload endpoint
    has_upload = any(
        ep.endpoint == '/upload' or ep.endpoint.endswith('/upload')
        for ep in agent_details.endpoints
    )
    
    if not has_upload:
        return False
    
    # Check if current endpoint needs file_id
    current_endpoint = next(
        (ep for ep in agent_details.endpoints if ep.endpoint == endpoint_path),
        None
    )
    
    if not current_endpoint:
        return False
    
    file_params = ['file_id', 'fileId', 'file_identifier', 'document_id', 'content_id']
    for param in current_endpoint.parameters:
        if param.name.lower() in [p.lower() for p in file_params]:
            logger.info(f"Agent {agent_details.id} endpoint {endpoint_path} requires file upload")
            return True
    
    return False


def get_file_parameter_name(endpoint_details: Any) -> Optional[str]:
    """Get the name of the file_id parameter for an endpoint"""
    file_params = ['file_id', 'fileId', 'file_identifier', 'document_id', 'content_id']
    
    for param in endpoint_details.parameters:
        if param.name.lower() in [p.lower() for p in file_params]:
            return param.name
    
    return None


# =============================================================================
# DIRECT FILE UPLOAD HELPER
# =============================================================================

async def _direct_upload_to_agent(
    file_path: str,
    file_name: str,
    agent_base_url: str,
    upload_endpoint: str = "/upload"
) -> Optional[str]:
    """
    Directly upload a file to an agent's upload endpoint.
    
    This is a fallback when the content service fails to handle the file.
    
    Args:
        file_path: Local path to the file
        file_name: Original filename
        agent_base_url: Base URL of the agent
        upload_endpoint: Upload endpoint path (default: /upload)
        
    Returns:
        Agent's file_id if successful, None otherwise
    """
    import httpx
    import mimetypes
    
    if not os.path.exists(file_path):
        logger.error(f"[DIRECT_UPLOAD] File not found: {file_path}")
        return None
    
    # Determine mime type
    mime_type, _ = mimetypes.guess_type(file_name)
    mime_type = mime_type or 'application/octet-stream'
    
    # Read file content
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    upload_url = f"{agent_base_url.rstrip('/')}{upload_endpoint}"
    logger.info(f"[DIRECT_UPLOAD] Uploading {file_name} to {upload_url}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (file_name, file_content, mime_type)}
            response = await client.post(upload_url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[DIRECT_UPLOAD] Success: {result}")
                
                # Extract file_id from response
                if isinstance(result, dict):
                    if 'result' in result and isinstance(result['result'], dict):
                        file_id = result['result'].get('file_id')
                        if file_id:
                            return file_id
                    if 'file_id' in result:
                        return result['file_id']
                    if 'id' in result:
                        return result['id']
                
                logger.warning(f"[DIRECT_UPLOAD] Could not extract file_id from response: {result}")
                return None
            else:
                logger.error(f"[DIRECT_UPLOAD] Failed with status {response.status_code}: {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"[DIRECT_UPLOAD] Exception: {e}")
        return None


# =============================================================================
# CONTENT PREPARATION FOR AGENTS
# =============================================================================

async def prepare_content_for_agent(
    state: Dict[str, Any],
    agent_details: Any,
    endpoint_path: str,
    orchestrator_config: Dict[str, Any]
) -> Tuple[Dict[str, str], str, List[Dict[str, Any]]]:
    """
    Prepare content for a task by uploading to the agent if needed.
    
    Args:
        state: Current orchestrator state
        agent_details: AgentCard for the target agent
        endpoint_path: The endpoint being called
        orchestrator_config: Orchestrator config with user_id etc.
    
    Returns:
        Tuple of (content_id_mapping, enhanced_context, updated_uploaded_files)
    """
    if not config.enabled:
        return {}, "", state.get("uploaded_files", [])
    
    content_service = get_content_service()
    content_id_mapping = {}
    enhanced_context = ""
    
    # Get uploaded files from state
    uploaded_files = state.get("uploaded_files", [])
    
    logger.info(f"[CONTENT_PREP] Preparing content for agent {agent_details.id}, endpoint {endpoint_path}")
    logger.info(f"[CONTENT_PREP] Found {len(uploaded_files)} uploaded files")
    
    if not uploaded_files:
        return content_id_mapping, enhanced_context, uploaded_files
    
    # Check if agent accepts file/content ID (generic check)
    file_param = get_file_parameter_name(next((ep for ep in agent_details.endpoints if ep.endpoint == endpoint_path), None))
    requires_file_id = file_param is not None
    
    # Check if agent requires actual file upload (legacy)
    requires_upload_endpoint = agent_requires_file_upload(agent_details, endpoint_path)
    
    if not requires_file_id and not requires_upload_endpoint:
        # Just provide file context
        enhanced_context = f'''
**Available Files:**
The user has uploaded files. Use the file information below:
```json
{json.dumps(uploaded_files, indent=2)}
```
'''
        return content_id_mapping, enhanced_context, uploaded_files
    
    logger.info(f"[CONTENT_PREP] Agent {agent_details.id} requires content ID (Legacy upload: {requires_upload_endpoint})")
    
    user_id = orchestrator_config.get("configurable", {}).get("user_id", "system")
    connection_config = getattr(agent_details, 'connection_config', None) or {}
    base_url = connection_config.get('base_url', '')
    
    # Process files
    file_info = []
    updated_uploaded_files = []
    
    for file_obj in uploaded_files:
        file_name = file_obj.get('file_name') or file_obj.get('original_name', 'unknown')
        file_path = file_obj.get('file_path') or file_obj.get('stored_path')
        content_id = file_obj.get('file_id') or file_obj.get('content_id')
        
        # Create a copy to update
        updated_file_obj = file_obj.copy()
        
        # 1. Ensure file is registered with CMS
        if not content_id and file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                metadata = await content_service.register_user_upload(
                    file_content=content,
                    filename=file_name,
                    user_id=user_id,
                    thread_id=state.get('thread_id')
                )
                content_id = metadata.id
                logger.info(f"[CONTENT_PREP] Registered file {file_name} -> {content_id}")
            except Exception as e:
                logger.error(f"[CONTENT_PREP] Failed to register file {file_name}: {e}")

        # 2. Handle Agent Specifics
        final_id_for_agent = content_id # Default to CMS ID
        
        if requires_upload_endpoint and base_url and content_id:
            # CMS -> Agent Upload
            try:
                agent_content_id = await content_service.upload_to_agent(
                    content_id=content_id,
                    agent_id=agent_details.id,
                    agent_base_url=base_url
                )
                if agent_content_id:
                    final_id_for_agent = agent_content_id
            except Exception as e:
                logger.error(f"[CONTENT_PREP] Failed upload to agent: {e}")
        
        elif requires_upload_endpoint and not content_id and file_path:
             # Legacy Direct Upload (Fallback)
             try:
                 final_id_for_agent = await _direct_upload_to_agent(file_path, file_name, base_url)
             except Exception:
                 pass
        
        if final_id_for_agent:
            # Map CMS ID (or filename) to Final Agent ID
            if content_id:
                content_id_mapping[content_id] = final_id_for_agent
            content_id_mapping[file_name] = final_id_for_agent
            
            updated_file_obj['file_id'] = final_id_for_agent
            file_info.append({
                "file_name": file_name,
                "file_id": final_id_for_agent,
                "cms_id": content_id
            })
            updated_uploaded_files.append(updated_file_obj)
        else:
            updated_uploaded_files.append(updated_file_obj)

    # Build Context
    if file_info:
        enhanced_context = f'''
**CRITICAL - Content Access:**
The following files are available for this task. 
Start your work by loading these files using the provided IDs.

```json
{json.dumps(file_info, indent=2)}
```

**IMPORTANT:** Use the "file_id" (or "agent_file_id" if distinct) when calling tool endpoints.
'''
    return content_id_mapping, enhanced_context, updated_uploaded_files


def inject_content_id_into_payload(
    payload: Dict[str, Any],
    content_id_mapping: Dict[str, str],
    endpoint_details: Any,
    uploaded_files: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Inject the correct content_id into a payload.
    
    This ensures the file_id is correct even if the LLM generates an incorrect value.
    """
    file_param_name = get_file_parameter_name(endpoint_details)
    
    if not file_param_name:
        return payload
    
    logger.info(f"[CONTENT_INJECT] Injecting {file_param_name} into payload")
    logger.info(f"[CONTENT_INJECT] content_id_mapping={content_id_mapping}")
    logger.info(f"[CONTENT_INJECT] uploaded_files={uploaded_files}")
    logger.info(f"[CONTENT_INJECT] payload before injection={payload}")
    
    if not content_id_mapping:
        logger.warning(f"[CONTENT_INJECT] No content_id_mapping available")
        return payload
    
    # Try to find by filename first
    for file_obj in uploaded_files:
        file_name = file_obj.get('file_name', '')
        if file_name in content_id_mapping:
            agent_content_id = content_id_mapping[file_name]
            payload[file_param_name] = agent_content_id
            logger.info(f"[CONTENT_INJECT] Injected {agent_content_id} for {file_name}")
            return payload
    
    # Use first available mapping
    first_id = list(content_id_mapping.values())[0]
    payload[file_param_name] = first_id
    logger.info(f"[CONTENT_INJECT] Injected first available: {first_id}")
    
    return payload


# =============================================================================
# AGENT OUTPUT CAPTURE
# =============================================================================

async def capture_agent_outputs(
    response_data: Dict[str, Any],
    agent_id: str,
    user_id: str,
    thread_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Capture any files/content generated by an agent from its response.
    
    Looks for common patterns:
    - file_url, download_url, output_file
    - base64 encoded data
    - file paths
    """
    import base64
    import httpx
    
    if not config.capture_agent_outputs:
        return []
    
    content_service = get_content_service()
    captured = []
    
    # Patterns that indicate file outputs
    file_indicators = [
        'file_url', 'download_url', 'output_file', 'generated_file',
        'image_url', 'document_url', 'result_file', 'attachment',
        'screenshot_url', 'export_url'
    ]
    
    def search_for_files(data: Any, path: str = "") -> List[Tuple[str, Any]]:
        """Recursively search for file indicators"""
        results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if any(ind in key.lower() for ind in file_indicators):
                    results.append((current_path, value))
                
                results.extend(search_for_files(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                results.extend(search_for_files(item, f"{path}[{i}]"))
        
        return results
    
    file_outputs = search_for_files(response_data)
    
    for path, value in file_outputs:
        try:
            if isinstance(value, str):
                # URL
                if value.startswith('http://') or value.startswith('https://'):
                    async with httpx.AsyncClient() as client:
                        response = await client.get(value, timeout=30.0)
                        if response.status_code == 200:
                            filename = value.split('/')[-1].split('?')[0]
                            if not filename or '.' not in filename:
                                filename = f"agent_output_{len(captured)}.bin"
                            
                            metadata = await content_service.register_agent_output(
                                content=response.content,
                                name=filename,
                                agent_id=agent_id,
                                user_id=user_id,
                                thread_id=thread_id
                            )
                            captured.append(metadata.to_file_object())
                
                # Base64 data URL
                elif len(value) > 100 and ';base64,' in value:
                    parts = value.split(';base64,')
                    if len(parts) == 2:
                        mime_type = parts[0].replace('data:', '')
                        content = base64.b64decode(parts[1])
                        
                        ext_map = {
                            'image/png': '.png',
                            'image/jpeg': '.jpg',
                            'application/pdf': '.pdf',
                            'text/plain': '.txt'
                        }
                        ext = ext_map.get(mime_type, '.bin')
                        filename = f"agent_output_{len(captured)}{ext}"
                        
                        metadata = await content_service.register_agent_output(
                            content=content,
                            name=filename,
                            agent_id=agent_id,
                            user_id=user_id,
                            thread_id=thread_id
                        )
                        captured.append(metadata.to_file_object())
        
        except Exception as e:
            logger.warning(f"Failed to capture output from {path}: {e}")
    
    if captured:
        logger.info(f"Captured {len(captured)} outputs from agent {agent_id}")
    
    return captured


# =============================================================================
# STATE COMPRESSION FOR SAVING
# =============================================================================

async def compress_state_for_saving(
    state: Dict[str, Any],
    thread_id: str
) -> Dict[str, Any]:
    """
    Compress state before saving to conversation history.
    Large fields are stored as artifacts and replaced with references.
    """
    if not config.enabled:
        return state
    
    content_service = get_content_service()
    compressed = dict(state)
    content_refs = {}
    
    # Fields to potentially compress
    compressible_fields = [
        ("canvas_content", ContentType.CANVAS),
        ("completed_tasks", ContentType.RESULT),
        ("task_agent_pairs", ContentType.DATA),
        ("task_plan", ContentType.PLAN),
    ]
    
    for field_name, content_type in compressible_fields:
        value = compressed.get(field_name)
        if not value:
            continue
        
        value_str = json.dumps(value, default=str) if not isinstance(value, str) else value
        threshold = config.artifact_thresholds.get(field_name, 2000)
        
        if len(value_str) > threshold:
            # Store as artifact
            metadata = await content_service.register_artifact(
                content=value,
                name=f"state_{field_name}_{thread_id}",
                content_type=content_type,
                thread_id=thread_id,
                priority=ContentPriority.MEDIUM
            )
            
            content_refs[field_name] = {
                "id": metadata.id,
                "type": content_type.value,
                "size": metadata.size_bytes
            }
            
            # Replace with placeholder
            if field_name == "completed_tasks" and isinstance(value, list):
                compressed[field_name] = [
                    {"task_name": t.get("task_name"), "status": "completed"}
                    for t in value
                ]
            else:
                compressed[field_name] = f"[CONTENT:{metadata.id}]"
            
            logger.info(f"Compressed {field_name} to artifact {metadata.id}")
    
    if content_refs:
        compressed["_content_refs"] = content_refs
    
    return compressed


async def expand_state_from_saved(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expand content references when loading a saved state.
    """
    if not config.enabled:
        return state
    
    content_refs = state.pop("_content_refs", {})
    if not content_refs:
        return state
    
    content_service = get_content_service()
    expanded = dict(state)
    
    for field_name, ref_info in content_refs.items():
        content_id = ref_info.get("id")
        if not content_id:
            continue
        
        result = content_service.get_content(content_id)
        if result:
            metadata, content = result
            expanded[field_name] = content
            logger.info(f"Expanded content {content_id} for field {field_name}")
        else:
            logger.warning(f"Failed to expand content {content_id} for {field_name}")
    
    return expanded


# =============================================================================
# CONTEXT OPTIMIZATION FOR LLM
# =============================================================================

def get_optimized_llm_context(
    state: Dict[str, Any],
    thread_id: str,
    focus_fields: Optional[List[str]] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get optimized context for an LLM call.
    
    Returns:
        {
            "context": Optimized context string,
            "content_refs": List of content references,
            "tokens_used": Estimated tokens,
            "tokens_saved": Tokens saved by compression
        }
    """
    if not config.enabled:
        return {
            "context": json.dumps(state, default=str),
            "content_refs": [],
            "tokens_used": len(json.dumps(state, default=str)) // 4,
            "tokens_saved": 0
        }
    
    content_service = get_content_service()
    result = content_service.get_optimized_context(
        thread_id=thread_id,
        max_tokens=max_tokens or config.max_context_tokens
    )
    
    return {
        "context": result["context_string"],
        "content_refs": result["references"],
        "tokens_used": len(result["context_string"]) // 4,
        "tokens_saved": result["tokens_saved"]
    }


# =============================================================================
# ORCHESTRATOR HOOKS
# =============================================================================

class ContentOrchestratorHooks:
    """
    Hooks that can be called from orchestrator nodes to automatically
    manage content.
    """
    
    @staticmethod
    async def on_task_complete(
        task_name: str,
        result: Dict[str, Any],
        thread_id: str
    ) -> Dict[str, Any]:
        """Called when a task completes. Compresses large results."""
        if not config.enabled:
            return result
        
        result_str = json.dumps(result, default=str)
        threshold = config.artifact_thresholds.get("task_result", 2000)
        
        if len(result_str) < threshold:
            return result
        
        content_service = get_content_service()
        metadata = await content_service.register_artifact(
            content=result,
            name=f"task_result_{task_name}",
            content_type=ContentType.RESULT,
            thread_id=thread_id,
            description=f"Result for task: {task_name}"
        )
        
        logger.info(f"Compressed task result '{task_name}' to {metadata.id}")
        
        return {
            "_content_ref": metadata.to_reference().to_dict(),
            "task_name": task_name,
            "status": result.get("status", "completed"),
            "summary": _generate_result_summary(result)
        }
    
    @staticmethod
    async def on_canvas_generated(
        canvas_content: str,
        canvas_type: str,
        thread_id: str
    ) -> Dict[str, Any]:
        """Called when canvas content is generated."""
        if not config.enabled or not canvas_content:
            return {"content": canvas_content, "type": canvas_type}
        
        content_service = get_content_service()
        metadata = await content_service.register_artifact(
            content=canvas_content,
            name=f"canvas_{canvas_type}_{thread_id}",
            content_type=ContentType.CANVAS,
            thread_id=thread_id,
            priority=ContentPriority.HIGH
        )
        
        logger.info(f"Stored canvas as {metadata.id}")
        
        return {
            "_content_ref": metadata.to_reference().to_dict(),
            "canvas_type": canvas_type,
            "preview": canvas_content[:200] + "..." if len(canvas_content) > 200 else canvas_content
        }
    
    @staticmethod
    async def on_screenshot_captured(
        screenshot_base64: str,
        step_name: str,
        thread_id: str
    ) -> Dict[str, Any]:
        """Called when a screenshot is captured."""
        if not config.enabled or not screenshot_base64:
            return {"base64": screenshot_base64}
        
        import base64
        content = base64.b64decode(screenshot_base64)
        
        content_service = get_content_service()
        metadata = await content_service.register_content(
            content=content,
            name=f"screenshot_{step_name}.png",
            source=ContentSource.BROWSER_CAPTURE,
            thread_id=thread_id,
            content_type=ContentType.SCREENSHOT,
            priority=ContentPriority.LOW,
            ttl_hours=24
        )
        
        logger.info(f"Stored screenshot as {metadata.id}")
        
        return {
            "_content_ref": metadata.to_reference().to_dict(),
            "step": step_name
        }
    
    @staticmethod
    async def before_save(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        """Called before saving conversation. Compresses state."""
        return await compress_state_for_saving(state, thread_id)
    
    @staticmethod
    async def after_load(state: Dict[str, Any]) -> Dict[str, Any]:
        """Called after loading conversation. Expands content."""
        return await expand_state_from_saved(state)


def _generate_result_summary(result: Dict[str, Any]) -> str:
    """Generate a brief summary of a task result"""
    if isinstance(result, dict):
        status = result.get('status', 'completed')
        if 'error' in result:
            return f"Failed: {str(result['error'])[:100]}"
        if 'summary' in result:
            return str(result['summary'])[:200]
        if 'result' in result:
            return f"Completed with result: {str(result['result'])[:100]}"
        return f"Status: {status}"
    return str(result)[:200]


# Global hooks instance
hooks = ContentOrchestratorHooks()

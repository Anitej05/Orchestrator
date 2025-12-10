"""
File Integration Module for Orchestrator

This module provides functions to integrate the file management service
with the orchestrator's task execution pipeline.

Key Features:
1. Automatic file upload to agents that require it
2. File ID injection into payloads
3. Agent output file capture
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


def agent_requires_file_upload(agent_details: Any, endpoint_path: str) -> bool:
    """
    Check if an agent requires files to be uploaded before other operations.
    
    ROBUST LOGIC:
    If an agent has an /upload endpoint AND the current endpoint requires a file_id parameter,
    then files should be uploaded first.
    
    Args:
        agent_details: AgentCard with endpoint information
        endpoint_path: The endpoint being called
    
    Returns:
        True if files should be uploaded first
    """
    # Check if agent has an upload endpoint
    has_upload_endpoint = any(
        ep.endpoint == '/upload' or ep.endpoint.endswith('/upload')
        for ep in agent_details.endpoints
    )
    
    if not has_upload_endpoint:
        return False
    
    # Check if current endpoint requires file_id
    current_endpoint = next(
        (ep for ep in agent_details.endpoints if ep.endpoint == endpoint_path),
        None
    )
    
    if not current_endpoint:
        return False
    
    # Check if any parameter is named file_id or similar
    file_params = ['file_id', 'fileId', 'file_identifier', 'document_id']
    for param in current_endpoint.parameters:
        if param.name.lower() in [p.lower() for p in file_params]:
            logger.info(f"[FILE_UPLOAD_CHECK] ✅ Agent {agent_details.id} endpoint {endpoint_path} requires file upload (has /upload endpoint and {param.name} parameter)")
            return True
    
    logger.info(f"[FILE_UPLOAD_CHECK] ❌ Agent {agent_details.id} endpoint {endpoint_path} does not require file upload")
    return False


def get_file_parameter_name(endpoint_details: Any) -> Optional[str]:
    """
    Get the name of the file_id parameter for an endpoint.
    
    Args:
        endpoint_details: EndpointDetail with parameters
    
    Returns:
        Parameter name if found, None otherwise
    """
    file_params = ['file_id', 'fileId', 'file_identifier', 'document_id']
    
    for param in endpoint_details.parameters:
        if param.name.lower() in [p.lower() for p in file_params]:
            return param.name
    
    return None


async def prepare_files_for_task(
    state: Dict[str, Any],
    agent_details: Any,
    endpoint_path: str,
    config: Dict[str, Any]
) -> Tuple[Dict[str, str], str]:
    """
    Prepare files for a task by uploading them to the agent if needed.
    
    This is the main integration point that:
    1. Checks if the agent needs file uploads
    2. Uploads files to the agent
    3. Returns file ID mappings and enhanced file context
    
    Args:
        state: Current orchestrator state
        agent_details: AgentCard for the target agent
        endpoint_path: The endpoint being called
        config: Orchestrator config with user_id etc.
    
    Returns:
        Tuple of (file_id_mapping, enhanced_file_context)
    """
    from services.file_management_service import file_manager, prepare_files_for_agent
    
    file_id_mapping = {}
    enhanced_context = ""
    
    uploaded_files = state.get("uploaded_files", [])
    logger.info(f"[FILE_INTEGRATION] prepare_files_for_task called for agent {agent_details.id}, endpoint {endpoint_path}")
    logger.info(f"[FILE_INTEGRATION] uploaded_files: {uploaded_files}")
    
    if not uploaded_files:
        logger.info("[FILE_INTEGRATION] No uploaded files in state")
        return file_id_mapping, enhanced_context
    
    # Check if this agent requires file upload
    requires_upload = agent_requires_file_upload(agent_details, endpoint_path)
    logger.info(f"[FILE_INTEGRATION] Agent requires file upload: {requires_upload}")
    logger.info(f"[FILE_INTEGRATION] agent_details.id={agent_details.id}, endpoint_path={endpoint_path}")
    
    if not requires_upload:
        # Just provide file paths as context
        logger.info(f"[FILE_INTEGRATION] Agent does not require file upload, providing file context only")
        enhanced_context = f'''
        **Available File Context:**
        The user has uploaded files. Use the file information below:
        ```json
        {json.dumps(uploaded_files, indent=2)}
        ```
        '''
        return file_id_mapping, enhanced_context
    
    # Agent requires file upload - upload files first
    logger.info(f"[FILE_INTEGRATION] Agent {agent_details.id} requires file upload. Preparing files...")
    logger.info(f"[FILE_INTEGRATION] About to call prepare_files_for_agent...")
    
    user_id = config.get("configurable", {}).get("user_id", "system")
    connection_config = getattr(agent_details, 'connection_config', None) or {}
    
    # Upload files to agent
    file_id_mapping = await prepare_files_for_agent(
        files=uploaded_files,
        agent_id=agent_details.id,
        agent_config=connection_config,
        user_id=user_id
    )
    
    logger.info(f"[FILE_INTEGRATION] file_id_mapping result = {file_id_mapping}")
    
    if file_id_mapping:
        # Build enhanced context with agent file IDs
        file_info = []
        for file_obj in uploaded_files:
            file_name = file_obj.get('file_name', 'unknown')
            agent_file_id = file_id_mapping.get(file_name)
            
            if agent_file_id:
                file_info.append({
                    "file_name": file_name,
                    "file_id": agent_file_id,  # This is the agent's file_id!
                    "file_type": file_obj.get('file_type', 'document')
                })
        
        enhanced_context = f'''
        **CRITICAL - File IDs for this Agent:**
        The following files have been uploaded to this agent. You MUST use the provided file_id values.
        
        ```json
        {json.dumps(file_info, indent=2)}
        ```
        
        **IMPORTANT:** Use the "file_id" value from above when the endpoint requires a file_id parameter.
        Do NOT use file paths or file names - use the exact file_id string provided.
        '''
        
        logger.info(f"Files prepared for agent {agent_details.id}: {file_id_mapping}")
    else:
        logger.warning(f"Failed to upload files to agent {agent_details.id}")
        enhanced_context = f'''
        **Available File Context (upload failed):**
        Files could not be uploaded to the agent. The following files are available:
        ```json
        {json.dumps(uploaded_files, indent=2)}
        ```
        '''
    
    return file_id_mapping, enhanced_context


def inject_file_id_into_payload(
    payload: Dict[str, Any],
    file_id_mapping: Dict[str, str],
    endpoint_details: Any,
    uploaded_files: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Inject the correct file_id into a payload if needed.
    
    This is a safety net that ensures the file_id is correct even if
    the LLM generates an incorrect value.
    
    Args:
        payload: The generated payload
        file_id_mapping: Mapping of file names to agent file IDs
        endpoint_details: EndpointDetail with parameters
        uploaded_files: List of uploaded file objects
    
    Returns:
        Modified payload with correct file_id
    """
    file_param_name = get_file_parameter_name(endpoint_details)
    
    logger.info(f"[FILE_ID_INJECTION] file_param_name={file_param_name}, file_id_mapping keys={list(file_id_mapping.keys()) if file_id_mapping else None}")
    logger.info(f"[FILE_ID_INJECTION] file_id_mapping values={list(file_id_mapping.values()) if file_id_mapping else None}")
    logger.info(f"[FILE_ID_INJECTION] current payload={payload}")
    logger.info(f"[FILE_ID_INJECTION] uploaded_files={[f.get('file_name') for f in uploaded_files]}")
    
    if not file_param_name:
        logger.info(f"[FILE_ID_INJECTION] No file parameter found in endpoint")
        return payload
    
    # Check if payload has a file_id that needs correction
    current_value = payload.get(file_param_name)
    logger.info(f"[FILE_ID_INJECTION] Current {file_param_name} value in payload: {current_value}")
    
    # ALWAYS inject the correct file_id if we have a mapping
    if file_id_mapping:
        # Try to find by filename first
        for file_obj in uploaded_files:
            file_name = file_obj.get('file_name', '')
            if file_name in file_id_mapping:
                agent_file_id = file_id_mapping[file_name]
                payload[file_param_name] = agent_file_id
                logger.info(f"[FILE_ID_INJECTION] ✅ Injected agent file_id '{agent_file_id}' for file '{file_name}'")
                return payload
        
        # If filename not found, use first available mapping value
        first_file_id = list(file_id_mapping.values())[0]
        payload[file_param_name] = first_file_id
        logger.info(f"[FILE_ID_INJECTION] ✅ Injected first available agent file_id '{first_file_id}'")
    else:
        logger.warning(f"[FILE_ID_INJECTION] ⚠️ No file_id_mapping available, cannot inject correct file_id")
    
    return payload


async def capture_agent_output_files(
    response_data: Dict[str, Any],
    agent_id: str,
    user_id: str,
    thread_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Capture any files generated by an agent from its response.
    
    This looks for common patterns in agent responses that indicate
    file outputs (URLs, base64 data, file paths).
    
    Args:
        response_data: Agent's response data
        agent_id: ID of the agent
        user_id: User who owns the files
        thread_id: Optional conversation thread
    
    Returns:
        List of captured file metadata
    """
    from services.file_management_service import file_manager
    import base64
    import httpx
    
    captured_files = []
    
    # Look for common file output patterns
    file_indicators = [
        'file_url', 'download_url', 'output_file', 'generated_file',
        'image_url', 'document_url', 'result_file', 'attachment'
    ]
    
    def search_for_files(data: Any, path: str = "") -> List[Tuple[str, Any]]:
        """Recursively search for file indicators in response"""
        results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this key indicates a file
                if any(indicator in key.lower() for indicator in file_indicators):
                    results.append((current_path, value))
                
                # Recurse into nested structures
                results.extend(search_for_files(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                results.extend(search_for_files(item, f"{path}[{i}]"))
        
        return results
    
    # Search response for file outputs
    file_outputs = search_for_files(response_data)
    
    for path, value in file_outputs:
        try:
            if isinstance(value, str):
                # Check if it's a URL
                if value.startswith('http://') or value.startswith('https://'):
                    # Download the file
                    async with httpx.AsyncClient() as client:
                        response = await client.get(value, timeout=30.0)
                        if response.status_code == 200:
                            # Determine filename from URL or content-disposition
                            filename = value.split('/')[-1].split('?')[0]
                            if not filename or '.' not in filename:
                                filename = f"agent_output_{len(captured_files)}.bin"
                            
                            managed_file = await file_manager.register_agent_output(
                                file_content=response.content,
                                filename=filename,
                                agent_id=agent_id,
                                user_id=user_id,
                                thread_id=thread_id
                            )
                            captured_files.append(managed_file.to_dict())
                
                # Check if it's base64 encoded
                elif len(value) > 100 and ';base64,' in value:
                    # Data URL format: data:mime/type;base64,XXXX
                    parts = value.split(';base64,')
                    if len(parts) == 2:
                        mime_type = parts[0].replace('data:', '')
                        file_content = base64.b64decode(parts[1])
                        
                        # Determine extension from mime type
                        ext_map = {
                            'image/png': '.png',
                            'image/jpeg': '.jpg',
                            'application/pdf': '.pdf',
                            'text/plain': '.txt'
                        }
                        ext = ext_map.get(mime_type, '.bin')
                        filename = f"agent_output_{len(captured_files)}{ext}"
                        
                        managed_file = await file_manager.register_agent_output(
                            file_content=file_content,
                            filename=filename,
                            agent_id=agent_id,
                            user_id=user_id,
                            thread_id=thread_id,
                            mime_type=mime_type
                        )
                        captured_files.append(managed_file.to_dict())
        
        except Exception as e:
            logger.warning(f"Failed to capture file from {path}: {e}")
    
    if captured_files:
        logger.info(f"Captured {len(captured_files)} output files from agent {agent_id}")
    
    return captured_files

"""
Creation Tools - Standard LangChain tools for creating content.
Replaces the keyword-based creation_handler.py.
"""

import os
import json
import logging
import httpx
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger("CreationTools")

# --- Schemas ---

class CreateDocumentSchema(BaseModel):
    """Parameters for creating a document."""
    content: str = Field(..., description="The text content to put in the document")
    file_name: str = Field(..., description="Name of the file to create (e.g. 'report.docx')")
    file_type: str = Field(
        ..., 
        description="Format of the file", 
        pattern="^(pdf|docx|txt)$"
    )

class CreateSpreadsheetSchema(BaseModel):
    """Parameters for creating a spreadsheet."""
    data: Any = Field(..., description="The data to write. Can be a CSV string or a list of dictionaries (rows).")
    filename: str = Field(..., description="Name of the spreadsheet file (e.g. 'sales.xlsx')")
    file_format: str = Field(
        "xlsx", 
        description="Format of the spreadsheet",
        pattern="^(xlsx|csv)$"
    )

# --- Tools ---

class CreateDocumentTool(BaseTool):
    name: str = "create_document"
    description: str = (
        "Create a text-based document (PDF, DOCX, TXT) with specific content. "
        "Use this tool when you need to generate a report, essay, or text file "
        "from provided content. Do not use for spreadsheets."
    )
    args_schema: Type[BaseModel] = CreateDocumentSchema

    def _run(self, content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """Synchronous run not implemented, strictly async."""
        raise NotImplementedError("Use ainvoke()")

    async def _arun(self, content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """Execute document creation via API."""
        return await _execute_creation_api(
            endpoint_type="documents",
            payload={
                "content": content,
                "file_name": file_name,
                "file_type": file_type
            }
        )

class CreateSpreadsheetTool(BaseTool):
    name: str = "create_spreadsheet"
    description: str = (
        "Create a spreadsheet (XLSX, CSV) from provided data. "
        "Use this tool ONLY when you have the raw data explicitly available "
        "(e.g. you generated a list of items). "
        "Do NOT use this if the data is inside an Agent (use the Agent instead)."
    )
    args_schema: Type[BaseModel] = CreateSpreadsheetSchema

    def _run(self, data: Any, filename: str, file_format: str = "xlsx") -> Dict[str, Any]:
        raise NotImplementedError("Use ainvoke()")

    async def _arun(self, data: Any, filename: str, file_format: str = "xlsx") -> Dict[str, Any]:
        """Execute spreadsheet creation via API."""
        return await _execute_creation_api(
            endpoint_type="spreadsheets",
            payload={
                "data": data,
                "filename": filename,
                "file_format": file_format
            }
        )

# --- Helper ---

async def _execute_creation_api(endpoint_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to call local API."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_base_url = os.getenv('ORCHESTRATOR_API_URL', 'http://localhost:8000').rstrip('/')
    endpoint = f'{api_base_url}/api/{endpoint_type}/create'
    
    logger.info(f"🔧 [CreationTool] Calling {endpoint} with keys: {list(payload.keys())}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers={'Content-Type': 'application/json', 'X-Internal-Request': 'true'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ [CreationTool] Success: {result.get('file_path')}")
                return {
                    "success": True,
                    "result": result,
                    "file_path": result.get('file_path'),
                    "preview_url": result.get('preview_url')
                }
            else:
                logger.error(f"❌ [CreationTool] Failed: {response.text}")
                return {"success": False, "error": response.text}
                
    except Exception as e:
        logger.error(f"❌ [CreationTool] Exception: {e}")
        return {"success": False, "error": str(e)}

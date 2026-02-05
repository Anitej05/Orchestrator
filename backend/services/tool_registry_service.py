"""
Tool Registry Service

Centralizes tool discovery, registration, and execution.
Replaces backend/orchestrator/tool_discovery.py and tool_registry.py.
"""

import json
import logging
import importlib
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

from services.telemetry_service import telemetry_service as telemetry

logger = logging.getLogger("ToolRegistryService")

# Paths
BACKEND_DIR = Path(__file__).parent.parent
TOOLS_DIR = BACKEND_DIR / "tools"
TOOL_ENTRIES_DIR = TOOLS_DIR / "tool_entries"

@dataclass
class ToolDefinition:
    """Represents a tool's metadata."""
    function_name: str
    display_name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]
    use_when: str
    not_for: str
    keywords: Set[str]
    tool_instance: Optional[BaseTool] = None  # The actual executable tool

class ToolRegistryService:
    """
    Service for managing all system tools.
    Handles discovery from JSON, registration of Python objects, and execution.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}  # Map function_name -> ToolDefinition
        self._lock = threading.RLock()
        self._initialized = False
        self._failed_imports = []

    def initialize(self):
        """Discover tools from JSON and load Python implementations."""
        if self._initialized:
            return
            
        with self._lock:
            # 1. Load Metadata from JSON
            self._discover_json_entries()
            
            # 2. Load Python Implementations
            self._load_tool_implementations()
            
            self._initialized = True
            logger.info(f"ToolRegistryService initialized with {len(self._tools)} tools.")
            if self._failed_imports:
                logger.warning(f"Failed to import tools: {self._failed_imports}")

    def _discover_json_entries(self):
        """Scan tool_entries/ for .json definitions."""
        if not TOOL_ENTRIES_DIR.exists():
            logger.warning(f"Tool entries directory not found: {TOOL_ENTRIES_DIR}")
            return

        for json_path in TOOL_ENTRIES_DIR.glob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                category = data.get("category", "other")
                for t_data in data.get("tools", []):
                    name = t_data["function_name"]
                    
                    # Store definitions
                    self._tools[name] = ToolDefinition(
                        function_name=name,
                        display_name=t_data.get("display_name", name),
                        description=t_data.get("description", ""),
                        category=category,
                        parameters=t_data.get("parameters", []),
                        use_when=t_data.get("use_when", ""),
                        not_for=t_data.get("not_for", ""),
                        keywords=set(kw.lower() for kw in t_data.get("keywords", [])),
                        tool_instance=None
                    )
            except Exception as e:
                logger.error(f"Failed to load tool entry {json_path}: {e}")

    def _load_tool_implementations(self):
        """Import python modules and link generic tools to definitions."""
        # Mapping of Category -> Python Module
        # We assume standard locations, but this could be dynamic
        modules_to_load = [
            ("finance", "tools.finance_tools"),
            ("news", "tools.news_tools"),
            ("wiki", "tools.wiki_tools"),
            ("search", "tools.search_tools"),
            ("image", "tools.image_tools"),
            ("creation", "tools.creation_tools"), 
        ]

        for cat, mod_name in modules_to_load:
            try:
                mod = importlib.import_module(mod_name)
                
                # Check for registered tools in our list matching this category
                # OR just check for attributes matching known function names
                for name, tool_def in self._tools.items():
                    if tool_def.category == cat:
                        # Try to find the function in the module
                        func = getattr(mod, name, None)
                        if func:
                            if isinstance(func, BaseTool):
                                tool_def.tool_instance = func
                            # If it's a creation tool object (like CreateDocumentTool()), strictly check name
                            elif hasattr(func, 'name') and func.name == name:
                                tool_def.tool_instance = func
                            # If it's just a function, we might need to wrap it? 
                            # Assuming tools are already LangChain BaseTool or @tool decorated
                            else:
                                # Check if it acts like a tool
                                if hasattr(func, 'invoke'):
                                    tool_def.tool_instance = func
                                    
                # Specialized loading for some tools that might need instantiation
                if cat == "creation":
                    from tools.creation_tools import CreateDocumentTool, CreateSpreadsheetTool
                    # Manually map creation tools if not found by name
                    # (definitions usually use snake_case function names, classes might differ)
                    if "create_document" in self._tools:
                        self._tools["create_document"].tool_instance = CreateDocumentTool()
                    if "create_spreadsheet" in self._tools:
                         self._tools["create_spreadsheet"].tool_instance = CreateSpreadsheetTool()

            except ImportError as e:
                logger.warning(f"Could not import module {mod_name}: {e}")
                self._failed_imports.append(mod_name)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get executable tool instance by name."""
        self.initialize()
        definition = self._tools.get(name)
        if definition:
            return definition.tool_instance
        # Fallback: Check case-insensitive
        for t_name, t_def in self._tools.items():
            if t_name.lower() == name.lower():
                return t_def.tool_instance
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return full list of tools metadata."""
        self.initialize()
        return [
            {
                "name": t.function_name,
                "display_name": t.display_name,
                "category": t.category,
                "description": t.description,
                "use_when": t.use_when,
                "parameters": t.parameters
            }
            for t in self._tools.values()
            if t.tool_instance is not None # Only return executable tools
        ]

    def get_tool_prompt_summary(self) -> str:
        """Generate a concise summary for LLM context."""
        self.initialize()
        summary = ["AVAILABLE TOOLS:\n"]
        by_cat = {}
        for t in self._tools.values():
            if t.tool_instance:
                by_cat.setdefault(t.category, []).append(t)
        
        for cat, tools in sorted(by_cat.items()):
            summary.append(f"## {cat.upper()}")
            for t in tools:
                summary.append(f"- {t.function_name}: {t.description}")
                if t.use_when:
                    summary.append(f"  Use when: {t.use_when}")
        return "\n".join(summary)

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with telemetry."""
        self.initialize()
        
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool '{tool_name}' not found."
            logger.error(error_msg)
            telemetry.log_error("tool_execution", error_msg, {"tool": tool_name})
            return {"success": False, "error": error_msg}
        
        # Telemetry Start
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"🔧 Executing tool: {tool_name}")
            
            # Simple parameter filtering could go here if needed
            
            # Execute
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(parameters)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.invoke, parameters)
                
            # Telemetry Success
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            telemetry.log_tool_call(tool_name, success=True, duration_ms=duration)
            
            return {
                "success": True, 
                "result": result
            }
            
        except Exception as e:
            # Telemetry Failure
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            telemetry.log_tool_call(tool_name, success=False, duration_ms=duration)
            telemetry.log_error("tool_execution", str(e), {"tool": tool_name})
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False, 
                "error": str(e)
            }

# Singleton
tool_registry = ToolRegistryService()

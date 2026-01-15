"""
Tool Discovery Engine - Auto-discovery and validation of tools from tool_entries/.

This module provides:
1. Automatic discovery of tool definitions from JSON files
2. Signature extraction from @tool decorated functions
3. Validation of JSON definitions against Python implementations
4. Tool card generation for orchestrator routing
"""

import json
import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool

logger = logging.getLogger("AgentOrchestrator")

# Paths
BACKEND_DIR = Path(__file__).parent.parent
TOOLS_DIR = BACKEND_DIR / "tools"
TOOL_ENTRIES_DIR = TOOLS_DIR / "tool_entries"


@dataclass
class ToolParameter:
    """Represents a tool parameter from JSON or introspection."""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class ToolDefinition:
    """Represents a single tool's complete definition."""
    function_name: str
    display_name: str
    description: str
    use_when: str
    not_for: str
    parameters: List[ToolParameter]
    returns: str
    example_queries: List[str]
    keywords: Set[str]
    module_id: str
    category: str


@dataclass
class ToolModule:
    """Represents a tool module (e.g., finance_tools) with all its tools."""
    id: str
    name: str
    description: str
    version: str
    category: str
    requires_env: List[str]
    tools: List[ToolDefinition]


class ToolDiscovery:
    """
    Auto-discover tools from tool_entries/ and Python modules.
    Provides validation and card generation for orchestrator routing.
    """
    
    def __init__(self):
        self._modules: Dict[str, ToolModule] = {}
        self._tools_by_name: Dict[str, ToolDefinition] = {}
        self._tools_by_keyword: Dict[str, List[ToolDefinition]] = {}
        self._initialized = False
    
    def discover_all(self) -> List[ToolModule]:
        """
        Scan tool_entries/*.json and load all tool definitions.
        Returns list of ToolModule objects.
        """
        if self._initialized:
            return list(self._modules.values())
        
        if not TOOL_ENTRIES_DIR.exists():
            logger.warning(f"Tool entries directory not found: {TOOL_ENTRIES_DIR}")
            return []
        
        json_files = list(TOOL_ENTRIES_DIR.glob("*.json"))
        logger.info(f"🔧 Discovering tools from {len(json_files)} JSON files...")
        
        for json_path in json_files:
            try:
                module = self._load_tool_module(json_path)
                if module:
                    self._modules[module.id] = module
                    
                    # Index tools by name and keyword
                    for tool in module.tools:
                        self._tools_by_name[tool.function_name] = tool
                        for keyword in tool.keywords:
                            if keyword not in self._tools_by_keyword:
                                self._tools_by_keyword[keyword] = []
                            self._tools_by_keyword[keyword].append(tool)
                    
                    logger.info(f"✅ Loaded {len(module.tools)} tools from {json_path.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {json_path.name}: {e}")
        
        self._initialized = True
        logger.info(f"🎉 Tool discovery complete: {len(self._tools_by_name)} tools across {len(self._modules)} modules")
        
        return list(self._modules.values())
    
    def _load_tool_module(self, json_path: Path) -> Optional[ToolModule]:
        """Load a single tool module from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tools = []
        for tool_data in data.get("tools", []):
            params = []
            for param_data in tool_data.get("parameters", []):
                params.append(ToolParameter(
                    name=param_data["name"],
                    type=param_data.get("type", "string"),
                    required=param_data.get("required", True),
                    default=param_data.get("default"),
                    description=param_data.get("description", "")
                ))
            
            tools.append(ToolDefinition(
                function_name=tool_data["function_name"],
                display_name=tool_data.get("display_name", tool_data["function_name"]),
                description=tool_data.get("description", ""),
                use_when=tool_data.get("use_when", ""),
                not_for=tool_data.get("not_for", ""),
                parameters=params,
                returns=tool_data.get("returns", ""),
                example_queries=tool_data.get("example_queries", []),
                keywords=set(kw.lower() for kw in tool_data.get("keywords", [])),
                module_id=data["id"],
                category=data.get("category", "other")
            ))
        
        return ToolModule(
            id=data["id"],
            name=data.get("name", data["id"]),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            category=data.get("category", "other"),
            requires_env=data.get("requires_env", []),
            tools=tools
        )
    
    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by its function name."""
        self._ensure_initialized()
        return self._tools_by_name.get(name)
    
    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a specific category."""
        self._ensure_initialized()
        return [t for t in self._tools_by_name.values() if t.category == category]
    
    # NOTE: Keyword-based matching methods (match_tools_by_keywords, route_to_tool)
    # were removed as the architecture is now fully LLM-first.
    # The discovery engine is solely responsible for loading and validating tool definitions.
    
    def list_tool_cards(self) -> List[Dict[str, Any]]:
        """
        Return structured tool metadata for routing (compatible with existing API).
        Each entry includes tool_name, category, description, capabilities, required_params.
        """
        self._ensure_initialized()
        
        cards = []
        for tool in self._tools_by_name.values():
            required_params = [p.name for p in tool.parameters if p.required]
            
            cards.append({
                "tool_name": tool.function_name,
                "display_name": tool.display_name,
                "category": tool.category,
                "description": tool.description,
                "use_when": tool.use_when,
                "not_for": tool.not_for,
                "capabilities": list(tool.keywords),
                "required_params": required_params,
                "example_queries": tool.example_queries
            })
        
        return cards
    
    def get_tool_summary_for_llm(self) -> str:
        """
        Generate a formatted summary of all tools for LLM context.
        Groups by category with use_when hints.
        """
        self._ensure_initialized()
        
        if not self._modules:
            return "No tools available."
        
        lines = ["AVAILABLE TOOLS (Use these for quick, stateless operations):\n"]
        
        # Group by category
        by_category: Dict[str, List[ToolDefinition]] = {}
        for tool in self._tools_by_name.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for category, tools in sorted(by_category.items()):
            lines.append(f"\n## {category.upper()}")
            for tool in tools:
                lines.append(f"  • {tool.display_name} ({tool.function_name})")
                lines.append(f"    USE WHEN: {tool.use_when}")
                if tool.not_for:
                    lines.append(f"    NOT FOR: {tool.not_for}")
        
        return "\n".join(lines)
    
    def validate_against_implementation(self, module_id: str) -> Dict[str, Any]:
        """
        Validate a tool module's JSON definition against its Python implementation.
        
        Returns:
            Dict with 'valid', 'errors', and 'warnings' keys.
        """
        self._ensure_initialized()
        
        module = self._modules.get(module_id)
        if not module:
            return {"valid": False, "errors": [f"Module '{module_id}' not found"], "warnings": []}
        
        errors = []
        warnings = []
        
        # Try to import the Python module
        python_module_name = f"tools.{module_id}"
        try:
            py_module = importlib.import_module(python_module_name)
        except ImportError as e:
            errors.append(f"Cannot import Python module '{python_module_name}': {e}")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check each tool
        for tool_def in module.tools:
            func_name = tool_def.function_name
            
            if not hasattr(py_module, func_name):
                errors.append(f"Function '{func_name}' not found in {python_module_name}")
                continue
            
            func = getattr(py_module, func_name)
            
            # Check if it's a LangChain tool
            if not isinstance(func, BaseTool):
                warnings.append(f"Function '{func_name}' is not a LangChain BaseTool")
            
            # Check parameters match
            if isinstance(func, BaseTool) and hasattr(func, 'args_schema'):
                try:
                    schema = func.args_schema.model_json_schema()
                    schema_props = set(schema.get("properties", {}).keys())
                    json_params = {p.name for p in tool_def.parameters}
                    
                    # Check for missing in JSON
                    for prop in schema_props:
                        if prop not in json_params:
                            warnings.append(f"Parameter '{prop}' in {func_name} not documented in JSON")
                    
                    # Check for extra in JSON
                    for param in json_params:
                        if param not in schema_props:
                            errors.append(f"Parameter '{param}' in JSON but not in {func_name} implementation")
                except Exception as e:
                    warnings.append(f"Could not validate parameters for {func_name}: {e}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _ensure_initialized(self):
        """Ensure discovery has been run."""
        if not self._initialized:
            self.discover_all()


# Singleton instance
_discovery = None


def get_tool_discovery() -> ToolDiscovery:
    """Get the singleton ToolDiscovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = ToolDiscovery()
    return _discovery


# Convenience functions that mirror existing tool_registry API
def discover_tools() -> List[ToolModule]:
    """Discover all tools from tool_entries/."""
    return get_tool_discovery().discover_all()


def get_discovered_tool_cards() -> List[Dict[str, Any]]:
    """Get tool cards from discovery engine (for routing)."""
    return get_tool_discovery().list_tool_cards()


def get_tool_summary() -> str:
    """Get formatted tool summary for LLM context."""
    return get_tool_discovery().get_tool_summary_for_llm()


__all__ = [
    'ToolDiscovery',
    'ToolModule', 
    'ToolDefinition',
    'ToolParameter',
    'get_tool_discovery',
    'discover_tools',
    'get_discovered_tool_cards',
    'get_tool_summary',
]

"""
Tool Registry - Direct function tool management for the orchestrator.

This module provides a registry of available tools that can be used directly
by the orchestrator without needing to route to separate agent services.
These tools are faster and more efficient for simple, stateless operations.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from langchain_core.tools import BaseTool

logger = logging.getLogger("AgentOrchestrator")

# Tool registry - maps capability names to tool instances
_tool_registry: Dict[str, BaseTool] = {}
_tool_categories: Dict[str, List[str]] = {}
@dataclass
class _ToolMeta:
    tool: BaseTool
    categories: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)


_tool_meta_by_name: Dict[str, _ToolMeta] = {}
_tools_initialized: bool = False  # Flag to track initialization


def _ensure_tools_initialized():
    """
    Lazy initialization - only initialize tools when first accessed.
    This is called automatically by all public functions.
    """
    global _tools_initialized
    if not _tools_initialized:
        logger.info("🔧 Lazy-loading tools on first use...")
        initialize_tools()
        _tools_initialized = True


def register_tool(tool: BaseTool, capabilities: List[str], category: str):
    """
    Register a tool with its capabilities.
    
    Args:
        tool: The LangChain tool instance
        capabilities: List of capability strings this tool provides
        category: Category of the tool (e.g., 'search', 'finance', 'wiki')
    """
    for capability in capabilities:
        _tool_registry[capability.lower()] = tool
    
    if category not in _tool_categories:
        _tool_categories[category] = []
    _tool_categories[category].extend(capabilities)

    # Track per-tool metadata for routing (unique tools, required args)
    tool_name = getattr(tool, "name", None) or tool.__class__.__name__
    meta = _tool_meta_by_name.get(tool_name)
    if meta is None:
        meta = _ToolMeta(tool=tool)
        _tool_meta_by_name[tool_name] = meta
    meta.categories.add(category)
    for capability in capabilities:
        meta.capabilities.add(capability)
    
    logger.info(f"Registered tool '{tool.name}' with capabilities: {capabilities}")


def list_tool_cards() -> List[Dict[str, Any]]:
    """Return structured tool metadata for routing.

    Each entry includes:
    - tool_name
    - category
    - description
    - capabilities
    - required_params (best-effort from args_schema)
    - use_when (from JSON definitions)
    - not_for (from JSON definitions)
    - example_queries (from JSON definitions)
    """
    _ensure_tools_initialized()

    # Try to get enriched data from discovery engine
    try:
        from orchestrator.tool_discovery import get_tool_discovery
        discovery = get_tool_discovery()
        discovery_cards = {card["tool_name"]: card for card in discovery.list_tool_cards()}
    except Exception as e:
        logger.warning(f"Could not load discovery engine: {e}")
        discovery_cards = {}

    cards: List[Dict[str, Any]] = []
    for tool_name, meta in _tool_meta_by_name.items():
        tool = meta.tool

        # Best-effort: derive required params from LangChain tool args_schema
        required_params: List[str] = []
        try:
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
                required_params = list(schema.get("required", []) or [])
        except Exception:
            required_params = []

        description = getattr(tool, "description", "") or ""

        # Get enriched data from discovery if available
        discovery_data = discovery_cards.get(tool_name, {})

        # If a tool has multiple categories (rare), emit one card per category for cleaner UX.
        categories = sorted(meta.categories) or ["other"]
        for category in categories:
            cards.append(
                {
                    "tool_name": tool_name,
                    "category": category,
                    "description": description,
                    "capabilities": sorted(meta.capabilities),
                    "required_params": required_params,
                    # Enriched from discovery engine
                    "use_when": discovery_data.get("use_when", ""),
                    "not_for": discovery_data.get("not_for", ""),
                    "example_queries": discovery_data.get("example_queries", []),
                }
            )

    return cards



def get_tool_for_capability(capability: str) -> Optional[BaseTool]:
    """
    Get the tool that handles a specific capability.
    Lazy-loads tools on first call.
    
    Args:
        capability: The capability string (task name)
        
    Returns:
        The tool instance or None if not found
    """
    _ensure_tools_initialized()
    return _tool_registry.get(capability.lower())


def get_all_tool_capabilities() -> List[str]:
    """
    Get list of all capabilities that can be handled by tools.
    Lazy-loads tools on first call.
    """
    _ensure_tools_initialized()
    return list(_tool_registry.keys())


def get_tools_by_category(category: str) -> List[str]:
    """
    Get all capabilities in a specific category.
    Lazy-loads tools on first call.
    """
    _ensure_tools_initialized()
    return _tool_categories.get(category, [])


# [REMOVED] Legacy keyword matching logic
# Superseded by LLM-First Routing (defined below)


async def execute_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name or capability with parameters.

    Resolution order:
    1) Exact match on registered tool name via _tool_meta_by_name
    2) Exact match on registered tool.name (case-insensitive)
    3) Exact/keyword match on capabilities via get_best_tool_match (legacy)
    """
    _ensure_tools_initialized()

    logger.info(f"🔧 EXECUTE_TOOL: Looking up tool='{tool_name}' with params={list(parameters.keys()) if parameters else 'none'}")
    print(f"🔧 EXECUTE_TOOL: Looking up tool='{tool_name}' with params={list(parameters.keys()) if parameters else 'none'}")

    tool = None

    # 1) Exact lookup by registered tool name
    meta = _tool_meta_by_name.get(tool_name)
    if meta:
        tool = meta.tool

    # 2) Case-insensitive match on tool.name across meta entries
    if tool is None:
        for meta_tool_name, meta_obj in _tool_meta_by_name.items():
            if meta_obj.tool and getattr(meta_obj.tool, "name", "").lower() == tool_name.lower():
                tool = meta_obj.tool
                break

    # 3) Fallback to capability-based matching (legacy behavior)
    if tool is None:
        tool = get_best_tool_match(tool_name)

    if not tool:
        available_names = list(_tool_meta_by_name.keys())
        available_caps = list(_tool_registry.keys())
        logger.error(f"🔧 EXECUTE_TOOL: No tool found for '{tool_name}'. Names: {available_names[:5]}..., Caps: {available_caps[:5]}...")
        print(f"🔧 EXECUTE_TOOL: No tool found for '{tool_name}'. Names: {available_names[:5]}..., Caps: {available_caps[:5]}...")
        return {
            "success": False,
            "error": f"No tool registered for: {tool_name}",
            "available_tools": available_names + available_caps
        }
    
    try:
        # Filter parameters to match the tool's schema (prevents validation errors)
        filtered_params = parameters or {}
        try:
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
                props = schema.get("properties", {}) or {}
                allowed_keys = set(props.keys())
                logger.info(f"🔧 EXECUTE_TOOL: Tool '{tool.name}' expects params: {list(allowed_keys)}")
                print(f"🔧 EXECUTE_TOOL: Tool '{tool.name}' expects: {list(allowed_keys)}, got: {list((parameters or {}).keys())}")
                if allowed_keys:
                    filtered_params = {k: v for k, v in (parameters or {}).items() if k in allowed_keys}
                    logger.info(f"🔧 EXECUTE_TOOL: After filtering, params: {list(filtered_params.keys())}")
        except Exception as e:
            logger.error(f"🔧 EXECUTE_TOOL: Failed to filter params: {e}")
            filtered_params = parameters or {}

        # Check if tool is async
        if hasattr(tool, 'ainvoke'):
            result = await tool.ainvoke(filtered_params)
        else:
            # Sync tool - run in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.invoke, filtered_params)
        
        logger.info(f"🔧 EXECUTE_TOOL: Success - tool '{tool.name}' returned result")
        print(f"🔧 EXECUTE_TOOL: Success - tool '{tool.name}' completed")
        return {
            "success": True,
            "result": result,
            "tool_name": tool.name
        }
    except Exception as e:
        logger.error(f"🔧 EXECUTE_TOOL: Tool execution failed for '{tool_name}': {e}", exc_info=True)
        print(f"🔧 EXECUTE_TOOL: FAILED - tool '{tool.name}' error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool.name
        }


def initialize_tools():
    """
    Initialize and register all available tools.
    Can be called multiple times safely (idempotent).
    Called automatically by _ensure_tools_initialized() on first use.
    """
    global _tool_registry, _tool_categories, _tool_meta_by_name
    
    # Skip if already initialized
    if _tool_registry:
        # Hot-reload safety: ensure metadata is present even if tools were registered
        # before this module version loaded.
        if not _tool_meta_by_name:
            for capability, tool in _tool_registry.items():
                tool_name = getattr(tool, "name", None) or tool.__class__.__name__
                meta = _tool_meta_by_name.get(tool_name)
                if meta is None:
                    meta = _ToolMeta(tool=tool)
                    _tool_meta_by_name[tool_name] = meta
                meta.capabilities.add(capability)

            for category, capabilities in _tool_categories.items():
                for cap in capabilities:
                    tool = _tool_registry.get(cap.lower())
                    if not tool:
                        continue
                    tool_name = getattr(tool, "name", None) or tool.__class__.__name__
                    meta = _tool_meta_by_name.get(tool_name)
                    if meta is None:
                        meta = _ToolMeta(tool=tool)
                        _tool_meta_by_name[tool_name] = meta
                    meta.categories.add(category)

        logger.debug("Tools already initialized, skipping...")
        return
    try:
        # Finance tools
        from tools.finance_tools import (
            get_stock_quote,
            get_stock_history,
            get_company_info,
            get_key_statistics
        )
        
        register_tool(get_stock_quote, [
            "get stock quote",
            "get stock price",
            "check stock price",
            "current stock price"
        ], "finance")
        
        register_tool(get_stock_history, [
            "get stock history",
            "stock price history",
            "historical stock data",
            "ohlc data"
        ], "finance")
        
        register_tool(get_company_info, [
            "get company info",
            "company information",
            "company details",
            "business summary"
        ], "finance")
        
        register_tool(get_key_statistics, [
            "get key statistics",
            "financial metrics",
            "pe ratio",
            "market cap"
        ], "finance")
        
        logger.info("✅ Registered finance tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import finance tools: {e}")
    
    try:
        # News tools
        from tools.news_tools import search_news, get_top_headlines
        
        register_tool(search_news, [
            "search news",
            "find news",
            "news articles",
            "get company news headlines"
        ], "news")
        
        register_tool(get_top_headlines, [
            "get top headlines",
            "top news",
            "latest headlines",
            "breaking news"
        ], "news")
        
        logger.info("✅ Registered news tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import news tools: {e}")
    
    try:
        # Wikipedia tools
        from tools.wiki_tools import (
            search_wikipedia,
            get_wikipedia_summary,
            get_wikipedia_section,
            get_wikipedia_images
        )
        
        register_tool(search_wikipedia, [
            "search wikipedia",
            "wikipedia search",
            "find wiki page"
        ], "wiki")
        
        register_tool(get_wikipedia_summary, [
            "get wikipedia summary",
            "wikipedia page",
            "wiki summary"
        ], "wiki")
        
        register_tool(get_wikipedia_section, [
            "get wikipedia section",
            "wiki section"
        ], "wiki")
        
        register_tool(get_wikipedia_images, [
            "get wikipedia images",
            "wikipedia images",
            "wiki images",
            "images from wikipedia page"
        ], "wiki")
        
        logger.info("✅ Registered wikipedia tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import wikipedia tools: {e}")
    
    try:
        # Search tools
        from tools.search_tools import web_search_and_summarize
        
        register_tool(web_search_and_summarize, [
            "web search",
            "search web",
            "internet search",
            "search and summarize",
            "research topic"
        ], "search")
        
        logger.info("✅ Registered web search tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import search tools: {e}")
    
    try:
        # Image tools
        from tools.image_tools import analyze_image
        
        register_tool(analyze_image, [
            "analyze image",
            "image analysis",
            "describe image",
            "vision"
        ], "image")
        
        logger.info("✅ Registered image tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import image tools: {e}")

    try:
        # Creation tools (LLM-First)
        from tools.creation_tools import CreateDocumentTool, CreateSpreadsheetTool
        
        register_tool(CreateDocumentTool(), [
            "create document",
            "write report", 
            "generate file"
        ], "creation")
        
        register_tool(CreateSpreadsheetTool(), [
            "create spreadsheet",
            "generate excel",
            "make csv"
        ], "creation")
        
        logger.info("✅ Registered creation tools")
        
    except ImportError as e:
        logger.warning(f"Failed to import creation tools: {e}")
    
    logger.info(f"🎉 Tool registry initialized with {len(_tool_registry)} capabilities across {len(_tool_categories)} categories")

# ============================================================================
# LLM-FIRST ROUTING (Keywords Deprecated)
# ============================================================================

def is_tool_capable(capability: str) -> bool:
    """
    Check if a capability can be handled by a direct tool.
    STRICT MODE: Only returns True for exact matches.
    Keyword matching has been removed in favor of LLM routing.
    """
    _ensure_tools_initialized()
    return capability.lower() in _tool_registry


def get_best_tool_match(capability: str) -> Optional[BaseTool]:
    """
    Get the best-matching tool for a capability.
    STRICT MODE: Only returns exact matches.
    Keyword matching has been removed in favor of LLM routing.
    """
    _ensure_tools_initialized()
    return _tool_registry.get(capability.lower())


def get_tool_descriptions() -> str:
    """
    Get a formatted description of all available tools for LLM context.
    Lazy-loads tools on first call.
    
    Returns:
        Formatted string describing all tools and their capabilities
    """
    _ensure_tools_initialized()
    if not _tool_registry:
        return "No direct tools available."
    
    # Try to get enhanced descriptions from discovery engine
    try:
        from orchestrator.tool_discovery import get_tool_discovery
        discovery = get_tool_discovery()
        return discovery.get_tool_summary_for_llm()
    except Exception as e:
        logger.debug(f"Discovery engine not available, using basic descriptions: {e}")
    
    # Fallback to basic descriptions
    descriptions = ["AVAILABLE DIRECT TOOLS (Fast, no agent needed):\n"]
    
    for category, capabilities in _tool_categories.items():
        descriptions.append(f"\n{category.upper()} TOOLS:")
        for capability in capabilities:
            tool = _tool_registry.get(capability.lower())
            if tool:
                descriptions.append(f"  • {capability}: {tool.description}")
    
    return "\n".join(descriptions)



def get_tool_registry():
    """
    Get the tool registry singleton instance for use by other modules.
    Returns a dict-like interface to the tool registry.
    """
    _ensure_tools_initialized()
    return ToolRegistryInterface()


class ToolRegistryInterface:
    """Interface to tool registry for external modules"""
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by exact name"""
        meta = _tool_meta_by_name.get(tool_name)
        if meta:
            return meta.tool
        
        # Try case-insensitive match
        for meta_obj in _tool_meta_by_name.values():
            if meta_obj.tool and getattr(meta_obj.tool, "name", "").lower() == tool_name.lower():
                return meta_obj.tool
        
        return None
    
    def get_required_params(self, tool_name: str) -> List[str]:
        """Get required parameters for a tool"""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            return []
        
        try:
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
                return list(schema.get("required", []) or [])
        except Exception:
            pass
        
        return []
    
    def match_by_keywords(self, keywords: set) -> List[Dict[str, Any]]:
        """
        Match tools by keyword overlap.
        Returns list of matches sorted by score (highest first).
        """
        matches = []
        
        for tool_name, meta in _tool_meta_by_name.items():
            tool = meta.tool
            description = getattr(tool, "description", "") or ""
            
            # Extract words from description and capabilities
            desc_words = set(description.lower().split())
            cap_words = set()
            for cap in meta.capabilities:
                cap_words.update(cap.lower().split())
            
            all_words = desc_words | cap_words
            
            # Calculate overlap
            overlap = keywords & all_words
            
            if not overlap:
                continue
            
            # Score: overlap size / keywords size (precision)
            score = len(overlap) / len(keywords) if keywords else 0
            
            matches.append({
                "tool_name": tool_name,
                "score": score,
                "matched_keywords": list(overlap)
            })
        
        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return matches


__all__ = [
    'register_tool',
    'get_tool_for_capability',
    'get_all_tool_capabilities',
    'get_tools_by_category',
    'is_tool_capable',
    'execute_tool',
    'initialize_tools',
    'get_tool_descriptions',
    'list_tool_cards',
    'get_tool_registry',
]

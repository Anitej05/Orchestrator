"""
Tool Registry - Direct function tool management for the orchestrator.

This module provides a registry of available tools that can be used directly
by the orchestrator without needing to route to separate agent services.
These tools are faster and more efficient for simple, stateless operations.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from langchain_core.tools import BaseTool

logger = logging.getLogger("AgentOrchestrator")

# Tool registry - maps capability names to tool instances
_tool_registry: Dict[str, BaseTool] = {}
_tool_categories: Dict[str, List[str]] = {}
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
    
    logger.info(f"Registered tool '{tool.name}' with capabilities: {capabilities}")


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


def is_tool_capable(capability: str) -> bool:
    """
    Check if a capability can be handled by a direct tool.
    Lazy-loads tools on first call.
    """
    _ensure_tools_initialized()
    return capability.lower() in _tool_registry


async def execute_tool(capability: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool for the given capability with parameters.
    Lazy-loads tools on first call.
    
    Args:
        capability: The capability/task name
        parameters: Dictionary of parameters for the tool
        
    Returns:
        Result dictionary with 'success', 'result', and optionally 'error'
    """
    _ensure_tools_initialized()
    tool = get_tool_for_capability(capability)
    if not tool:
        return {
            "success": False,
            "error": f"No tool registered for capability: {capability}"
        }
    
    try:
        # Check if tool is async
        if hasattr(tool, 'ainvoke'):
            result = await tool.ainvoke(parameters)
        else:
            # Sync tool - run in executor to avoid blocking
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.invoke, parameters)
        
        return {
            "success": True,
            "result": result,
            "tool_name": tool.name
        }
    except Exception as e:
        logger.error(f"Tool execution failed for {capability}: {e}", exc_info=True)
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
    global _tool_registry, _tool_categories
    
    # Skip if already initialized
    if _tool_registry:
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
            get_wikipedia_section
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
    
    logger.info(f"🎉 Tool registry initialized with {len(_tool_registry)} capabilities across {len(_tool_categories)} categories")


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
    
    descriptions = ["AVAILABLE DIRECT TOOLS (Fast, no agent needed):\n"]
    
    for category, capabilities in _tool_categories.items():
        descriptions.append(f"\n{category.upper()} TOOLS:")
        for capability in capabilities:
            tool = _tool_registry.get(capability.lower())
            if tool:
                descriptions.append(f"  • {capability}: {tool.description}")
    
    return "\n".join(descriptions)


__all__ = [
    'register_tool',
    'get_tool_for_capability',
    'get_all_tool_capabilities',
    'get_tools_by_category',
    'is_tool_capable',
    'execute_tool',
    'initialize_tools',
    'get_tool_descriptions',
]

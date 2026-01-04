"""
Tools module - Direct function tools for the orchestrator.
These replace lightweight HTTP agents with direct function calls.
"""

from .news_tools import search_news, get_top_headlines
from .wiki_tools import search_wikipedia, get_wikipedia_summary, get_wikipedia_section, get_wikipedia_images
from .finance_tools import get_stock_quote, get_stock_history, get_company_info, get_key_statistics
from .search_tools import web_search_and_summarize
from .image_tools import analyze_image

__all__ = [
    # News tools
    "search_news",
    "get_top_headlines",
    # Wikipedia tools
    "search_wikipedia",
    "get_wikipedia_summary", 
    "get_wikipedia_section",
    "get_wikipedia_images",
    # Finance tools
    "get_stock_quote",
    "get_stock_history",
    "get_company_info",
    "get_key_statistics",
    # Search tools
    "web_search_and_summarize",
    # Image tools
    "analyze_image",
]

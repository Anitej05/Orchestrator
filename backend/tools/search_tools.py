"""
Web search tools using Groq's compound model.
Converted from groq_search_agent.py to direct function tools.
"""

import os
import json
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client for compound model
_groq_llm = None

def _get_groq_llm():
    """Lazy initialization of Groq LLM."""
    global _groq_llm
    if _groq_llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not configured")
        _groq_llm = ChatGroq(model="groq/compound")
    return _groq_llm


@tool
async def web_search_and_summarize(
    query: str,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> Dict:
    """
    Perform real-time web search and return summarized answer with citations.
    Uses Groq's compound-beta model for search-augmented responses.
    
    Args:
        query: The question or topic to search for
        include_domains: Optional list of domains to restrict search to
        exclude_domains: Optional list of domains to exclude from search
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        llm = _get_groq_llm()
        
        # Build model kwargs
        model_kwargs = {"model": "compound-beta"}
        
        # Add search settings if provided
        search_settings = {}
        if include_domains:
            search_settings["include_domains"] = include_domains
        if exclude_domains:
            search_settings["exclude_domains"] = exclude_domains
        
        if search_settings:
            model_kwargs["search_settings"] = search_settings
        
        # Bind kwargs and invoke
        bound_llm = llm.bind(**model_kwargs)
        response = await bound_llm.ainvoke([HumanMessage(content=query)])
        
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Extract sources from tool calls
        sources = []
        if response.additional_kwargs and "tool_calls" in response.additional_kwargs:
            for tool_call in response.additional_kwargs["tool_calls"]:
                if tool_call.get("function", {}).get("name") == "search":
                    try:
                        output_str = tool_call.get("function", {}).get("output", "[]")
                        output = json.loads(output_str)
                        sources.extend(output)
                    except json.JSONDecodeError:
                        pass
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

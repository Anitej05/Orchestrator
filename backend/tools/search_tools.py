"""
Web search tools using Groq LLM.
"""

import os
import json
from typing import Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def web_search_and_summarize(
    query: str,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> Dict:
    """
    Perform real-time web search and return summarized answer with citations.
    Uses Groq's LLM for generating responses based on the query.
    
    Note: Real-time web search is currently unavailable. This tool provides
    LLM-generated responses based on the model's training data.
    
    Args:
        query: The question or topic to search for
        include_domains: Optional list of domains to restrict search to (not used)
        exclude_domains: Optional list of domains to exclude from search (not used)
        
    Returns:
        Dictionary with answer and sources
    """
    try:
        from langchain_groq import ChatGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"error": "GROQ_API_KEY not configured", "answer": None, "sources": []}
        
        # Use Groq's compound-mini model for web search
        llm = ChatGroq(model="groq/compound-mini", api_key=api_key)
        
        # Generate response based on query
        response = llm.invoke([HumanMessage(content=f"""You are a helpful search assistant. Answer the following query based on your knowledge.

Query: {query}

Provide a helpful, accurate answer. If you're not certain about current events or real-time data, acknowledge that your information may not be up to date.""")])
        
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        
        return {
            "answer": answer,
            "sources": [],  # No real-time sources available
            "query": query,
            "note": "Response based on LLM knowledge. Real-time web search not available."
        }

    except Exception as e:
        # Return error dict instead of raising - prevents infinite retries
        return {
            "error": f"Search failed: {str(e)}",
            "answer": None,
            "sources": [],
            "query": query
        }



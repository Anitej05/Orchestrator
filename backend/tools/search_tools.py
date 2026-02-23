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
        from dotenv import load_dotenv
        
        # Ensure .env is loaded (override=True so .env wins over stale system vars)
        load_dotenv(override=True)
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {"error": "GROQ_API_KEY not configured", "answer": None, "sources": []}
        
        # Use Groq's compound-mini model for web search
        # max_tokens prevents 413 errors from oversized responses
        llm = ChatGroq(
            model="groq/compound-mini",
            api_key=api_key,
            max_tokens=2000,
        )
        
        # Generate response based on query
        response = llm.invoke([HumanMessage(content=f"""You are a helpful search assistant. Answer the following query concisely.

Query: {query}

Provide a helpful, accurate, CONCISE answer (under 500 words). Focus on key facts and data points. If you're not certain about current events, acknowledge that.""")])
        
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Truncate very long responses to prevent downstream prompt bloat
        if len(answer) > 3000:
            answer = answer[:3000] + "\n\n[Response truncated for brevity]"
        
        return {
            "answer": answer,
            "sources": [],  # No real-time sources available
            "query": query,
            "note": "Response based on LLM knowledge. Real-time web search not available."
        }

    except Exception as e:
        error_str = str(e)
        # On 413 (too large), return a simpler response
        if "413" in error_str:
            try:
                llm_fallback = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=api_key,
                    max_tokens=1000,
                )
                response = llm_fallback.invoke([HumanMessage(content=f"Briefly answer: {query}")])
                return {
                    "answer": response.content[:2000],
                    "sources": [],
                    "query": query,
                    "note": "Fallback response due to size limits."
                }
            except Exception:
                pass
        
        # Return error dict instead of raising - prevents infinite retries
        return {
            "error": f"Search failed: {error_str}",
            "answer": None,
            "sources": [],
            "query": query
        }



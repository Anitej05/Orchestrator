"""
Wikipedia tools using wikipedia library.
Converted from wiki_agent.py to direct function tools.
"""

import wikipedia
from typing import List, Dict
from langchain_core.tools import tool


@tool
def search_wikipedia(query: str) -> Dict:
    """
    Search Wikipedia for pages matching a query.
    
    Args:
        query: The search term to look for (e.g., 'Artificial Intelligence')
        
    Returns:
        Dictionary with list of matching page titles
    """
    try:
        results = wikipedia.search(query)
        if not results:
            return {"error": f"No search results found for '{query}'"}
        return {"results": results}
    except Exception as e:
        return {"error": f"Wikipedia search failed: {str(e)}"}


@tool
def get_wikipedia_summary(title: str) -> Dict:
    """
    Get the full summary of a Wikipedia page.
    
    Args:
        title: The exact title of the Wikipedia page
        
    Returns:
        Dictionary with title, summary, and url
    """
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }
    except wikipedia.exceptions.PageError:
        return {"error": f"Page '{title}' does not exist"}
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "error": f"Title '{title}' is ambiguous",
            "options": e.options[:10]  # Limit to first 10 options
        }
    except Exception as e:
        return {"error": f"Failed to get summary: {str(e)}"}


@tool
def get_wikipedia_section(title: str, section: str) -> Dict:
    """
    Get content of a specific section from a Wikipedia page.
    
    Args:
        title: The exact title of the Wikipedia page
        section: The section title to retrieve (e.g., 'History', 'Applications')
        
    Returns:
        Dictionary with title, section name, and content
    """
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        section_content = page.section(section)
        
        if not section_content:
            return {"error": f"Section '{section}' not found on page '{title}'"}
        
        return {
            "title": page.title,
            "section": section,
            "content": section_content
        }
    except wikipedia.exceptions.PageError:
        return {"error": f"Page '{title}' does not exist"}
    except Exception as e:
        return {"error": f"Failed to get section: {str(e)}"}

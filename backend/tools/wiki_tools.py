"""
Wikipedia tools using wikipedia-api library (Python 3.13 compatible).
Converted from wiki_agent.py to direct function tools.
"""

import wikipediaapi
from typing import List, Dict
from langchain_core.tools import tool

# Initialize Wikipedia API client
wiki_wiki = wikipediaapi.Wikipedia('Orbimesh/1.0', 'en')


@tool
def search_wikipedia(query: str = "", search_query: str = "") -> Dict:
    """
    Search Wikipedia for pages matching a query.

    Args:
        query: The search term to look for (e.g., 'Artificial Intelligence')
        search_query: Alias for query

    Returns:
        Dictionary with list of matching page titles
    """
    try:
        # Accept both query and search_query parameters
        search_term = query or search_query
        if not search_term:
            return {"error": "No search query provided"}

        # wikipedia-api doesn't have built-in search, so we'll try to get the page
        page = wiki_wiki.page(search_term)
        if page.exists():
            return {"results": [page.title], "summary": page.summary[:200]}
        else:
            return {"error": f"No page found for '{search_term}'. Try a more specific query."}
    except Exception as e:
        return {"error": f"Wikipedia search failed: {str(e)}"}


@tool
def get_wikipedia_summary(title: str = "", page_title: str = "") -> Dict:
    """
    Get the full summary of a Wikipedia page.

    Args:
        title: The exact title of the Wikipedia page
        page_title: Alias for title

    Returns:
        Dictionary with title, summary, and url
    """
    try:
        # Accept both title and page_title parameters
        query = title or page_title
        if not query:
            return {"error": "No page title provided"}

        page = wiki_wiki.page(query)
        if not page.exists():
            return {"error": f"Page '{query}' does not exist"}
        
        return {"title": page.title, "summary": page.summary, "url": page.fullurl}
    except Exception as e:
        return {"error": f"Failed to get summary: {str(e)}"}


@tool
def get_wikipedia_section(
    title: str = "", page_title: str = "", section: str = "", section_name: str = "", section_title: str = ""
) -> Dict:
    """
    Get content of a specific section from a Wikipedia page.

    Args:
        title: The exact title of the Wikipedia page
        page_title: Alias for title
        section: The section title to retrieve (e.g., 'History', 'Applications')
        section_name: Alias for section
        section_title: Alias for section

    Returns:
        Dictionary with title, section name, and content
    """
    try:
        # Accept both title and page_title parameters
        query_title = title or page_title
        query_section = section or section_name or section_title

        if not query_title:
            return {"error": "No page title provided"}
        if not query_section:
            return {"error": "No section name provided"}

        page = wiki_wiki.page(query_title)
        if not page.exists():
            return {"error": f"Page '{query_title}' does not exist"}

        # Find the section
        section_obj = None
        for sec in page.sections:
            if sec.title.lower() == query_section.lower():
                section_obj = sec
                break

        if not section_obj:
            available_sections = [s.title for s in page.sections[:10]]
            return {
                "error": f"Section '{query_section}' not found on page '{query_title}'",
                "available_sections": available_sections
            }

        return {
            "title": page.title,
            "section": section_obj.title,
            "content": section_obj.text,
        }
    except Exception as e:
        return {"error": f"Failed to get section: {str(e)}"}


@tool
def get_wikipedia_images(title: str = "", page_title: str = "") -> Dict:
    """
    Get Wikipedia page information (images not directly available in wikipedia-api).

    Args:
        title: The exact title of the Wikipedia page
        page_title: Alias for title

    Returns:
        Dictionary with title and page information
    """
    try:
        # Accept both title and page_title parameters
        query = title or page_title
        if not query:
            return {"error": "No page title provided"}

        page = wiki_wiki.page(query)
        if not page.exists():
            return {"error": f"Page '{query}' does not exist"}

        # wikipedia-api doesn't provide direct image access
        # Return page info and note about image limitations
        return {
            "title": page.title,
            "url": page.fullurl,
            "note": "Image extraction not available in this API version. Visit the URL for images."
        }
    except Exception as e:
        return {"error": f"Failed to get page info: {str(e)}"}

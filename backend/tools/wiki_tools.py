"""
Wikipedia tools using wikipedia library.
Converted from wiki_agent.py to direct function tools.
"""

import wikipedia
from typing import List, Dict
from langchain_core.tools import tool


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

        results = wikipedia.search(search_term)
        if not results:
            return {"error": f"No search results found for '{search_term}'"}
        return {"results": results}
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

        page = wikipedia.page(query, auto_suggest=False, redirect=True)
        return {"title": page.title, "summary": page.summary, "url": page.url}
    except wikipedia.exceptions.PageError:
        return {"error": f"Page '{query}' does not exist"}
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "error": f"Title '{query}' is ambiguous",
            "options": e.options[:10],  # Limit to first 10 options
        }
    except Exception as e:
        return {"error": f"Failed to get summary: {str(e)}"}


@tool
def get_wikipedia_section(
    title: str = "", page_title: str = "", section: str = "", section_name: str = ""
) -> Dict:
    """
    Get content of a specific section from a Wikipedia page.

    Args:
        title: The exact title of the Wikipedia page
        page_title: Alias for title
        section: The section title to retrieve (e.g., 'History', 'Applications')
        section_name: Alias for section

    Returns:
        Dictionary with title, section name, and content
    """
    try:
        # Accept both title and page_title parameters
        query_title = title or page_title
        query_section = section or section_name

        if not query_title:
            return {"error": "No page title provided"}
        if not query_section:
            return {"error": "No section name provided"}

        page = wikipedia.page(query_title, auto_suggest=False, redirect=True)
        section_content = page.section(query_section)

        if not section_content:
            return {
                "error": f"Section '{query_section}' not found on page '{query_title}'"
            }

        return {
            "title": page.title,
            "section": query_section,
            "content": section_content,
        }
    except wikipedia.exceptions.PageError:
        return {"error": f"Page '{query_title}' does not exist"}
    except Exception as e:
        return {"error": f"Failed to get section: {str(e)}"}


@tool
def get_wikipedia_images(title: str = "", page_title: str = "") -> Dict:
    """
    Get all image URLs from a Wikipedia page.

    Args:
        title: The exact title of the Wikipedia page
        page_title: Alias for title

    Returns:
        Dictionary with title and list of image URLs
    """
    try:
        # Accept both title and page_title parameters
        query = title or page_title
        if not query:
            return {"error": "No page title provided"}

        page = wikipedia.page(query, auto_suggest=False, redirect=True)

        if not page.images:
            return {"error": f"No images found on page '{query}'"}

        return {"title": page.title, "images": page.images}
    except wikipedia.exceptions.PageError:
        return {"error": f"Page '{query}' does not exist"}
    except wikipedia.exceptions.DisambiguationError as e:
        return {
            "error": f"Title '{query}' is ambiguous",
            "options": e.options[:10],  # Limit to first 10 options
        }
    except Exception as e:
        return {"error": f"Failed to get images: {str(e)}"}

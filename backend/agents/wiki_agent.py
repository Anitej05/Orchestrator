from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import wikipedia
import uvicorn

AGENT_DEFINITION = {
  "id": "wikipedia_agent",
  "owner_id": "orbimesh-vendor",
  "name": "Orbimesh Wikipedia Agent",
  "description": "Provides comprehensive access to Wikipedia content. It can search for pages, retrieve full-page summaries, fetch specific sections of a page, and get links to images on a page.",
  "capabilities": [
    "search for a wikipedia page",
    "get a wikipedia page summary",
    "get a specific section from a wikipedia page",
    "get image URLs from a wikipedia page"
  ],
  "price_per_call_usd": 0.0015,
  "status": "active",
  "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA3FcU8hPhmFLgez6qPf801aQahasAlG5S4MPb16nWJPA=\n-----END PUBLIC KEY-----",
  "endpoints": [
    {
      "endpoint": "http://localhost:8030/search",
      "http_method": "GET",
      "description": "Searches Wikipedia for a given query and returns a list of matching page titles.",
      "parameters": [
        {
          "name": "query",
          "param_type": "string",
          "required": True,
          "description": "The search term to look for on Wikipedia (e.g., 'Artificial Intelligence')."
        }
      ]
    },
    {
      "endpoint": "http://localhost:8030/summary",
      "http_method": "GET",
      "description": "Retrieves a detailed summary of a specific Wikipedia page by its exact title.",
      "parameters": [
        {
          "name": "title",
          "param_type": "string",
          "required": True,
          "description": "The exact title of the Wikipedia page. Use the search endpoint first to find the correct title."
        }
      ]
    },
    {
      "endpoint": "http://localhost:8030/section",
      "http_method": "GET",
      "description": "Fetches the content of a specific section from a Wikipedia page.",
      "parameters": [
        {
          "name": "title",
          "param_type": "string",
          "required": True,
          "description": "The exact title of the Wikipedia page."
        },
        {
          "name": "section",
          "param_type": "string",
          "required": True,
          "description": "The title of the section to retrieve (e.g., 'History', 'Applications')."
        }
      ]
    },
    {
      "endpoint": "http://localhost:8030/images",
      "http_method": "GET",
      "description": "Retrieves a list of image URLs from a specified Wikipedia page.",
      "parameters": [
        {
          "name": "title",
          "param_type": "string",
          "required": True,
          "description": "The exact title of the Wikipedia page."
        }
      ]
    }
  ]
}

app = FastAPI(
    title="Enhanced Wikipedia Agent",
    description="An API to search for and retrieve structured content from Wikipedia.",
    version="3.0.0"
)

def get_page_or_raise(title: str) -> wikipedia.WikipediaPage:
    """Helper to retrieve a Wikipedia page and handle common errors."""
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        return page
    except wikipedia.exceptions.PageError:
        raise HTTPException(status_code=404, detail=f"Wikipedia page with title '{title}' does not exist.")
    except wikipedia.exceptions.DisambiguationError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"The title '{title}' is ambiguous. Please choose one of the following options:",
                "options": e.options
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return AGENT_DEFINITION

# --- Pydantic Models ---
class PageSummaryResponse(BaseModel):
    title: str
    summary: str
    url: str

class SearchResult(BaseModel):
    results: List[str]

class SectionResponse(BaseModel):
    title: str
    section: str
    content: str

class ImageResponse(BaseModel):
    title: str
    images: List[str]

# --- API Endpoints ---
@app.get("/search", response_model=SearchResult)
def search_wikipedia(query: str):
    """Searches Wikipedia and returns a list of matching page titles."""
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            raise HTTPException(status_code=404, detail=f"No search results found for '{query}'")
        return {"results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary", response_model=PageSummaryResponse)
def get_wikipedia_summary(title: str):
    """Retrieves the full summary of a Wikipedia page."""
    page = get_page_or_raise(title)
    return {
        "title": page.title,
        "summary": page.summary,
        "url": page.url
    }

@app.get("/section", response_model=SectionResponse)
def get_wikipedia_section(title: str, section: str):
    """Fetches the content of a specific section from a page."""
    page = get_page_or_raise(title)
    try:
        section_content = page.section(section)
        if not section_content:
             raise HTTPException(status_code=404, detail=f"Section '{section}' not found on page '{title}'.")
        return {
            "title": page.title,
            "section": section,
            "content": section_content
        }
    except Exception:
         raise HTTPException(status_code=404, detail=f"Could not retrieve section '{section}'. It may not exist or is formatted unexpectedly.")

@app.get("/images", response_model=ImageResponse)
def get_wikipedia_images(title: str):
    """Retrieves all image URLs from a Wikipedia page."""
    page = get_page_or_raise(title)
    if not page.images:
        raise HTTPException(status_code=404, detail=f"No images found on page '{title}'.")
    return {
        "title": page.title,
        "images": page.images
    }

if __name__ == "__main__":
    uvicorn.run("wiki_agent:app", host="0.0.0.0", port=8030, reload=False)
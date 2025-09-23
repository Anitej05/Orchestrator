from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file
load_dotenv()

# --- Configuration & API Key Check ---
NEWS_API_KEY = os.getenv("NEWS_AGENT_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2"

if not NEWS_API_KEY:
    raise RuntimeError("NEWS_AGENT_API_KEY is not set in the environment. The agent cannot start.")

AGENT_DEFINITION = {
    "id": "news_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Orbimesh News Agent",
    "description": "Fetches news articles from a wide range of sources via the NewsAPI. It can search for articles on any topic or retrieve top business headlines from specific countries.",
    "capabilities": [
        "search for news articles on a specific topic",
        "get top business headlines from a country"
    ],
    "price_per_call_usd": 0.002,
    "status": "active",
    "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA3FcU8hPhmFLgez6qPf801aQahasAlG5S4MPb16nWJPA=\n-----END PUBLIC KEY-----",
    "endpoints": [
        {
            "endpoint": "http://localhost:8020/everything",
            "http_method": "GET",
            "description": "Searches for news articles matching a keyword or query. Ideal for finding news about a specific company, person, or event.",
            "parameters": [
                {
                    "name": "query",
                    "param_type": "string",
                    "required": True,
                    "description": "The search keyword or phrase (e.g., 'Tesla', 'stock market trends')."
                },
                {
                    "name": "language",
                    "param_type": "string",
                    "required": False,
                    "default_value": "en",
                    "description": "The 2-letter ISO-639-1 code of the language you want to get articles in (e.g., 'en', 'es', 'fr')."
                },
                {
                    "name": "page_size",
                    "param_type": "integer",
                    "required": False,
                    "default_value": "10",
                    "description": "The number of results to return per page (maximum 100)."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8020/top-headlines",
            "http_method": "GET",
            "description": "Provides live top business headlines for a specific country.",
            "parameters": [
                {
                    "name": "country",
                    "param_type": "string",
                    "required": True,
                    "description": "The 2-letter ISO 3166-1 code of the country you want to get headlines for (e.g., 'us', 'gb', 'in')."
                },
                {
                    "name": "page_size",
                    "param_type": "integer",
                    "required": False,
                    "default_value": "10",
                    "description": "The number of results to return per page (maximum 100)."
                }
            ]
        }
    ]
}

app = FastAPI(title="Enhanced News Agent")

# --- Pydantic Models for Response Structure ---
class ArticleSource(BaseModel):
    id: Optional[str] = None
    name: str

class Article(BaseModel):
    source: ArticleSource
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: str
    publishedAt: datetime

class NewsResponse(BaseModel):
    status: str
    totalResults: int
    articles: List[Article]

# --- Shared API Fetching Logic ---
def fetch_from_news_api(endpoint: str, params: dict):
    headers = {"X-Api-Key": NEWS_API_KEY}
    try:
        response = requests.get(f"{NEWS_API_BASE_URL}/{endpoint}", params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Extract error message from NewsAPI response if available
        error_detail = e.response.json().get("message", str(e))
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from News API: {e}")

@app.get("/")
def read_root():
    return AGENT_DEFINITION

# --- API Endpoints ---
@app.get("/everything", response_model=NewsResponse)
def get_everything(
    query: str = Query(..., description="The search keyword or phrase."),
    language: str = Query("en", description="The language of the articles."),
    page_size: int = Query(10, ge=1, le=100, description="Number of results to return.")
):
    """Search for news articles matching a query."""
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    }
    data = fetch_from_news_api("everything", params)
    if not data.get("articles"):
        raise HTTPException(status_code=404, detail=f"No news found for query '{query}'")
    return data

@app.get("/top-headlines", response_model=NewsResponse)
def get_top_headlines(
    country: str = Query(..., min_length=2, max_length=2, description="The 2-letter country code."),
    page_size: int = Query(10, ge=1, le=100, description="Number of results to return.")
):
    """Get top business headlines for a country."""
    params = {
        "country": country,
        "category": "business",
        "pageSize": page_size
    }
    data = fetch_from_news_api("top-headlines", params)
    if not data.get("articles"):
        raise HTTPException(status_code=404, detail=f"No top business headlines found for country '{country}'")
    return data

if __name__ == "__main__":
    uvicorn.run("news_agent:app", host="127.0.0.1", port=8020, reload=True)
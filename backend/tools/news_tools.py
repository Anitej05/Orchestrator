"""
News tools using NewsAPI.
Converted from news_agent.py to direct function tools.
"""

import os
import requests
from typing import List, Dict, Optional
from langchain_core.tools import tool

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2"


@tool
def search_news(
    query: str,
    language: str = "en",
    page_size: int = 10
) -> Dict:
    """
    Search for news articles matching a keyword or query.
    
    Args:
        query: The search keyword or phrase (e.g., 'Tesla', 'stock market trends')
        language: The 2-letter ISO-639-1 code of the language (default: 'en')
        page_size: Number of results to return (1-100, default: 10)
        
    Returns:
        Dictionary with status, totalResults, and articles list
    """
    if not NEWS_API_KEY:
        return {"error": "NEWS_API_KEY not configured"}
    
    params = {
        "q": query,
        "language": language,
        "pageSize": min(page_size, 100),
        "sortBy": "publishedAt"
    }
    headers = {"X-Api-Key": NEWS_API_KEY}
    
    try:
        response = requests.get(
            f"{NEWS_API_BASE_URL}/everything",
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Check for API-level errors
        if data.get("status") == "error":
            return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
        
        if not data.get("articles"):
            return {
                "status": "ok",
                "totalResults": 0,
                "articles": [],
                "message": f"No news articles found for query '{query}'. Try different keywords or check back later."
            }
        
        return {
            "status": data.get("status"),
            "totalResults": data.get("totalResults", 0),
            "articles": [
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name")
                }
                for article in data.get("articles", [])[:page_size]
            ]
        }
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors specifically
        if e.response and e.response.status_code == 401:
            return {"error": "NewsAPI authentication failed. Check API key."}
        elif e.response and e.response.status_code == 429:
            return {"error": "NewsAPI rate limit exceeded. Try again later."}
        else:
            return {"error": f"NewsAPI request failed: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to NewsAPI: {str(e)}"}


@tool
def get_top_headlines(
    country: str = "us",
    page_size: int = 10
) -> Dict:
    """
    Get top business headlines for a specific country.
    
    Args:
        country: The 2-letter ISO 3166-1 country code (e.g., 'us', 'gb', 'in')
        page_size: Number of results to return (1-100, default: 10)
        
    Returns:
        Dictionary with status, totalResults, and articles list
    """
    if not NEWS_API_KEY:
        return {"error": "NEWS_API_KEY not configured"}
    
    params = {
        "country": country.lower(),
        "category": "business",
        "pageSize": min(page_size, 100)
    }
    headers = {"X-Api-Key": NEWS_API_KEY}
    
    try:
        response = requests.get(
            f"{NEWS_API_BASE_URL}/top-headlines",
            params=params,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Check for API-level errors
        if data.get("status") == "error":
            return {"error": f"NewsAPI error: {data.get('message', 'Unknown error')}"}
        
        if not data.get("articles"):
            return {
                "status": "ok",
                "totalResults": 0,
                "articles": [],
                "message": f"No headlines found for country '{country}'. Try a different country code."
            }
        
        return {
            "status": data.get("status"),
            "totalResults": data.get("totalResults", 0),
            "articles": [
                {
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "url": article.get("url"),
                    "publishedAt": article.get("publishedAt"),
                    "source": article.get("source", {}).get("name")
                }
                for article in data.get("articles", [])[:page_size]
            ]
        }
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors specifically
        if e.response and e.response.status_code == 401:
            return {"error": "NewsAPI authentication failed. Check API key."}
        elif e.response and e.response.status_code == 429:
            return {"error": "NewsAPI rate limit exceeded. Try again later."}
        else:
            return {"error": f"NewsAPI request failed: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to NewsAPI: {str(e)}"}

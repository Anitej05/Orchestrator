"""
News tools using NewsAPI.
Converted from news_agent.py to direct function tools.
"""

import os
import requests
import logging
from typing import List, Dict, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Log API key status at import time for debugging
if not NEWS_API_KEY:
    logger.warning("⚠️ NEWS_API_KEY environment variable not set! News tools will not work.")
else:
    logger.info("✅ NEWS_API_KEY loaded successfully")


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
        error_msg = "NEWS_API_KEY not configured - Set NEWS_API_KEY environment variable"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "status": "error"}
    
    logger.info(f"🔍 [SEARCH_NEWS] Searching for: '{query}' (language={language}, page_size={page_size})")
    
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
        
        logger.info(f"📡 [SEARCH_NEWS] API Response: status={data.get('status')}, totalResults={data.get('totalResults')}")
        
        # Check for API-level errors
        if data.get("status") == "error":
            error_msg = f"NewsAPI error: {data.get('message', 'Unknown error')}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "status": "error"}
        
        if not data.get("articles"):
            message = f"No news articles found for query '{query}'. Try different keywords or check back later."
            logger.warning(f"⚠️ [SEARCH_NEWS] {message}")
            return {
                "status": "ok",
                "totalResults": 0,
                "articles": [],
                "message": message
            }
        
        article_count = len(data.get("articles", []))
        logger.info(f"✅ [SEARCH_NEWS] Found {article_count} articles for '{query}'")
        
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
            error_msg = "NewsAPI authentication failed. Check API key. STATUS: 401 Unauthorized"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "status": "error"}
        elif e.response and e.response.status_code == 429:
            error_msg = "NewsAPI rate limit exceeded. Try again later. STATUS: 429"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "status": "error"}
        else:
            error_msg = f"NewsAPI request failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "status": "error"}
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to connect to NewsAPI: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "status": "error"}


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

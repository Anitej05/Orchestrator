"""
Finance tools using yfinance library.
Converted from finance_agent.py to direct function tools.
"""

import yfinance as yf
from typing import Dict, Optional
from langchain_core.tools import tool


def _safe_get(data: Dict, key: str):
    """Safely extract value from dict, handling numpy/pandas types."""
    value = data.get(key)
    if value is None:
        return None
    # Convert numpy/pandas types to native Python
    if hasattr(value, "item"):
        try:
            return value.item()
        except:
            pass
    return value


@tool
def get_stock_quote(ticker: str) -> Dict:
    """
    Get current stock price and basic quote data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        
    Returns:
        Dictionary with current price, day high/low, volume, market cap
    """
    try:
        stock = yf.Ticker(ticker.upper())
        
        # Try fast_info first
        try:
            fast = stock.fast_info
            if fast and isinstance(fast, dict):
                return {
                    "ticker": ticker.upper(),
                    "price": _safe_get(fast, "lastPrice") or _safe_get(fast, "regularMarketPrice"),
                    "dayHigh": _safe_get(fast, "dayHigh"),
                    "dayLow": _safe_get(fast, "dayLow"),
                    "volume": _safe_get(fast, "lastVolume"),
                    "marketCap": _safe_get(fast, "marketCap"),
                    "source": "fast_info"
                }
        except:
            pass
        
        # Fallback to recent history
        hist = stock.history(period="1d")
        if hist is not None and not hist.empty:
            latest = hist.iloc[-1]
            return {
                "ticker": ticker.upper(),
                "price": float(latest['Close']),
                "dayHigh": float(latest['High']),
                "dayLow": float(latest['Low']),
                "volume": int(latest['Volume']),
                "marketCap": None,
                "source": "history"
            }
        
        return {"error": f"No data available for ticker '{ticker}'"}
        
    except Exception as e:
        return {"error": f"Failed to get quote: {str(e)}"}


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> Dict:
    """
    Get historical stock price data (OHLCV).
    
    Args:
        ticker: Stock ticker symbol
        period: Time period - '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'
        
    Returns:
        Dictionary with ticker and list of historical data points
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        if hist is None or hist.empty:
            return {"error": f"No historical data for ticker '{ticker}'"}
        
        # Convert to list of dicts
        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "data": data
        }
        
    except Exception as e:
        return {"error": f"Failed to get history: {str(e)}"}


@tool
def get_company_info(ticker: str) -> Dict:
    """
    Get company information and business summary.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with company name, sector, industry, website, summary
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info or {}
        
        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "summary": info.get("longBusinessSummary"),
            "employees": info.get("fullTimeEmployees")
        }
        
    except Exception as e:
        return {"error": f"Failed to get company info: {str(e)}"}


@tool
def get_key_statistics(ticker: str) -> Dict:
    """
    Get key financial statistics and metrics.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with PE ratio, EPS, market cap, dividend yield, etc.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info or {}
        
        return {
            "ticker": ticker.upper(),
            "marketCap": info.get("marketCap"),
            "beta": info.get("beta"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "trailingEps": info.get("trailingEps"),
            "forwardEps": info.get("forwardEps"),
            "dividendYield": info.get("dividendYield"),
            "payoutRatio": info.get("payoutRatio"),
            "priceToBook": info.get("priceToBook"),
            "52WeekHigh": info.get("fiftyTwoWeekHigh"),
            "52WeekLow": info.get("fiftyTwoWeekLow")
        }
        
    except Exception as e:
        return {"error": f"Failed to get statistics: {str(e)}"}

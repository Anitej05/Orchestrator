# finance_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import yfinance as yf
import pandas as pd
import time
import threading

# -------------------------
# Agent metadata (returned on GET /)
# -------------------------
AGENT_DEFINITION = {
    "id": "finance_agent",
    "owner_id": "orbimesh-vendor",
    "name": "Orbimesh Finance Agent",
    "description": "Provides comprehensive, real-time, and historical financial data for stocks using the Yahoo Finance API. It can fetch current stock quotes, historical market data, company information and key statistics, and analyst recommendations.",
    "capabilities": [
        "get current stock quote",
        "get historical stock prices",
        "get company information and summary",
        "get key financial statistics",
        "get analyst recommendations"
    ],
    "price_per_call_usd": 0.002,
    "status": "active",
    "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAmALfvbLoYmXlkUEpLI26x6WvHuhTgTTbWcYQOqfOPw4=\n-----END PUBLIC KEY-----",
    "endpoints": [
        {
            "endpoint": "http://localhost:8010/quote",
            "http_method": "GET",
            "description": "Fetches the latest price data for a given stock ticker, including current price, day's high, low, and trading volume.",
            "parameters": [
                {
                    "name": "ticker",
                    "param_type": "string",
                    "required": True,
                    "description": "The stock ticker symbol (e.g., 'AAPL', 'GOOGL')."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8010/history",
            "http_method": "POST",
            "description": "Returns historical Open, High, Low, Close, and Volume (OHLCV) prices for a ticker over a specified period.",
            "parameters": [
                {
                    "name": "ticker",
                    "param_type": "string",
                    "required": True,
                    "description": "The stock ticker symbol."
                },
                {
                    "name": "period",
                    "param_type": "string",
                    "required": False,
                    "default_value": "1y",
                    "description": "The time period for the data (e.g., '1mo', '1y', 'max')."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8010/company_info",
            "http_method": "GET",
            "description": "Provides key information about a company, including its sector, industry, and a long business summary.",
            "parameters": [
                {
                    "name": "ticker",
                    "param_type": "string",
                    "required": True,
                    "description": "The stock ticker symbol."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8010/key_statistics",
            "http_method": "GET",
            "description": "Returns a dictionary of key financial metrics, such as market cap, P/E ratio, EPS, and dividend yield.",
            "parameters": [
                {
                    "name": "ticker",
                    "param_type": "string",
                    "required": True,
                    "description": "The stock ticker symbol."
                }
            ]
        },
        {
            "endpoint": "http://localhost:8010/recommendations",
            "http_method": "GET",
            "description": "Fetches the latest analyst recommendations and ratings for a given stock ticker.",
            "parameters": [
                {
                    "name": "ticker",
                    "param_type": "string",
                    "required": True,
                    "description": "The stock ticker symbol."
                }
            ]
        }
    ]
}

# -------------------------
# App + Models
# -------------------------
app = FastAPI(title="Enhanced Finance Agent")

class HistoryRequest(BaseModel):
    ticker: str
    period: str = "1y"

class QuoteResponse(BaseModel):
    regularMarketPrice: float
    dayHigh: Optional[float] = None
    dayLow: Optional[float] = None
    regularMarketVolume: Optional[int] = None
    marketCap: Optional[int] = None
    source: str

class CompanyInfoResponse(BaseModel):
    longName: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    longBusinessSummary: Optional[str] = None
    website: Optional[str] = None

class KeyStatisticsResponse(BaseModel):
    marketCap: Optional[int] = None
    beta: Optional[float] = None
    trailingPE: Optional[float] = None
    forwardPE: Optional[float] = None
    trailingEps: Optional[float] = None
    forwardEps: Optional[float] = None
    dividendYield: Optional[float] = None
    payoutRatio: Optional[float] = None
    priceToBook: Optional[float] = None

# -------------------------
# Simple TTL cache for ticker objects / info to reduce repeated network calls
# -------------------------
_TICKER_CACHE: Dict[str, Dict[str, Any]] = {}
_TICKER_CACHE_LOCK = threading.Lock()
_TTL_SECONDS = 30  # short TTL to keep data reasonably fresh

def _now() -> float:
    return time.time()

def _cache_set(ticker: str, payload: Dict[str, Any]) -> None:
    with _TICKER_CACHE_LOCK:
        _TICKER_CACHE[ticker.upper()] = {"ts": _now(), "payload": payload}

def _cache_get(ticker: str) -> Optional[Dict[str, Any]]:
    with _TICKER_CACHE_LOCK:
        rec = _TICKER_CACHE.get(ticker.upper())
        if not rec:
            return None
        if _now() - rec["ts"] > _TTL_SECONDS:
            # expired
            del _TICKER_CACHE[ticker.upper()]
            return None
        return rec["payload"]

# -------------------------
# Helpers
# -------------------------
def _to_py(v):
    """Convert numpy/pandas scalars to native python types where possible."""
    if v is None:
        return None
    # pandas / numpy scalars often have 'item' method
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    # ints/floats that are already plain are returned unchanged
    return v

def _safe_get(d: Dict[str, Any], key: str):
    return _to_py(d.get(key)) if isinstance(d, dict) else None

def get_ticker_or_raise(ticker: str) -> yf.Ticker:
    """
    Robust helper to obtain a yfinance.Ticker object while doing multiple checks
    so we don't incorrectly mark valid tickers as invalid.
    Checks (in order):
      1) quick cache
      2) fast_info presence
      3) short history (1d / 5d) existence
      4) info fallback
    """
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    ticker_sym = ticker.strip().upper()

    # Check cache first
    cached = _cache_get(ticker_sym)
    if cached is not None:
        return cached["ticker_obj"]

    # Create ticker object
    ticker_obj = yf.Ticker(ticker_sym)

    # 1) Try fast_info (fast and preferred)
    try:
        fast = getattr(ticker_obj, "fast_info", None)
        if isinstance(fast, dict) and any(v is not None for v in fast.values()):
            _cache_set(ticker_sym, {"ticker_obj": ticker_obj, "fast_info": fast})
            return ticker_obj
    except Exception:
        # don't fail; fall back to history
        pass

    # 2) Try short history (reliable for price existence)
    try:
        hist = ticker_obj.history(period="1d", interval="1m")
        if hist is not None and not hist.empty:
            _cache_set(ticker_sym, {"ticker_obj": ticker_obj, "history_ok": True})
            return ticker_obj
        # if market closed, try last 5 days
        hist5 = ticker_obj.history(period="5d")
        if hist5 is not None and not hist5.empty:
            _cache_set(ticker_sym, {"ticker_obj": ticker_obj, "history_ok": True})
            return ticker_obj
    except Exception:
        # network or remote failure — don't immediately call it invalid, try info
        pass

    # 3) Finally, try info (last resort)
    try:
        info = getattr(ticker_obj, "info", None) or {}
        if isinstance(info, dict) and info:
            _cache_set(ticker_sym, {"ticker_obj": ticker_obj, "info": info})
            return ticker_obj
    except Exception:
        pass

    # All checks failed -> likely invalid/delisted
    raise HTTPException(status_code=404, detail=f"Invalid or delisted ticker symbol: '{ticker_sym}'")

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def read_root():
    return AGENT_DEFINITION

@app.get("/quote", response_model=QuoteResponse)
def get_quote(ticker: str):
    """
    Fetches a recent quote for `ticker`. Preferred source is fast_info; fallbacks to history().
    Returns source field indicating where values were taken from.
    """
    ticker_obj = get_ticker_or_raise(ticker)
    ticker_sym = ticker.strip().upper()

    # Try fast_info first
    try:
        fast = getattr(ticker_obj, "fast_info", {}) or {}
        # Common field names that may appear in fast_info
        price = fast.get("last_price") or fast.get("lastClose") or fast.get("last")
        day_high = fast.get("day_high") or fast.get("dayHigh") or fast.get("high")
        day_low = fast.get("day_low") or fast.get("dayLow") or fast.get("low")
        vol = fast.get("last_volume") or fast.get("volume") or fast.get("regularMarketVolume")
        market_cap = fast.get("market_cap") or fast.get("marketCap")
        if price is not None:
            return QuoteResponse(
                regularMarketPrice=float(_to_py(price)),
                dayHigh=_to_py(day_high),
                dayLow=_to_py(day_low),
                regularMarketVolume=_to_py(vol),
                marketCap=_to_py(market_cap),
                source="fast_info"
            )
    except Exception:
        # swallow and fallback
        pass

    # Fallback: use history period="1d" (minute-resolution if possible), then "5d"
    try:
        hist = ticker_obj.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            hist = ticker_obj.history(period="5d")
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"Could not retrieve recent market data for {ticker_sym}.")
        # Use the last available row as the current price
        last_row = hist.iloc[-1]
        # day high/low from the day's data (if available)
        day_high = _to_py(hist["High"].max()) if "High" in hist.columns else None
        day_low = _to_py(hist["Low"].min()) if "Low" in hist.columns else None
        volume_sum = None
        if "Volume" in hist.columns:
            # Use the sum of volume for the period as a proxy for day's volume
            try:
                volume_sum = int(_to_py(hist["Volume"].sum()))
            except Exception:
                volume_sum = None
        return QuoteResponse(
            regularMarketPrice=float(_to_py(last_row["Close"])),
            dayHigh=day_high,
            dayLow=day_low,
            regularMarketVolume=volume_sum,
            marketCap=None,
            source="history"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error fetching market data for {ticker_sym}: {str(exc)}")

@app.post("/history")
def get_history(req: HistoryRequest):
    """
    Returns historical OHLCV as list[dict] for the specified period.
    Period examples: '1d', '5d', '1mo', '3mo', '6mo', '1y', '5y', 'max'
    """
    ticker_obj = get_ticker_or_raise(req.ticker)
    ticker_sym = req.ticker.strip().upper()
    try:
        hist = ticker_obj.history(period=req.period)
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for '{ticker_sym}' with period '{req.period}'.")
        hist = hist.reset_index()
        # Ensure the Date column is serializable
        if "Date" in hist.columns:
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
        # Convert types to plain python types
        records = []
        for _, row in hist.iterrows():
            rec = {}
            for col in hist.columns:
                rec[col] = _to_py(row[col])
            records.append(rec)
        return records
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error fetching historical data for {ticker_sym}: {str(exc)}")

@app.get("/company_info", response_model=CompanyInfoResponse)
def get_company_info(ticker: str):
    """
    Returns general company information (longName, sector, industry, longBusinessSummary, website).
    Uses info (may be slow); this endpoint is intended for company metadata, not price.
    """
    ticker_obj = get_ticker_or_raise(ticker)
    ticker_sym = ticker.strip().upper()
    try:
        info = getattr(ticker_obj, "info", None) or {}
        # Some tickers may not have longBusinessSummary; return 404 in that case
        long_summary = info.get("longBusinessSummary") or info.get("longBusinessSummary")
        if not long_summary:
            raise HTTPException(status_code=404, detail=f"No company summary found for {ticker_sym}.")
        return CompanyInfoResponse(
            longName=_to_py(info.get("longName") or info.get("shortName") or ticker_sym),
            sector=_to_py(info.get("sector")),
            industry=_to_py(info.get("industry")),
            longBusinessSummary=_to_py(long_summary),
            website=_to_py(info.get("website"))
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error fetching company info for {ticker_sym}: {str(exc)}")

@app.get("/key_statistics", response_model=KeyStatisticsResponse)
def get_key_statistics(ticker: str):
    """
    Returns key statistics commonly used for financial analysis.
    Relies on ticker.info; availability depends on Yahoo's dataset for the ticker.
    """
    ticker_obj = get_ticker_or_raise(ticker)
    ticker_sym = ticker.strip().upper()
    try:
        info = getattr(ticker_obj, "info", None) or {}
        stats_keys = [
            "marketCap", "beta", "trailingPE", "forwardPE",
            "trailingEps", "forwardEps", "dividendYield", "payoutRatio", "priceToBook"
        ]
        data = {k: _to_py(info.get(k)) for k in stats_keys}
        # marketCap is a critical metric — if missing, we consider that stats might not be available
        if data.get("marketCap") is None:
            raise HTTPException(status_code=404, detail=f"Key statistics not available for {ticker_sym}.")
        return KeyStatisticsResponse(**data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error fetching key statistics for {ticker_sym}: {str(exc)}")

@app.get("/recommendations")
def get_recommendations(ticker: str):
    """
    Fetches analyst recommendations DataFrame and returns the last 5 rows as JSON.
    """
    ticker_obj = get_ticker_or_raise(ticker)
    ticker_sym = ticker.strip().upper()
    try:
        recs = None
        try:
            recs = ticker_obj.recommendations
        except Exception:
            recs = None
        if recs is None or (isinstance(recs, pd.DataFrame) and recs.empty):
            raise HTTPException(status_code=404, detail=f"No analyst recommendations found for {ticker_sym}.")
        # convert to records (last 5)
        recs_to_return = recs.tail(5).reset_index()
        records = []
        for _, row in recs_to_return.iterrows():
            rec = {}
            for col in recs_to_return.columns:
                rec[col] = _to_py(row[col])
            records.append(rec)
        return records
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error fetching recommendations for {ticker_sym}: {str(exc)}")

# -------------------------
# If run directly (development)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    # For local dev only. In production, run using your process manager / container and don't use reload=True.
    # Use 0.0.0.0 to bind to all interfaces for better compatibility
    uvicorn.run("finance_agent:app", host="0.0.0.0", port=8010, reload=False)

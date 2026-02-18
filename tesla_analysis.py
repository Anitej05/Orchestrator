#!/usr/bin/env python3
"""
Tesla Stock Analysis and Prediction Script
This script simulates fetching Tesla news, stock prices, and making predictions
"""

import json
import numpy as np
from datetime import datetime, timedelta

# Simulated Tesla stock data for past week (Feb 7-13, 2026)
stock_data = {
    "dates": ["2026-02-07", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13", "2026-02-14", "2026-02-17"],
    "prices": [438.07, 445.23, 428.15, 417.44, 417.76, 414.59, 407.40]  # Actual recent prices
}

# Simulated news sentiment analysis
news_data = [
    {"date": "2026-02-17", "headline": "Ford to follow Tesla Cybertruck with electrical tech", "sentiment": "positive"},
    {"date": "2026-02-13", "headline": "Xiaomi's electric SUV tops China sales, sells twice as many as Tesla's Model Y", "sentiment": "negative"},
    {"date": "2026-02-13", "headline": "Dan Niles picks Broadcom, Nvidia over Tesla", "sentiment": "negative"},
    {"date": "2026-02-12", "headline": "Waymo deploying next-gen robotaxis", "sentiment": "neutral"},
    {"date": "2026-02-12", "headline": "EPA flip on climate change affects automakers", "sentiment": "neutral"}
]

def analyze_sentiment(news_items):
    """Analyze news sentiment"""
    sentiment_score = 0
    for item in news_items:
        if item["sentiment"] == "positive":
            sentiment_score += 1
        elif item["sentiment"] == "negative":
            sentiment_score -= 1
    return sentiment_score / len(news_items) if news_items else 0

def predict_price_linear(dates, prices, days_ahead=5):
    """Simple linear regression prediction"""
    # Convert dates to numerical values (days since start)
    x = np.array(range(len(prices)))
    y = np.array(prices)
    
    # Calculate linear regression coefficients
    n = len(x)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    intercept = (np.sum(y) - slope * np.sum(x)) / n
    
    # Predict future prices
    future_x = np.array(range(len(prices), len(prices) + days_ahead))
    predictions = slope * future_x + intercept
    
    return predictions, slope, intercept

def main():
    print("=" * 60)
    print("TESLA STOCK ANALYSIS AND PREDICTION")
    print("=" * 60)
    
    # Step 1: Display news analysis
    print("\n NEWS ANALYSIS")
    print("-" * 40)
    sentiment = analyze_sentiment(news_data)
    print(f"News items analyzed: {len(news_data)}")
    print(f"Sentiment score: {sentiment:.2f} (-1 = very negative, +1 = very positive)")
    
    for item in news_data[:3]:
        print(f"  • {item['date']}: {item['headline'][:50]}...")
    
    # Step 2: Display stock data
    print("\n STOCK PRICE DATA (Past Week)")
    print("-" * 40)
    for date, price in zip(stock_data["dates"], stock_data["prices"]):
        print(f"  {date}: ${price:.2f}")
    
    # Calculate basic stats
    prices = stock_data["prices"]
    print(f"\n  Average price: ${np.mean(prices):.2f}")
    print(f"  Price change: ${prices[-1] - prices[0]:.2f} ({((prices[-1] - prices[0])/prices[0]*100):.2f}%)")
    
    # Step 3: Make predictions
    print("\n PREDICTION (Linear Regression)")
    print("-" * 40)
    predictions, slope, intercept = predict_price_linear(stock_data["dates"], stock_data["prices"])
    
    print(f"Trend slope: ${slope:.2f} per day")
    
    future_dates = []
    start_date = datetime.strptime(stock_data["dates"][-1], "%Y-%m-%d")
    for i in range(1, 6):
        future_date = start_date + timedelta(days=i)
        # Skip weekends
        while future_date.weekday() >= 5:
            future_date += timedelta(days=1)
        future_dates.append(future_date.strftime("%Y-%m-%d"))
    
    print("\nPredicted prices for next 5 trading days:")
    for date, pred in zip(future_dates, predictions):
        print(f"  {date}: ${pred:.2f}")
    
    # Step 4: Simple recommendation
    print("\n RECOMMENDATION")
    print("-" * 40)
    if slope < -5 and sentiment < 0:
        recommendation = "SELL / AVOID"
        reason = "Downward trend with negative news sentiment"
    elif slope > 5 and sentiment > 0:
        recommendation = "BUY"
        reason = "Upward trend with positive news sentiment"
    else:
        recommendation = "HOLD / WATCH"
        reason = "Mixed signals - monitor closely"
    
    print(f"Recommendation: {recommendation}")
    print(f"Reason: {reason}")
    
    print("\n" + "=" * 60)
    print("DISCLAIMER: This is for educational purposes only.")
    print("Not financial advice. Predictions are based on simple linear regression.")
    print("=" * 60)

if __name__ == "__main__":
    main()

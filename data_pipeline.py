
import yfinance as yf
import pandas as pd
from transformers import pipeline
import numpy as np
import requests # <- Yeh naya import hai

# Ek function jo historical data fetch karega (Ismein koi badlav nahi hai)
def get_historical_data(ticker, period='1y'):
    """
    Yahoo Finance se historical stock data (OHLCV) fetch karta hai.
    """
    try:
        stock_data = yf.Ticker(ticker)
        df = stock_data.history(period=period)
        if df.empty:
            print(f"No data found for {ticker}")
            return None
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# --- PURANE get_sentiment FUNCTION KI JAGAH YEH NAYA FUNCTION USE KAREIN ---
def get_sentiment(ticker):
    """
    NewsAPI ka use karke live financial news fetch karta hai aur uska sentiment nikalta hai.
    """
    # APNI NEWSAPI KEY YAHA PASTE KAREIN
    api_key = "*********************************"
    
    # Ticker se company ka naam search karna behtar rehta hai
    company_names = {"GOOGL": "Alphabet", "MSFT": "Microsoft", "AAPL": "Apple", "NVDA": "Nvidia"}
    search_term = company_names.get(ticker.upper(), ticker)

    # API request ke liye URL
    url = f"https://newsapi.org/v2/everything?q={search_term}&language=en&sortBy=relevancy&apiKey={api_key}"

    try:
        # News fetch karne ki koshish
        response = requests.get(url)
        response.raise_for_status()  # Agar koi error ho (jaise 404, 500) to ruk jaye
        articles = response.json().get("articles", [])
        
        # Top 15 headlines ka istemal karenge analysis ke liye
        news_headlines = [article['title'] for article in articles[:15] if article and article['title']]

        if not news_headlines:
            print(f"'{search_term}' ke liye koi taaza khabar nahi mili. Neutral sentiment return kar rahe hain.")
            return 0.0 # Agar koi khabar na mile to neutral score (0.0) de

    except requests.exceptions.RequestException as e:
        print(f"News API se data fetch karne mein error: {e}. Neutral sentiment return kar rahe hain.")
        return 0.0

    # Baaki ka sentiment analysis logic pehle jaisa hi hai
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = sentiment_classifier(news_headlines)
    
    positive_scores = [s['score'] for s in sentiments if s['label'] == 'POSITIVE']
    negative_scores = [s['score'] for s in sentiments if s['label'] == 'NEGATIVE']
    
    avg_pos = np.mean(positive_scores) if positive_scores else 0
    avg_neg = np.mean(negative_scores) if negative_scores else 0
    
    overall_sentiment = avg_pos - avg_neg
    return overall_sentiment

# Main function jo sabse zaroori hai (Ismein koi badlav nahi hai)
def create_dataset(ticker, period='1y'):
    """
    Historical data aur sentiment score ko combine karke final dataset banata hai.
    """
    print(f"Fetching data for {ticker}...")
    df = get_historical_data(ticker, period=period)
    if df is None:
        return None
    
    print(f"Analyzing sentiment for {ticker}...")
    sentiment_score = get_sentiment(ticker)
    
    df['sentiment'] = sentiment_score
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
    
    print("Dataset created successfully! Here's a preview:")
    return df

# Niche ke test code mein koi badlav nahi hai
if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    final_data = create_dataset(ticker_symbol)
    if final_data is not None:
        print(final_data.head())
        print(final_data.tail())

        print(f"Shape of the final dataset: {final_data.shape}")

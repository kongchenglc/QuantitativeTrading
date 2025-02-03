import pandas as pd
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import nltk

nltk.download('vader_lexicon')

api_key = "d69e1e66950947279a7beb45fc221fd6"
ticker = "NVDA"


def fetch_news(api_key, ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)["compound"]

articles = fetch_news(api_key, ticker)
news_list = []
for article in articles:
    date = pd.to_datetime(article["publishedAt"]).date()
    description = article["description"] if article["description"] is not None else ""
    news_list.append({
        "date": date,
        "title": article["title"],
        "sentiment": analyze_sentiment(article["title"] + " " + description)
    })

news_df = pd.DataFrame(news_list)
news_df = news_df.groupby("date")["sentiment"].mean().reset_index()
news_df["date"] = pd.to_datetime(news_df["date"])

news_df.to_csv("./data/NVDA_with_Sentiment.csv")
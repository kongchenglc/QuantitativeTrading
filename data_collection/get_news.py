import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")


def analyze_sentiment_finbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.detach().numpy()[0]  # [positive, neutral, negative]


def fetch_news(years=6):
    news_data = []

    current_year = datetime.now().year
    for year in range(current_year - years, current_year + 1):
        date_start = f"{year}-01-01"
        date_end = f"{year}-12-31"

        url = f"https://news.google.com/rss/search?q=NVIDIA+after:{date_start}+before:{date_end}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.content, "xml")

        seen_links = set()
        for item in soup.find_all("item"):
            link = item.link.text.split("?oc=")[0]
            if link in seen_links:
                continue
            seen_links.add(link)

            pub_date = pd.to_datetime(item.pubDate.text)
            title = item.title.text
            sentiment = analyze_sentiment_finbert(title)
            news_data.append(
                {
                    "Date": pub_date,
                    "Title": title,
                    "Source": item.source.text,
                    "Link": link,
                    "Sentiment_Positive": sentiment[0],
                    "Sentiment_Neutral": sentiment[1],
                    "Sentiment_Negative": sentiment[2],
                }
            )

    df = pd.DataFrame(news_data)
    df = df.drop_duplicates(subset=["Link"], keep="first")

    df.sort_values("Date", ascending=True, inplace=True)
    df.set_index("Date", inplace=True)
    df.to_csv("data/nvidia_news_en.csv", index=True)
    return df

import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


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
            sentiment = analyzer.polarity_scores(title)["compound"]
            news_data.append(
                {
                    "date": pub_date,
                    "title": title,
                    "source": item.source.text,
                    "link": link,
                    "sentiment": sentiment,
                }
            )

    df = pd.DataFrame(news_data)
    df = df.drop_duplicates(subset=["link"], keep="first")

    df.sort_values("date", ascending=False, inplace=True)
    df.to_csv("data/nvidia_news_en.csv", index=False)
    return df

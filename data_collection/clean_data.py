import pandas as pd
from data_collection.get_historial_stock_data import fetch_historial_stock_data
from data_collection.get_marco_data import fetch_macro_data
from data_collection.get_news import fetch_news


def get_cleaned_data():
    print('Fetching data...')
    historical_data = fetch_historial_stock_data()
    articles = fetch_news()
    macro_data = fetch_macro_data()

    historical_data.index = pd.to_datetime(historical_data.index).date
    articles.index = pd.to_datetime(articles.index).date
    macro_data.index = pd.to_datetime(macro_data.index).date

    articles_grouped = articles.groupby(articles.index).agg(
        Sentiment_Positive=("Sentiment_Positive", "mean"),
        Sentiment_Neutral=("Sentiment_Neutral", "mean"),
        Sentiment_Negative=("Sentiment_Negative", "mean"),
    )

    merged_data = pd.merge(
        historical_data,
        articles_grouped,
        left_index=True,
        right_index=True,
        how="outer",
    )
    merged_data = pd.merge(
        merged_data, macro_data, left_index=True, right_index=True, how="outer"
    )
    merged_data = merged_data[merged_data.index >= pd.to_datetime('2019-02-15').date()]
    merged_data.index.name = "Date"

    merged_data[['Open', 'Close', 'High', 'Low', 'Volume']] = merged_data[['Open', 'Close', 'High', 'Low', 'Volume']].ffill()
    merged_data[['Dividends', 'Stock Splits']] = merged_data[['Dividends', 'Stock Splits']].fillna(0.0)

    merged_data['Return'] = (merged_data['Close'] - merged_data['Close'].shift(1)) / merged_data['Close'].shift(1)
    merged_data = merged_data[1:]

    merged_data[['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14', 'MACD', 'Signal_Line', 'BB_Mid', 'BB_Upper', 'BB_Lower']] = merged_data[
        ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI_14', 'MACD', 'Signal_Line', 'BB_Mid', 'BB_Upper', 'BB_Lower']].ffill()

    merged_data[['Interest Rate', 'Inflation Rate', 'Unemployment Rate', 'GDP Growth']] = merged_data[
        ['Interest Rate', 'Inflation Rate', 'Unemployment Rate', 'GDP Growth']].ffill()

    merged_data[['Sentiment_Positive', 'Sentiment_Neutral', 'Sentiment_Negative']] = merged_data[['Sentiment_Positive', 'Sentiment_Neutral', 'Sentiment_Negative']].ffill()
    
    
    if merged_data.isnull().values.any():
        missing_rows = merged_data[merged_data.isnull().any(axis=1)]
        for index, row in missing_rows.iterrows():
            missing_columns = row[row.isnull()].index.tolist()
            print(f"Data contains missing values (NaN)! Row {index} has missing data in columns: {missing_columns}")
       
    merged_data = merged_data.dropna()

    merged_data.to_csv("data/cleaned_data.csv", index=True)
    return merged_data

import pandas as pd
from data_collection.get_historial_stock_data import fetch_historial_stock_data
from data_collection.get_marco_data import fetch_macro_data
from data_collection.get_news import fetch_news
from datetime import datetime


def get_cleaned_data():
    print('Fetching data...')
    # historical_data = fetch_historial_stock_data()
    historical_data = pd.read_csv("data/nvidia_historical_data.csv", index_col="Date")
    articles = fetch_news()
    macro_data = fetch_macro_data()
    
    # Because we can only get data from one month ago, need to shift(1)
    current_month = datetime.today().date().replace(day=1)
    empty_row = pd.DataFrame([[None] * len(macro_data.columns)], columns=macro_data.columns, index=[current_month])
    macro_data = pd.concat([macro_data, empty_row])
    macro_data[['Interest Rate', 'Inflation Rate', 'Unemployment Rate', 'GDP Growth']] = macro_data[
        ['Interest Rate', 'Inflation Rate', 'Unemployment Rate', 'GDP Growth']].shift(1)


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
    print("Missed row dropped!")
    
    # Ensure the dates are continuous
    full_date_range = pd.date_range(start=merged_data.index.min(), end=merged_data.index.max(), freq='D')
    merged_data = merged_data.reindex(full_date_range)
    merged_data = merged_data.ffill()
    
    merged_data["Return"] = merged_data["Close"].pct_change()
    merged_data = merged_data[1:]
    merged_data["Weekday"] = merged_data.index.weekday
    merged_data["Month"] = merged_data.index.month

    merged_data.index.name = "Date"
    merged_data.to_csv("data/cleaned_data.csv", index=True)
    return merged_data

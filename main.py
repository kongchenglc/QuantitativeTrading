from data_collection.get_historial_stock_data import fetch_historial_stock_data
from data_collection.get_marco_data import fetch_macro_data
from data_collection.get_news import fetch_news

def main():
    news_api_key = "d69e1e66950947279a7beb45fc221fd6"
    fred_api_key = "5363320bd7af2401dafe2914c2c974c8"
    ticker = "NVDA"
    
    historical_data = fetch_historial_stock_data()
    articles = fetch_news(news_api_key, ticker)
    macro_data = fetch_macro_data(fred_api_key) 

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd

def fetch_historial_stock_data():

    nvda = yf.Ticker("NVDA")
    data = nvda.history(period="6y")


    # Calculate Simple Moving Average (SMA)
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Calculate Exponential Moving Average (EMA)
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Calculate Relative Strength Index (RSI)
    def compute_rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['RSI_14'] = compute_rsi(data['Close'])

    # Calculate MACD Indicator
    def compute_macd(data, short=12, long=26, signal=9):
        short_ema = data['Close'].ewm(span=short, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long, adjust=False).mean()
        data['MACD'] = short_ema - long_ema
        data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()

    compute_macd(data)

    # Calculate Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['Close'].rolling(window=20).std()


    data.to_csv("./data/NVDA_Historical_Data.csv")

    return data
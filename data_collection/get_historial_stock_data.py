import yfinance as yf
import pandas as pd

def fetch_historial_stock_data():
    # Fetch NVIDIA historical data for the last 6 years
    nvda = yf.Ticker("NVDA")
    data = nvda.history(period="6y")

    # -------------------------------
    # Calculate Simple Moving Average (SMA)
    # The SMA is the average price over a specified time period, which smooths out price data
    # to help identify the trend direction.
    # Here, we calculate 10-day and 50-day SMAs to observe short-term and mid-term trends.
    # -------------------------------
    data['SMA_10'] = data['Close'].rolling(window=10).mean()  # 10-day SMA
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day SMA

    # -------------------------------
    # Calculate Exponential Moving Average (EMA)
    # The EMA is similar to the SMA but gives greater weight to recent prices,
    # making it more sensitive to recent price changes.
    # We compute 10-day and 50-day EMAs for capturing short-term and mid-term price movements.
    # -------------------------------
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()  # 10-day EMA
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()  # 50-day EMA

    # -------------------------------
    # Calculate Relative Strength Index (RSI)
    # The RSI is a momentum indicator that measures the speed and change of price movements.
    # It is used to identify potential overbought or oversold conditions.
    # A common period for RSI calculation is 14 days.
    # -------------------------------
    def compute_rsi(series, period=14):
        delta = series.diff(1)  # Compute daily price change
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Average gain over the period
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Average loss over the period
        rs = gain / loss  # Compute Relative Strength (RS)
        return 100 - (100 / (1 + rs))  # Compute RSI

    data['RSI_14'] = compute_rsi(data['Close'])  # 14-day RSI

    # -------------------------------
    # Calculate Moving Average Convergence Divergence (MACD)
    # The MACD measures the difference between short-term and long-term EMAs,
    # providing insights into the momentum of a stock's price.
    # A signal line, usually a 9-day EMA of the MACD line, is also computed.
    # Crossovers between the MACD line and the signal line can indicate buy or sell signals.
    # -------------------------------
    def compute_macd(data, short=12, long=26, signal=9):
        short_ema = data['Close'].ewm(span=short, adjust=False).mean()  # Compute short-term EMA (typically 12 days)
        long_ema = data['Close'].ewm(span=long, adjust=False).mean()    # Compute long-term EMA (typically 26 days)
        data['MACD'] = short_ema - long_ema  # Calculate MACD line as the difference between short and long EMA
        data['Signal_Line'] = data['MACD'].ewm(span=signal, adjust=False).mean()  # Compute Signal Line as the EMA of MACD

    compute_macd(data)

    # -------------------------------
    # Calculate Bollinger Bands
    # Bollinger Bands consist of a middle band (a moving average) and an upper and lower band based on
    # the standard deviation of the price. The upper band is the middle band plus two times the standard deviation,
    # while the lower band is the middle band minus two times the standard deviation.
    # They help assess market volatility and potential overbought or oversold conditions.
    # -------------------------------
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()  # Middle band: 20-day moving average
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['Close'].rolling(window=20).std()  # Upper Bollinger Band
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['Close'].rolling(window=20).std()  # Lower Bollinger Band

    # Save the processed data to a CSV file
    data.to_csv("./data/nvidia_historical_data.csv")

    return data
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.dates as mdates

def determine_stock_trend(stock_name, start_date, end_date, day_window=3):
    # Download NVIDIA's stock data between start and end dates
    # End date is not included
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    # Calculate simple moving average based on closing prices
    # Day window is set to 3 by default
    stock_data['Moving_Average'] = stock_data['Close'].rolling(day_window).mean()

    # Calculate the slope that determines the stock trend using linear regression
    # Day window is set to 3
    stock_data['Slope'] = calculate_slope_with_linear_regression(stock_data['Moving_Average'], day_window)

    # Initialize Signal and Signal_Detail
    # If slope crosses zero, a 'buy' signal should be generated
    # If slope is trending below zero, a 'sell' signal should be generated
    stock_data['Signal'] = 0
    stock_data['Signal_Detail'] = 'Neutral' # set to 'Neutral' by default

    generate_buy_and_sell_signals(stock_data)

    # Plot the slope and price trend
    plot_graphs(stock_name, day_window, stock_data)

    # Print trend slope summary and signals
    print_summary(stock_data)

    return stock_data

# Apply linear regression to calculate the slope and determine the stock trend
def calculate_slope_with_linear_regression(data, day_window):
    
    def apply_linear_regression(y):
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    return data.rolling(day_window).apply(apply_linear_regression)

def generate_buy_and_sell_signals(stock_data):
    is_buy_signal_generated = False

    for i in range(1, len(stock_data)):
        current_slope_value = stock_data['Slope'].iloc[i]
        previous_slope_value = stock_data['Slope'].iloc[i-1]
        index = stock_data.index[i]

        # Generate 'Buy' signal if the slope is trending over zero
        # This indicates an uptrend
        if not is_buy_signal_generated and current_slope_value > 0 and previous_slope_value <= 0:
            stock_data.loc[index, 'Signal'] = 1
            stock_data.loc[index, 'Signal_Detail'] = 'Buy'
            is_buy_signal_generated = True

        # Generate 'Sell' signal if the slope is trending below zero
        # This indicates a downtrend
        elif is_buy_signal_generated and current_slope_value < 0 and previous_slope_value >= 0:
            stock_data.loc[index, 'Signal'] = -1
            stock_data.loc[index, 'Signal_Detail'] = 'Sell'
            is_buy_signal_generated = False

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_graphs(stock_name, day_window, stock_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # UPPER SUBPLOT (Price and Moving Average)
    ax1.plot(
        stock_data.index,
        stock_data['Close'],
        label='Close Price',
        alpha=0.7
    )
    ax1.plot(
        stock_data.index,
        stock_data['Moving_Average'],
        label=f'MA ({day_window} days)',
        alpha=0.8
    )

    ax1.set_ylabel('Price')
    ax1.set_title(f'{stock_name} Stock Price with {day_window}-Day MA')
    ax1.legend()
    ax1.grid(True)

    # Configure date axis
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # LOWER SUBPLOT (Slope)
    ax2.plot(
        stock_data.index,
        stock_data['Slope'],
        label='Slope of MA',
        color='green',
        alpha=0.8
    )
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Slope')
    ax2.set_title('Moving Average Trend Slope')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()

def print_summary(stock_data):
    print("\nSummary:")
    print(f"Average Slope: {stock_data['Slope'].mean():.4f}")
    print("\nSignals (March only):")
    march_signals = stock_data[(stock_data['Signal'] != 0) & (stock_data.index.month == 3)] # Only print March signals
    print(march_signals[['Close', 'Signal_Detail', 'Slope']])

stock_name = 'NVDA'
start_date = '2025-01-01'
end_date = '2025-03-24'
result = determine_stock_trend(stock_name, start_date, end_date, 3)
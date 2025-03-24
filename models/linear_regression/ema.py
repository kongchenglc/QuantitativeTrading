import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.dates as mdates

def determine_stock_trend(stock_name, start_date, end_date, day_window=3):
    stock_data = yf.download(stock_name, start=start_date, end=end_date)

    # Use EMA instead of SMA
    stock_data['Moving_Average'] = stock_data['Close'].ewm(span=day_window, adjust=False).mean()

    # Linear regression on EMA
    stock_data['Slope'] = calculate_slope_with_linear_regression(stock_data['Moving_Average'], day_window)

    # Initialize signal columns
    stock_data['Signal'] = 0
    stock_data['Signal_Detail'] = 'Neutral'

    generate_buy_and_sell_signals(stock_data)

    plot_graphs(stock_name, day_window, stock_data)

    print_summary(stock_data)

    predict_tomorrow_action(stock_data)

    return stock_data

def calculate_slope_with_linear_regression(data, day_window):
    def apply_linear_regression(y):
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    return data.rolling(day_window).apply(apply_linear_regression)

def generate_buy_and_sell_signals(stock_data, slope_threshold=0.02):
    is_position_open = False

    for i in range(2, len(stock_data)):
        try:
            # Extract scalar values
            slope_now = stock_data['Slope'].iloc[i].item()
            slope_prev = stock_data['Slope'].iloc[i - 1].item()
            slope_prev2 = stock_data['Slope'].iloc[i - 2].item()
            close_now = stock_data['Close'].iloc[i].item()
            ma_now = stock_data['Moving_Average'].iloc[i].item()
        except ValueError:
            # Skip rows where .item() fails (usually NaN or multiple values)
            continue

        index = stock_data.index[i]

        # --- BUY Logic ---
        if not is_position_open and slope_now > slope_threshold:
            stock_data.loc[index, 'Signal'] = 1
            stock_data.loc[index, 'Signal_Detail'] = 'Buy (Uptrend Confirmed)'
            is_position_open = True

        elif not is_position_open and slope_now > slope_prev and slope_prev > slope_prev2 and close_now > ma_now:
            stock_data.loc[index, 'Signal'] = 1
            stock_data.loc[index, 'Signal_Detail'] = 'Buy (Momentum Shift)'
            is_position_open = True

        # --- SELL Logic ---
        elif is_position_open and slope_now < -slope_threshold:
            stock_data.loc[index, 'Signal'] = -1
            stock_data.loc[index, 'Signal_Detail'] = 'Sell (Downtrend Confirmed)'
            is_position_open = False

        elif is_position_open and slope_now < slope_prev and slope_prev < slope_prev2 and close_now < ma_now:
            stock_data.loc[index, 'Signal'] = -1
            stock_data.loc[index, 'Signal_Detail'] = 'Sell (Momentum Shift)'
            is_position_open = False


def plot_graphs(stock_name, day_window, stock_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot price and EMA
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', alpha=0.7)
    ax1.plot(stock_data.index, stock_data['Moving_Average'], label=f'EMA ({day_window} days)', alpha=0.8)
    ax1.set_ylabel('Price')
    ax1.set_title(f'{stock_name} Stock Price with {day_window}-Day EMA')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Plot slope of EMA
    ax2.plot(stock_data.index, stock_data['Slope'], label='Slope of EMA', color='green', alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Slope')
    ax2.set_title('EMA Trend Slope')
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
    march_signals = stock_data[(stock_data['Signal'] != 0) & (stock_data.index.month == 3)]
    print(march_signals[['Close', 'Signal_Detail', 'Slope']])

def predict_tomorrow_action(stock_data, slope_threshold=0.02):
    print("\n📊 PREDICTION FOR TOMORROW:")

    if len(stock_data) < 5:
        print("❌ Not enough data to make a prediction.")
        return

    recent = stock_data.tail(5).copy()
    print("\n📈 Last 5 Days of Market Data:")
    print(recent[['Close', 'Moving_Average', 'Slope']].round(3))

    # Extract latest values safely
    slope_now = recent['Slope'].iloc[-1].item()
    slope_prev = recent['Slope'].iloc[-2].item()
    slope_prev2 = recent['Slope'].iloc[-3].item()
    close_now = recent['Close'].iloc[-1].item()
    ma_now = recent['Moving_Average'].iloc[-1].item()

    # Derived values
    slope_change = slope_now - slope_prev
    price_vs_ema_pct = (close_now - ma_now) / ma_now * 100

    # Display momentum breakdown
    print("\n📉 Momentum Breakdown:")
    print(f"Slope Trend     : {slope_prev2:.4f} -> {slope_prev:.4f} -> {slope_now:.4f}")
    print(f"Change in Slope : {slope_change:+.4f}")
    print(f"Close vs EMA    : {close_now:.2f} vs {ma_now:.2f} ({price_vs_ema_pct:+.2f}%)")

    # Decision logic with detailed reasoning
    print("\n🧠 Decision:")

    if slope_now > slope_threshold and close_now > ma_now:
        print("🔼 BUY — Strong Uptrend Confirmed:")
        print(f"    --> Slope is high ({slope_now:.4f} > {slope_threshold}) and price is above EMA")
    elif slope_now > slope_prev > slope_prev2 and close_now > ma_now:
        print("📈 BUY — Momentum Shift Detected:")
        print("     --> Slope is increasing and price is above EMA")
    elif slope_now < -slope_threshold:
        print("🔽 SELL — Strong Downtrend Detected:")
        print(f"    --> Slope is sharply negative ({slope_now:.4f} < -{slope_threshold})")
    elif slope_now < slope_prev < slope_prev2 and close_now < ma_now:
        print("📉 SELL — Momentum Weakening:")
        print("     --> Slope is decreasing and price is below EMA")
    else:
        print("⏸️ HOLD — No Clear Signal:")
        print("     --> Slope and price movement do not confirm a strong trend")

# Run it on NVIDIA
stock_name = 'NVDA'
start_date = '2025-01-01'
end_date = '2025-03-28'
result = determine_stock_trend(stock_name, start_date, end_date, day_window=3)
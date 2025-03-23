# 📈 Quantitative Trading: Nvidia Stock Prediction 🚀

The project includes a predictive model for forecasting Nvidia's stock price using an LSTM-based neural network.

---

## 📌 Current Features

- **Data Collection**

  - **Historical Stock Price Analysis:**  
    Utilizes the [Yahoo Finance API](https://finance.yahoo.com/) to fetch Nvidia (NVDA) stock data, while also computing key technical indicators:
    - Simple Moving Averages (SMA)
    - Exponential Moving Averages (EMA)
    - Relative Strength Index (RSI)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
  - **Macro-Economic Data:**  
    Leverages the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API via the `fredapi` package to obtain:
    - Interest Rates
    - Inflation Rates
    - Unemployment Rates
    - GDP Data
  - **News & Sentiment Analysis:**  
    Scrapes Nvidia-related news from the Google News RSS feed and computes sentiment scores using VADER.

- **Stock Prediction Model (LSTM)**

  - **LSTM-based Stock Price Prediction:**  
    Implements a Long Short-Term Memory (LSTM) model to predict Nvidia's stock closing prices based on historical data.
  - **Features:**
    - Predicts closing prices for the next day.
    - Trains on historical stock data, incorporating sequence-based time series analysis.

- **Linear Regression Model**
  - Implements a linear regression model that is based on moving day averages. See `models/linear_regression`
  - **Features:**
    - Pulls NVIDIA's data from `yfinance` (since January 1, 2025)
    - Calculates the trend slope and plots the result
    - The trend slope provides insights into whether the price is trending upwards or downwards. The insight can help with making more informed decisions

---

## 🛠 Installation

Ensure you have **Python 3.8+** installed. Then clone the repository and install the required dependencies:

```bash
git clone https://github.com/kongchenglc/QuantitativeTrading.git
cd QuantitativeTrading
pip install -r requirements.txt
```

---

## 🧠 LSTM Stock Prediction Model

The LSTM model is designed to predict the closing prices of Nvidia stock based on historical price data. It uses a sequence-based model where past prices are used to predict future prices.

### Key Components:

- **LSTM Model:**
  - Input: Historical closing prices (last `n_steps` days).
  - Output: Predicted closing price for the next day.
  - The model is built using PyTorch and consists of an LSTM layer followed by a fully connected layer.
- **Data Preprocessing:**
  - The data is normalized using MinMaxScaler.
  - The dataset is split into training and testing sets, with a rolling window approach for time-series prediction.

### Usage:

1. **Prepare the Data:**  
   The data should be in a DataFrame format containing historical stock prices.
2. **Train the Model:**  
   Use the `StockPredictor` class to preprocess the data and train the model.

   ```bash
   python main.py
   ```

3. **Backtest and Predict:**  
   After training, perform backtesting on the test set and predict future stock prices.

   Use Optuna to get the best hyperparameters:

   ```bash
   python -m util.optuna_tuner.py
   ```

   It will save the best result in the models > model_loder. Then you can recur the best model by:

   ```bash
   python -m models.model_loder.lstm_loader
   ```

---

## 📈 Linear Regression Model

⚠️ Please note that this model should be run separately

1. Set your directory to `models/linear_regression`
2. Run `python main.py`

Output: The `main.py` will run and create plots in a new window

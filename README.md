# 📈 Quantitative Trading: Nvidia Stock Prediction 🚀

**Work In Progress (WIP):** Currently, only the data collection modules are implemented.

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

---

## 🛠 Planned Features
- **Machine Learning Models:**  
  Develop and integrate predictive models aimed at forecasting Nvidia's stock price movements.

---

## 🛠 Installation
Ensure you have **Python 3.8+** installed. Then clone the repository and install the required dependencies:

```bash
git clone https://github.com/kongchenglc/QuantitativeTrading.git
cd QuantitativeTrading
pip install -r requirements.txt
```

---

## ⚙️ Usage
Run the main script to execute all the data collection procedures. The script collects:
- Historical stock data (saved as `data/nvidia_historical_data.csv`)
- Macro-economic data (saved as `data/us_macro_data.csv`)
- Nvidia news articles with sentiment scores (saved as `data/nvidia_news_en.csv`)

```bash
python main.py
```

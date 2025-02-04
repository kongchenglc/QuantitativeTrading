# **📈 Quantitative Trading: Nvidia Stock Prediction 🚀**  
A **quantitative trading project** that leverages **data mining and machine learning** to predict **Nvidia (NVDA) stock price movements** and optimize trading strategies.
---

## **📌 Features**
✅ **Historical stock price analysis** using **Yahoo Finance API**  
✅ **Feature engineering** for stock market prediction (e.g., moving averages, RSI, MACD)  
✅ **Machine learning models** for **price prediction**  
✅ **Backtesting trading strategies** to optimize portfolio returns  

---

## **🛠 Installation**
Make sure you have **Python 3.8+** installed, then clone the repository:
```bash
git clone https://github.com/kongchenglc/QuantitativeTrading.git
cd QuantitativeTrading
```
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## **📊 Data Collection**
The project fetches historical **Nvidia (NVDA) stock data** using `yfinance`:
```python
import yfinance as yf

# Download Nvidia stock data
nvda = yf.download("NVDA", start="2015-01-01", end="2024-01-01")
nvda.to_csv("data/nvda_stock.csv")
```

---

## **📈 Feature Engineering**
Key technical indicators used:
- **Moving Averages** (SMA, EMA)  
- **Relative Strength Index (RSI)**  
- **MACD (Moving Average Convergence Divergence)**  
- **Bollinger Bands**  

```python
from technical_indicators import add_indicators
data = add_indicators(nvda)
```

---
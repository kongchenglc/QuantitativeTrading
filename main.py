import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from util.pca import pca
from models.return_rate_predictor import StockReturnPredictor
from models.close_price_predictor import StockPricePredictor


def main():
    # Mac GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, defaulting to CPU.")

    # data = get_cleaned_data()
    data = pd.read_csv("data/cleaned_data.csv", index_col="Date")
    # data = pca(data)
    predictor = StockReturnPredictor(
        data,
        device,
        features=[
            "Low",
            "High",
            "Open",
            "Close",
            "EMA_10",
            "EMA_50",
            "SMA_10",
            "SMA_50",
            "BB_Mid",
            "Sentiment_Positive",
            "RSI_14",
            "MACD",
            "Volume",
            "Signal_Line",
            "Sentiment_Negative",
            "BB_Lower",
            "BB_Upper",
        ],
        **{
            "n_steps": 56,
            "hidden_size": 10,
            "num_layers": 1,
            "dropout": 0.18586952964756054,
            "lr": 0.0020896342748212677,
            "l1_weight_decay": 0.00017979197881835818,
            "l2_weight_decay": 4.293130104168741e-05,
        }
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    # predictor.trade_signal()


if __name__ == "__main__":
    main()

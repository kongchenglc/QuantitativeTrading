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
    predictor = StockPricePredictor(
        data,
        device,
        features=[
            "Low",
            "High",
            "EMA_50",
            "Open",
            "EMA_10",
            "SMA_10",
            "SMA_50",
            "BB_Mid",
            "Sentiment_Positive",
            "Close",
            "RSI_14",
            "MACD",
            "Volume",
            "Signal_Line",
            "Sentiment_Negative",
            "BB_Lower",
            "BB_Upper",
            "Weekday",
        ],
        **{
            "n_steps": 39,
            "lr": 0.001923083680975567,
            "patience": 22,
            "num_layers": 1,
            "hidden_size": 57,
            "dropout": 0.36156698327237025,
            "l1_weight_decay": 1.09858177977042e-05,
            "l2_weight_decay": 2.451001443916554e-05,
        },
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.trade_signal()


if __name__ == "__main__":
    main()

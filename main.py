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

    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv", index_col="Date")
    # data = pca(data)
    predictor = StockReturnPredictor(
        data,
        device,
        features=[
            "Close",
            "RSI_14",
            "Volume",
            "Sentiment_Negative",
            "MACD",
            "Open",
            "Signal_Line",
            "Sentiment_Positive",
            "Weekday",
            "EMA_50",
        ],
        **{
            "n_steps": 51,
            "hidden_size": 70,
            "num_layers": 1,
            "dropout": 0.2058109162360236,
            "lr": 0.006446231866757532,
            "l1_weight_decay": 0.0006212931938381016,
            "l2_weight_decay": 0.0006751080354646047,
        },
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    # predictor.trade_signal()


if __name__ == "__main__":
    main()

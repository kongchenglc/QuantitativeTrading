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
    data = pca(data)
    predictor = StockReturnPredictor(
        data,
        device,
        # features=[
        #     "Low",
        #     "High",
        #     "EMA_50",
        #     "Open",
        #     "EMA_10",
        #     "SMA_10",
        #     "SMA_50",
        #     "BB_Mid",
        #     "Sentiment_Positive",
        #     "Close",
        #     "RSI_14",
        #     "MACD",
        #     "Volume",
        #     "Signal_Line",
        #     "Sentiment_Negative",
        #     "BB_Lower",
        #     "BB_Upper",
        # ],
        **{
            "n_steps": 40,
            "hidden_size": 90,
            "num_layers": 1,
            "dropout": 0.4202435333188902,
            "lr": 0.0030516017985814593,
            "l1_weight_decay": 0.0004554026305197984,
            "l2_weight_decay": 2.7946097579790568e-05,
        }
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    # predictor.trade_signal()


if __name__ == "__main__":
    main()

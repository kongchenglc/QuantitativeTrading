import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data

from models.close_price_predictor import StockPricePredictor
from models.return_rate_predictor import StockReturnPredictor


def main():
    # Mac GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, defaulting to CPU.")

    # data = get_cleaned_data()
    data = pd.read_csv("data/cleaned_data.csv", index_col="Date")

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
        ],
        **{
            "n_steps": 70,
            "hidden_size": 30,
            "num_layers": 1,
            "dropout": 0.09831267302180555,
            "lr": 0.004798262999407178,
            "l2_weight_decay": 8.850780694683764e-07,
        }
    )

    # predictor = StockReturnPredictor(
    #     data,
    #     device,
    #     features=[
    #         "Close",
    #         "RSI_14",
    #         "Volume",
    #         "Open",
    #         "MACD",
    #         "Signal_Line",
    #         "BB_Lower",
    #         "Sentiment_Negative",
    #         "Sentiment_Neutral",
    #         "BB_Upper",
    #     ],
    #     **{
    #         "n_steps": 90,
    #         "hidden_size": 10,
    #         "num_layers": 3,
    #         "dropout": 0.27473146300173074,
    #         "lr": 0.0025990817107881307,
    #         "l2_weight_decay": 6.546354488797982e-05,
    #     }
    # )

    predictor.train()
    predictor.plot_results()
    predictor.backtest()
    predictor.predict_next_day()


if __name__ == "__main__":
    main()

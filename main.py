import torch
import pandas as pd
import numpy as np
from data_collection.clean_data import get_cleaned_data
from util.pca import pca
from models.close_price_predictor import StockPricePredictor

torch.manual_seed(42)
np.random.seed(42)


def main():

    # data = get_cleaned_data()
    data = pd.read_csv("data/cleaned_data.csv", index_col="Date")
    # data = pca(data)
    predictor = StockPricePredictor(
        data,
        features=[
            "Open",
            "Close",
            "Volume",
            "RSI_14",
            "EMA_50",
            "Sentiment_Negative",
            "Sentiment_Neutral",
            "Sentiment_Positive",
            "MACD",
            "Signal_Line",
            "Weekday",
        ],
        **{
            "n_steps": 127,
            "lr": 0.0007736014356373611,
            "patience": 38,
            "num_layers": 1,
            "hidden_size": 51,
            "dropout": 0.1523776191371796,
            "l1_weight_decay": 9.719495959920035e-08,
            "l2_weight_decay": 9.516139606710898e-07,
            "directional_weight": 0.16471075859414067,
        },
        transaction_fee=0.0,
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.predict_tomorrow_signal()


if __name__ == "__main__":
    main()

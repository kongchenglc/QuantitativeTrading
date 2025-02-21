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
            "n_steps": 6,
            "lr": 0.000924801815592734,
            "patience": 18,
            "num_layers": 3,
            "hidden_size": 100,
            "dropout": 0.3490065309761265,
            "l1_weight_decay": 9.716636735240595e-05,
            "l2_weight_decay": 8.014956268988414e-05,
        },
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()


if __name__ == "__main__":
    main()

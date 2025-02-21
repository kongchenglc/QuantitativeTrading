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
            "n_steps": 1,
            "lr": 0.003371794993230376,
            "patience": 78,
            "num_layers": 3,
            "hidden_size": 54,
            "dropout": 0.2015330730602193,
            "l1_weight_decay": 8.929352173716531e-07,
            "l2_weight_decay": 8.912680393271883e-08,
            "directional_weight": 0.9288828977846361,
        },
        transaction_fee=0.0,
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.predict_tomorrow_signal()


if __name__ == "__main__":
    main()

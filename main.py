import torch
import pandas as pd
import numpy as np
from data_collection.clean_data import get_cleaned_data
from util.pca import pca
from models.close_price_predictor import StockPricePredictor

torch.manual_seed(42)
np.random.seed(42)


def main():

    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv", index_col="Date")
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
            "n_steps": 12,
            "lr": 0.002213838769531753,
            "patience": 83,
            "num_layers": 3,
            "hidden_size": 125,
            "dropout": 0.13417513363743458,
            "l1_weight_decay": 4.7690766289656494e-08,
            "l2_weight_decay": 3.8888865586307615e-08,
            "directional_weight": 0.5150496912546632,
        },
        # transaction_fee=0.0,
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.predict_tomorrow_signal()


if __name__ == "__main__":
    main()

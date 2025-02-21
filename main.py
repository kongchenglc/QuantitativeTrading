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
            "n_steps": 45,
            "lr": 0.002689451684045019,
            "patience": 24,
            "num_layers": 1,
            "hidden_size": 13,
            "dropout": 0.3685227691893708,
            "l1_weight_decay": 4.1004497102613036e-05,
            "l2_weight_decay": 1.6973196937459353e-05,
            "directional_weight": 0.6086909549054128,
        },
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.predict_tomorrow_signal()


if __name__ == "__main__":
    main()

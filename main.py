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
            "n_steps": 6,
            "lr": 0.000755571383967672,
            "patience": 14,
            "num_layers": 2,
            "hidden_size": 87,
            "dropout": 0.2359330236387035,
            "l1_weight_decay": 5.305815744047952e-05,
            "l2_weight_decay": 9.789275455192073e-05,
            "directional_weight": 0.3475018811006327,
        },
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.predict_tomorrow_signal()


if __name__ == "__main__":
    main()

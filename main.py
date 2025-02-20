import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from util.pca import pca
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
    predictor = StockPricePredictor(
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
            "n_steps": 90,
            "hidden_size": 100,
            "num_layers": 1,
            "dropout": 0.2159311577299293,
            "lr": 0.005097796045513521,
            "l1_weight_decay": 8.017865606374595e-07,
            "l2_weight_decay": 0.0007019470271381507,
        }
    )

    predictor.train()
    predictor.plot_results()
    predictor.test()
    predictor.trade_signal()


if __name__ == "__main__":
    main()

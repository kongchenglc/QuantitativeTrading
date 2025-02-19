import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.close_price_predictor import StockPredictor

# from models.return_rate_predictor import StockPredictor


def main():
    # Mac GPU acceleration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, defaulting to CPU.")

    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv", index_col="Date")

    predictor = StockPredictor(
        data,
        device,
        **{
            "n_steps": 60,
            "hidden_size": 50,
            "num_layers": 1,
            "dropout": 0.07776739070270114,
            "lr": 0.00250387241170138,
            "l2_weight_decay": 1.1911996805587735e-06,
        }
    )
    predictor.train()
    predictor.plot_results()
    predictor.backtest()


if __name__ == "__main__":
    main()

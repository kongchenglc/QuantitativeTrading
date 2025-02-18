import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.close_price_predictor import StockPredictor
# from models.return_rate_predictor import StockPredictor

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, defaulting to CPU.")
    
    # data = get_cleaned_data()
    data = pd.read_csv("data/cleaned_data.csv")

    predictor = StockPredictor(data, device)
    predictor.train()
    predictor.plot_results()
    predictor.backtest()

if __name__ == "__main__":
    main()

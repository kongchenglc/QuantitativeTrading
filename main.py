import torch
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.lstm import StockPredictor

def main():
    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv")

    predictor = StockPredictor(data)
    predictor.train()
    predictor.plot_results()
    predictor.backtest()
    print(f"\nNext day prediction: {predictor.predict_next_close():.2f}")

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
    main()

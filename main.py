import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.lstm import StockPredictor

def main():
    
    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv")

    predictor = StockPredictor(data)

    predictor.train()

    predictor.plot_results()

if __name__ == "__main__":
    main()

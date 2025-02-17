import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.lstm import StockClosePredictor

def main():
    
    data = get_cleaned_data()
    # data = pd.read_csv("data/cleaned_data.csv")

    predictor = StockClosePredictor(data, n_steps=20, hidden_size=64, num_layers=2, lr=0.001, epochs=100)

    predictor.train()

    predictor.plot_results()

if __name__ == "__main__":
    main()

import torch
import numpy as np
import pandas as pd
from data_collection.clean_data import get_cleaned_data
from models.close_price_predictor import StockPricePredictor

torch.manual_seed(42)
np.random.seed(42)

# data = get_cleaned_data()
data = pd.read_csv("data/cleaned_data.csv", index_col="Date")
# data = pd.read_csv("data/test_data.csv", index_col="Date")

# Load the checkpoint (saved model)
pth_file_list = [
    "best_model_20250319_202207.pth",  # Training data end at 2024, transaction_fee=0.00, fixed position_size=2000
    # "best_model_20250319_145441.pth",  # Training data end at 2024, transaction_fee=0.01, fixed position_size=2000
    # "best_model_20250221_204147.pth",  # Training data end at 2025-02-20, transaction_fee=0.01, fixed position_size=all
    "best_model_20250221_151802.pth",  # Training data end at 2025-02-20, transaction_fee=0.0, position_size=all
]
checkpoint = torch.load(f"./models/best_model/{pth_file_list[0]}")  # 0 is newest

# Get the hyperparameters from the checkpoint
hyperparameters = checkpoint["hyperparameters"]

print("-------torch load pth file-------")
print(hyperparameters)

# Initialize the StockPricePredictor with the necessary hyperparameters
predictor = StockPricePredictor(
    data,
    transaction_fee=0.0,
    **hyperparameters,  # Use the features saved in the checkpoint
)

predictor.model.load_state_dict(checkpoint["model_state_dict"])

predictor.plot_results()
predictor.test()
predictor.predict_tomorrow_signal()

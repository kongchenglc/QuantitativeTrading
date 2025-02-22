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
    "best_model_20250221_151802.pth", # fit when transaction_fee=0.0 
    "best_model_20250221_132936.pth"
]
checkpoint = torch.load(f"./models/best_model/{pth_file_list[0]}")  # 0 is newest

print("-------torch load pth file-------")
print(checkpoint)

# Get the hyperparameters from the checkpoint
hyperparameters = checkpoint["hyperparameters"]

# Initialize the StockPricePredictor with the necessary hyperparameters
predictor = StockPricePredictor(
    data,
    # transaction_fee=0.0,
    features=hyperparameters["features"],  # Use the features saved in the checkpoint
    n_steps=hyperparameters["n_steps"],
    lr=hyperparameters["lr"],
    patience=hyperparameters["patience"],
    num_layers=hyperparameters["num_layers"],
    hidden_size=hyperparameters["hidden_size"],
    dropout=hyperparameters["dropout"],
    l1_weight_decay=hyperparameters["l1_weight_decay"],
    l2_weight_decay=hyperparameters["l2_weight_decay"],
)

predictor.model.load_state_dict(checkpoint["model_state_dict"])

predictor.plot_results()
predictor.test()
predictor.predict_tomorrow_signal()

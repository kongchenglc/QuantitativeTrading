import torch
import pandas as pd
from models.close_price_predictor import StockPricePredictor

# Load the checkpoint (saved model)
checkpoint = torch.load("./models/best_model/best_model_20250221_010203.pth")
data = pd.read_csv("data/cleaned_data.csv", index_col="Date")

print('-------checkpoint')
print(checkpoint)

# Get the hyperparameters from the checkpoint
hyperparameters = checkpoint["hyperparameters"]

# Initialize the StockPricePredictor with the necessary hyperparameters
predictor = StockPricePredictor(
    data, 
    features=hyperparameters["features"],  # Use the features saved in the checkpoint
    n_steps=hyperparameters["n_steps"],
    lr=hyperparameters["lr"],
    patience=hyperparameters["patience"],
    num_layers=hyperparameters["num_layers"],
    hidden_size=hyperparameters["hidden_size"],
    dropout=hyperparameters["dropout"],
    l1_weight_decay=hyperparameters["l1_weight_decay"],
    l2_weight_decay=hyperparameters["l2_weight_decay"]
)

predictor.model.load_state_dict(checkpoint["model_state_dict"])

predictor.plot_results()
predictor.test()


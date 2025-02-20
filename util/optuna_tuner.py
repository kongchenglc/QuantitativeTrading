import torch
import optuna
import pandas as pd
import numpy as np
from util.pca import pca
from models.close_price_predictor import StockPricePredictor

torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("MPS device is not available, defaulting to CPU.")

df = pd.read_csv("data/cleaned_data.csv")
df = pca(df)


def objective(trial):
    """Objective function: Use Optuna to automatically optimize LSTM hyperparameters"""
    # Let Optuna choose hyperparameters
    n_steps = trial.suggest_int("n_steps", 10, 90, step=10)
    hidden_size = trial.suggest_categorical("hidden_size", [10, 20, 30, 50, 70, 100])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    l1_weight_decay = trial.suggest_float("l1_weight_decay", 0.0, 1e-3)
    l2_weight_decay = trial.suggest_float("l2_weight_decay", 0.0, 1e-3)

    model = StockPricePredictor(
        df,
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
        n_steps=n_steps,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        l1_weight_decay=l1_weight_decay,
        l2_weight_decay=l2_weight_decay,
    )

    model.train()

    return model.evaluate(show_plot=False)  # The objective


# Run 50 trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=3600)

# Output the best parameters
print(f"Best parameters: {study.best_params}")

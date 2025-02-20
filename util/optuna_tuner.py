import torch
import optuna
import pandas as pd
import numpy as np
from util.pca import pca
from models.close_price_predictor import StockPricePredictor
from models.return_rate_predictor import StockReturnPredictor

torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("MPS device is not available, defaulting to CPU.")

df = pd.read_csv("data/cleaned_data.csv")
# df = pca(df)


def objective(trial):
    """Objective function: Use Optuna to automatically optimize LSTM hyperparameters"""
    # Let Optuna choose hyperparameters
    n_steps = trial.suggest_int("n_steps", 5, 60, step=5)
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.05, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    l1_weight_decay = trial.suggest_float("l1_weight_decay", 0.0, 1e-4)
    l2_weight_decay = trial.suggest_float("l2_weight_decay", 0.0, 1e-4)

    model = StockReturnPredictor(
        df,
        device,
        features=[
            "Close",
            "RSI_14",
            "Volume",
            "Sentiment_Negative",
            "MACD",
            "Open",
            "Signal_Line",
            "Sentiment_Positive",
            "Weekday",
            "EMA_50",
        ],
        n_steps=n_steps,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        l1_weight_decay=l1_weight_decay,
        l2_weight_decay=l2_weight_decay,
    )

    model.train()

    return model.test(show_plot=False)  # The objective


# Run 50 trials
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=3600 * 4)

# Output the best parameters
print(f"Best parameters: {study.best_params}")

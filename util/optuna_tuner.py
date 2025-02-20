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
    n_steps = trial.suggest_int("n_steps", 1, 60)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    patience = trial.suggest_int("patience", 3, 30)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 8, 512, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    l1_weight_decay = trial.suggest_float("l1_weight_decay", 0.0, 1e-4)
    l2_weight_decay = trial.suggest_float("l2_weight_decay", 0.0, 1e-4)

    model = StockPricePredictor(
        df,
        device,
        features=[
            "Low",
            "High",
            "EMA_50",
            "Open",
            "EMA_10",
            "SMA_10",
            "SMA_50",
            "BB_Mid",
            "Sentiment_Positive",
            "Close",
            "RSI_14",
            "MACD",
            "Volume",
            "Signal_Line",
            "Sentiment_Negative",
            "BB_Lower",
            "BB_Upper",
            "Weekday",
        ],
        n_steps=n_steps,
        lr=lr,
        patience=patience,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        l1_weight_decay=l1_weight_decay,
        l2_weight_decay=l2_weight_decay,
    )

    model.train()

    return model.test(show_plot=False)  # The objective


def callback(study, trial):
    best_params = study.best_params
    best_value = study.best_value
    print('------------')
    print(f"New best trial: {trial.number}")
    print(f"  Value: {best_value}")
    print(f"  Parameters: {best_params}")
    print('------------')


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=3600 * 2, callbacks=[callback])

# Output the best parameters
print(f"Best parameters: {study.best_params}")

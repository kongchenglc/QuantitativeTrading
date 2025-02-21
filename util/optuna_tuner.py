import torch
import optuna
import datetime
import pandas as pd
import numpy as np
from util.pca import pca
from models.close_price_predictor import StockPricePredictor

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv("data/cleaned_data.csv")
# df = pca(df)

start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
best_score = -float("inf")


def objective(trial):
    """Objective function: Use Optuna to automatically optimize LSTM hyperparameters"""

    global best_score

    # Let Optuna choose hyperparameters
    n_steps = trial.suggest_int("n_steps", 1, 60)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    patience = trial.suggest_int("patience", 3, 30)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 8, 125, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    l1_weight_decay = trial.suggest_float("l1_weight_decay", 0.0, 1e-4)
    l2_weight_decay = trial.suggest_float("l2_weight_decay", 0.0, 1e-4)
    directional_weight = trial.suggest_float("directional_weight", 0.0, 1.0)

    hyperparameters = {
        "n_steps": n_steps,
        "lr": lr,
        "patience": patience,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "l1_weight_decay": l1_weight_decay,
        "l2_weight_decay": l2_weight_decay,
        "directional_weight": directional_weight,
        "features": [
            "Open",
            "Close",
            "Volume",
            "RSI_14",
            "EMA_50",
            "Sentiment_Negative",
            "Sentiment_Neutral",
            "Sentiment_Positive",
            "MACD",
            "Signal_Line",
            "Weekday",
        ],
    }

    model = StockPricePredictor(
        df,
        **hyperparameters,
    )

    model.train()

    score = model.test(show_plot=False)  # The objective

    if score > best_score:
        best_score = score
        best_model_path = f"./models/best_model/best_model_{start_time}.pth"
        torch.save(
            {
                "model_state_dict": model.model.state_dict(),
                "hyperparameters": hyperparameters,
            },
            best_model_path,
        )
        print(f"New best model found! Score: {score}")
        print(f"Saving model to {best_model_path}")

    return score


def callback(study, trial):
    best_params = study.best_params
    best_value = study.best_value
    print("------------")
    print(f"Best trial: {study.best_trial}")
    print(f"Best Value: {best_value}")
    print(f"Best Parameters: {best_params}")
    print("------------")


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=3600 * 2, callbacks=[callback])

# Output the best parameters
print(f"Best parameters: {study.best_params}")

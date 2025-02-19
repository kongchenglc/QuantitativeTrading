import torch
import optuna
import pandas as pd
from models.close_price_predictor import StockPricePredictor
from models.return_rate_predictor import StockReturnPredictor


def objective(trial):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("MPS device is not available, defaulting to CPU.")

    df = pd.read_csv("data/cleaned_data.csv")

    """Objective function: Use Optuna to automatically optimize LSTM hyperparameters"""
    # Let Optuna choose hyperparameters
    n_steps = trial.suggest_int("n_steps", 10, 90, step=10)
    hidden_size = trial.suggest_categorical("hidden_size", [10, 20, 30, 50])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    l2_weight_decay = trial.suggest_float("l2_weight_decay", 1e-7, 1e-4)

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
            "BB_Lower",
            "BB_Upper",
            "BB_Mid",
        ],
        n_steps=n_steps,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        l2_weight_decay=l2_weight_decay,
    )

    # model = StockReturnPredictor(
    #     df,
    #     device,
    #     features=[
    #         "Close",
    #         "RSI_14",
    #         "Volume",
    #         "Open",
    #         "MACD",
    #         "Signal_Line",
    #         "BB_Lower",
    #         "Sentiment_Negative",
    #         "Sentiment_Neutral",
    #         "BB_Upper",
    #     ],
    #     n_steps=n_steps,
    #     hidden_size=hidden_size,
    #     num_layers=num_layers,
    #     dropout=dropout,
    #     lr=lr,
    #     l2_weight_decay=l2_weight_decay,
    # )

    model.train()

    # Calculate validation loss
    loss = model.backtest(show_plot=False)
    return loss  # The objective is to minimize the loss


# Run 50 trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Output the best parameters
print(f"Best parameters: {study.best_params}")

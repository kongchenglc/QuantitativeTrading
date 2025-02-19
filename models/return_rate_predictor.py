import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTMModel(nn.Module):
    """LSTM model for predicting stock returns"""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockReturnPredictor:
    """Handles data processing, training, prediction, and visualization"""

    def __init__(
        self,
        df,
        device,
        epochs=1500,
        patience=30,
        lr=0.0001,
        n_steps=10,
        hidden_size=50,
        num_layers=3,
        dropout=0,
        l2_weight_decay=0,
        test_ratio=0.1,
        features=None,  # Add features as a parameter
    ):
        """
        Initialize the stock predictor model with provided hyperparameters.

        :param df: Input DataFrame with stock data (requires 'Close' column).
        :param device: Device (either "mps" or "cpu").
        :param n_steps: Number of previous time steps to use for prediction.
        :param hidden_size: Number of hidden units in the LSTM layer.
        :param num_layers: Number of layers in the LSTM.
        :param lr: Learning rate for the optimizer.
        :param epochs: Number of epochs for training.
        :param test_ratio: The proportion of data to reserve for testing.
        :param dropout: Dropout rate for LSTM layers.
        :param l2_weight_decay: Strength of L2 regularization.
        :param patience: Number of epochs to wait before stopping if no improvement.
        :param features: List of columns to use as features (optional).
        """
        self.df = df.copy()
        self.device = device
        self.n_steps = n_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.dropout = dropout
        self.l2_weight_decay = l2_weight_decay
        self.patience = patience

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.result_scaler = MinMaxScaler(feature_range=(-1, 1))

        # Use provided 'features' or default to all columns except 'Date' and 'index'
        if features is None:
            features = [col for col in self.df.columns if col not in ["index", "Date"]]
        self.features = features

        self.X_train, self.y_train, self.X_test, self.y_test, self.feature_size = (
            self.prepare_data()
        )
        print("Feature size:", self.feature_size)
        self.initialize_model()

    def prepare_data(self):
        """Prepare training and testing data with the selected features"""
        data = self.df.dropna()  # Remove rows with NaN values

        # Use the 'features' attribute to select columns
        feature_columns = self.features

        # Split data into train and test segments
        train_size = int(len(data) * (1 - self.test_ratio))
        test_start_idx = train_size - self.n_steps
        train_data = data[:train_size]
        test_data = data[test_start_idx:]

        # Normalize features based on training data
        self.scaler.fit(train_data[feature_columns].values)
        train_data_scaled = self.scaler.transform(train_data[feature_columns].values)
        test_data_scaled = self.scaler.transform(test_data[feature_columns].values)

        self.result_scaler.fit(train_data["Return"].values.reshape(-1, 1))
        train_return_scaled = self.result_scaler.transform(
            train_data["Return"].values.reshape(-1, 1)
        )
        test_return_scaled = self.result_scaler.transform(
            test_data["Return"].values.reshape(-1, 1)
        )

        def create_sequences(scaled_data, target_data):
            X, y = [], []
            for i in range(len(scaled_data) - self.n_steps - 1):
                X.append(scaled_data[i : i + self.n_steps])
                y.append(target_data[i + self.n_steps])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(
            train_data_scaled, train_return_scaled.flatten()
        )
        X_test, y_test = create_sequences(
            test_data_scaled, test_return_scaled.flatten()
        )

        return (
            torch.tensor(X_train, dtype=torch.float32).to(self.device),
            torch.tensor(y_train, dtype=torch.float32).to(self.device),
            torch.tensor(X_test, dtype=torch.float32).to(self.device),
            torch.tensor(y_test, dtype=torch.float32).to(self.device),
            X_train.shape[2],
        )

    def initialize_model(self):
        """Initialize the LSTM model"""
        self.model = LSTMModel(
            self.feature_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )

    def train(self):
        """Train the model with early stopping"""
        print("Training model...")

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train.unsqueeze(1))

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

            # Early stopping mechanism
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def backtest(self, show_plot=True):
        """Perform backtesting on the test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred_return = self.model(self.X_test).cpu().numpy().flatten()
            predicted_test_return_original = self.result_scaler.inverse_transform(
                test_pred_return.reshape(-1, 1)
            ).flatten()

            actual_test_close = (
                self.df["Close"].iloc[len(self.df) - len(self.y_test) :].values
            )
            actual_test_return = (
                self.df["Return"].iloc[len(self.df) - len(self.y_test) :].values
            )

            predicted_test_close = actual_test_close[:-1] * (
                1 + predicted_test_return_original[1:]
            )

            # Compute backtest metrics
            mse = mean_squared_error(actual_test_close[1:], predicted_test_close)
            mae = mean_absolute_error(actual_test_close[1:], predicted_test_close)
            rmse = np.sqrt(mse)

            print("\nBacktest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            if show_plot:
                # Create subplots for returns and price
                fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                # Plot actual vs. predicted returns on the top subplot
                axs[0].plot(actual_test_return, label="Actual Return", color="blue")
                axs[0].plot(
                    predicted_test_return_original,
                    label="Predicted Return",
                    color="red",
                    linestyle="--",
                )
                axs[0].axhline(
                    0,
                    color="black",
                    linestyle="-",
                    linewidth=1,
                    label="Baseline (Return = 0)",
                )
                axs[0].set_title("Backtest Results: Actual vs Predicted Returns")
                axs[0].set_ylabel("Return")
                axs[0].legend(loc="upper left")

                # Plot actual vs. predicted close prices on the bottom subplot
                axs[1].plot(actual_test_close[1:], label="Actual Close", color="blue")
                axs[1].plot(
                    predicted_test_close,
                    label="Predicted Close",
                    color="red",
                    linestyle="--",
                )
                axs[1].set_title("Backtest Results: Actual vs Predicted Close Prices")
                axs[1].set_xlabel("Time Steps")
                axs[1].set_ylabel("Price")
                axs[1].legend(loc="upper left")

                # Show the plots
                plt.tight_layout()
                plt.show()

            return rmse

    def plot_results(self):
        """Plot training results with predicted and actual close prices and returns"""
        self.model.eval()
        with torch.no_grad():
            train_pred_return = self.model(self.X_train).cpu().numpy().flatten()
            predicted_train_return_original = self.result_scaler.inverse_transform(
                train_pred_return.reshape(-1, 1)
            ).flatten()

            actual_train_close = self.df["Close"].iloc[: len(self.X_train)].values
            actual_train_return = self.df["Return"].iloc[: len(self.X_train)].values

            predicted_train_close = actual_train_close[:-1] * (
                1 + predicted_train_return_original[1:]
            )

            # Create subplots for returns and price
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Plot actual vs. predicted returns on the top subplot
            axs[0].plot(actual_train_return, label="Actual Return", color="blue")
            axs[0].plot(
                predicted_train_return_original,
                label="Predicted Return",
                color="red",
                linestyle="--",
            )
            axs[0].axhline(
                0,
                color="black",
                linestyle="-",
                linewidth=1,
                label="Baseline (Return = 0)",
            )
            axs[0].set_title("Training Results: Actual vs Predicted Returns")
            axs[0].set_ylabel("Return")
            axs[0].legend(loc="upper left")

            # Plot actual vs. predicted close prices on the bottom subplot
            axs[1].plot(actual_train_close, label="Actual Close", color="blue")
            axs[1].plot(
                np.concatenate(([actual_train_close[0]], predicted_train_close)),
                label="Predicted Close",
                color="orange",
                linestyle="--",
            )
            axs[1].set_title("Training Results: Actual vs Predicted Close Prices")
            axs[1].set_xlabel("Time Steps")
            axs[1].set_ylabel("Price")
            axs[1].legend(loc="upper left")

            # Show the plots
            plt.tight_layout()
            plt.show()

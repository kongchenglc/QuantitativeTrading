import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTMModel(nn.Module):
    """LSTM model for predicting Close price"""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Fully connected layer for the output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockPricePredictor:
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
        features=None,  # New parameter to select features
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
        :param patience: Patience for early stopping, in terms of epochs without improvement.
        :param l2_weight_decay: Strength of L2 regularization.
        :param features: List of features to use for the model, if None, use all columns except 'Date' and 'index'
        """
        self.df = df.copy()
        self.device = device  # Store the device for later use
        self.n_steps = n_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.patience = patience
        self.dropout = dropout
        self.l2_weight_decay = l2_weight_decay
        self.features = (
            features
            if features is not None
            else [col for col in self.df.columns if col not in ["index", "Date"]]
        )  # Default to all columns except 'Date' and 'index'

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.result_scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train, self.X_test, self.y_test, self.feature_size = (
            self.prepare_data()
        )
        self.initialize_model()

    def prepare_data(self):
        """Prepare training and testing data with time series split"""
        data = self.df.dropna()  # Remove rows with NaN values

        # Use the features provided (self.features)
        feature_columns = self.features

        # Split data into train and test segments
        train_size = int(len(data) * (1 - self.test_ratio))
        test_start_idx = train_size - self.n_steps  # Maintain window size for test data
        train_data = data[:train_size]
        test_data = data[test_start_idx:]  # Include last n_steps from training

        # Fit scaler only on training data (exclude non-numeric columns)
        self.scaler.fit(
            train_data[feature_columns].values
        )  # Fit scaler only on numeric columns
        train_data_scaled = self.scaler.transform(train_data[feature_columns].values)
        test_data_scaled = self.scaler.transform(test_data[feature_columns].values)

        self.result_scaler.fit(train_data["Close"].values.reshape(-1, 1))
        train_return_scaled = self.result_scaler.transform(
            train_data["Close"].values.reshape(-1, 1)
        )
        test_return_scaled = self.result_scaler.transform(
            test_data["Close"].values.reshape(-1, 1)
        )

        def create_sequences(scaled_data, target_data):
            X, y = [], []
            for i in range(len(scaled_data) - self.n_steps - 1):
                X.append(scaled_data[i : i + self.n_steps])
                y.append(target_data[i + self.n_steps])
            return np.array(X), np.array(y)

        # Create sequences
        X_train, y_train = create_sequences(train_data_scaled, train_return_scaled)
        X_test, y_test = create_sequences(test_data_scaled, test_return_scaled)

        # Move data to device
        return (
            torch.tensor(X_train, dtype=torch.float32).to(self.device),
            torch.tensor(y_train, dtype=torch.float32).to(self.device),
            torch.tensor(X_test, dtype=torch.float32).to(self.device),
            torch.tensor(y_test, dtype=torch.float32).to(self.device),
            X_train.shape[2],
        )

    def initialize_model(self):
        """Initialize model components"""
        self.model = LSTMModel(
            self.feature_size, self.hidden_size, self.num_layers, self.dropout
        ).to(
            self.device
        )  # Move model to device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay
        )  # L2 Regularization

    def train(self):
        print("Training model...")
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

            # Early stopping logic
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping at epoch {epoch+1} due to no improvement in loss"
                )
                break

    def backtest(self, show_plot=True):
        """Perform backtesting on test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred = (
                self.model(self.X_test).cpu().numpy().flatten()
            )  # Move to CPU for plotting

            # Inverse scaling
            test_pred = self.result_scaler.inverse_transform(
                test_pred.reshape(-1, 1)
            ).flatten()
            actual_test_close = (
                self.df["Close"].iloc[len(self.df) - len(self.y_test) :].values
            )

            # Calculate metrics
            mse = mean_squared_error(actual_test_close, test_pred)
            mae = mean_absolute_error(actual_test_close, test_pred)
            rmse = np.sqrt(mse)

            print("\nBacktest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            if show_plot:
                # Plot results
                plt.figure(figsize=(12, 6))
                plt.plot(actual_test_close, label="Actual Close", color="blue")
                plt.plot(
                    test_pred, label="Predicted Close", color="red", linestyle="--"
                )
                # Add grid, display ticks by day
                plt.xticks(range(0, len(test_pred), 1))  # Display tick for each day
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
                plt.title("Backtest Results")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.legend()
                plt.show()

        return rmse

    def predict_next_close(self):
        self.model.eval()
        last_n_days = self.df[self.features].values[-self.n_steps :]
        last_n_days_scaled = self.scaler.transform(last_n_days)
        input_tensor = (
            torch.tensor(last_n_days_scaled, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            predicted_scaled = self.model(input_tensor).item()
        return self.result_scaler.inverse_transform([[predicted_scaled]])[0][0]

    def plot_results(self):
        """Plot training results"""
        self.model.eval()
        with torch.no_grad():
            train_pred = (
                self.model(self.X_train).cpu().numpy().flatten()
            )  # Move to CPU for plotting

            # Inverse scaling
            train_pred_original = self.result_scaler.inverse_transform(
                train_pred.reshape(-1, 1)
            ).flatten()
            actual_train_close = self.df["Close"].iloc[: len(self.X_train)].values

            plt.figure(figsize=(12, 6))
            plt.plot(actual_train_close, label="Actual Close", color="blue")
            plt.plot(
                train_pred_original,
                label="Predicted Close",
                color="orange",
                linestyle="--",
            )
            # Add grid, display ticks by day
            plt.xticks(
                range(0, len(train_pred_original), 1)
            )  # Display tick for each day
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            plt.title("Training Results")
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

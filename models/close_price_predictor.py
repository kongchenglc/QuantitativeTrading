import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
        epochs=3000,
        patience=30,
        lr=0.0001,
        n_steps=10,
        hidden_size=50,
        num_layers=3,
        dropout=0,
        l2_weight_decay=0,
        test_ratio=0.2,
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

    def test(self, show_plot=True):
        """Perform testing on test set"""
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

    def evaluate(self, show_plot=True):
        self.model.eval()
        with torch.no_grad():
            # Predict on the validation set
            valid_pred = self.model(self.X_test).cpu().numpy().flatten()
            valid_pred = self.result_scaler.inverse_transform(
                valid_pred.reshape(-1, 1)
            ).flatten()
            # Assume the actual validation values are in the last segment of df["Close"]
            actual_valid_close = self.df["Close"].iloc[-len(valid_pred) :].values

        # Calculate the RMSE for the validation set
        valid_rmse = np.sqrt(mean_squared_error(actual_valid_close, valid_pred))

        print("Validation RMSE: {:.4f}".format(valid_rmse))

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(actual_valid_close, label="Validation Actual", color="blue")
            plt.plot(
                valid_pred, label="Validation Predicted", color="red", linestyle="--"
            )
            plt.title("Validation Predictions")
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

        return valid_rmse

    # WIP
    def trade_signal(self, show_plot=True):
        """Perform trading simulation based on predicted price trends"""
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(self.X_test).cpu().numpy().flatten()
            test_pred = self.result_scaler.inverse_transform(
                test_pred.reshape(-1, 1)
            ).flatten()
            actual_close = self.df["Close"].iloc[-len(test_pred) :].values
            dates = self.df.index[-len(test_pred) :]

        # Create trading dataframe
        trade_df = pd.DataFrame(
            {"date": dates, "close": actual_close, "pred": test_pred}
        ).set_index("date")

        # Generate trading signals based on trend detection
        # If predicted price trend is upward (price is increasing), signal buy (1)
        # If predicted price trend is downward (price is decreasing), signal sell (-1)
        trade_df["trend"] = (
            trade_df["pred"].diff().shift(-1)
        )  # Calculate trend based on diff
        trade_df["signal"] = np.where(trade_df["trend"] > 0, 1, -1)

        # Trading parameters
        initial_cash = 10000.0
        cash = initial_cash
        shares = 0
        position = 0  # 0: no position, 1: long position
        trade_df = trade_df.assign(action="hold", portfolio_value=initial_cash)

        # Simulate trading based on predicted trend
        for i in range(1, len(trade_df)):
            current_price = trade_df["close"].iloc[i]
            prev_signal = trade_df["signal"].iloc[i - 1]

            # Update portfolio value
            trade_df.iloc[i, trade_df.columns.get_loc("portfolio_value")] = (
                cash + shares * current_price
            )

            # Execute trading logic based on predicted trend
            if prev_signal == 1 and position == 0:  # Buy signal
                max_shares = (cash * 0.99) / current_price  # 1% transaction fee
                shares = max_shares
                cash = 0
                position = 1
                trade_df.iloc[i, trade_df.columns.get_loc("action")] = "buy"

            elif prev_signal == -1 and position == 1:  # Sell signal
                cash = shares * current_price * 0.99  # 1% transaction fee
                shares = 0
                position = 0
                trade_df.iloc[i, trade_df.columns.get_loc("action")] = "sell"

            # Update portfolio value after transaction
            trade_df.iloc[i, trade_df.columns.get_loc("portfolio_value")] = (
                cash + shares * current_price
            )

        # Force liquidation at last day
        if shares > 0:
            cash = shares * trade_df["close"].iloc[-1] * 0.99
            trade_df.iloc[-1, trade_df.columns.get_loc("portfolio_value")] = cash
            trade_df.iloc[-1, trade_df.columns.get_loc("action")] = "sell"

        # Calculate metrics
        mse = mean_squared_error(trade_df["close"], trade_df["pred"])
        mae = mean_absolute_error(trade_df["close"], trade_df["pred"])
        final_value = trade_df["portfolio_value"].iloc[-1]
        returns = (final_value - initial_cash) / initial_cash

        print("\nComprehensive Evaluation Results:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"Initial Capital: ${initial_cash:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Return: {returns*100:.2f}%")

        # Visualization
        if show_plot:
            plt.figure(figsize=(14, 10))

            # Price and signals plot
            ax1 = plt.subplot(2, 1, 1)
            plt.plot(trade_df["close"], label="Actual Price", zorder=1)
            plt.plot(
                trade_df["pred"], label="Predicted Price", linestyle="--", zorder=1
            )
            plt.scatter(
                trade_df[trade_df["action"] == "buy"].index,
                trade_df[trade_df["action"] == "buy"]["close"],
                marker="^",
                color="g",
                s=100,
                label="Buy",
                zorder=2,
            )
            plt.scatter(
                trade_df[trade_df["action"] == "sell"].index,
                trade_df[trade_df["action"] == "sell"]["close"],
                marker="v",
                color="r",
                s=100,
                label="Sell",
                zorder=2,
            )
            plt.title("Price and Trading Signals")
            plt.legend()

            # Portfolio value plot
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            plt.plot(
                trade_df["portfolio_value"], label="Portfolio Value", color="purple"
            )
            plt.axhline(
                initial_cash, color="gray", linestyle="--", label="Initial Capital"
            )
            plt.fill_between(
                trade_df.index,
                trade_df["portfolio_value"],
                initial_cash,
                where=(trade_df["portfolio_value"] > initial_cash),
                facecolor="green",
                alpha=0.3,
            )
            plt.fill_between(
                trade_df.index,
                trade_df["portfolio_value"],
                initial_cash,
                where=(trade_df["portfolio_value"] < initial_cash),
                facecolor="red",
                alpha=0.3,
            )
            plt.title(f"Portfolio Value (Final: ${final_value:.2f})")
            plt.legend()

            plt.tight_layout()
            plt.show()

        return final_value

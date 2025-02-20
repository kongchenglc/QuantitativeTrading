import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
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
        epochs=2000,
        patience=10,
        lr=0.0001,
        n_steps=10,
        hidden_size=50,
        num_layers=3,
        dropout=0,
        l2_weight_decay=0,
        l1_weight_decay=0,  # New parameter for L1 regularization
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
        :param l1_weight_decay: Strength of L1 regularization.
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
        self.l1_weight_decay = l1_weight_decay  # Store L1 weight decay
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

    def l1_regularization(self):
        """Calculate L1 regularization term"""
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        return self.l1_weight_decay * l1_norm

    def train(self):
        """Train the model with early stopping based on validation loss"""
        print("Training model...")

        # Split training data into training and validation sets (10% for validation)
        val_ratio = 0.1
        val_size = int(len(self.X_train) * val_ratio)

        X_val, y_val = (
            self.X_train[-val_size:],
            self.y_train[-val_size:],
        )  # Use the last val_size samples as validation set
        X_train, y_train = (
            self.X_train[:-val_size],
            self.y_train[:-val_size],
        )  # Use the remaining samples as training set

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            self.optimizer.zero_grad()

            # Forward pass for training
            output = self.model(X_train)
            train_loss = self.criterion(output, y_train)

            # Add L1 regularization if weight decay is applied
            if self.l1_weight_decay > 0:
                train_loss += self.l1_regularization()

            train_loss.backward()
            self.optimizer.step()

            # Validation loss calculation
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_output = self.model(X_val)
                val_loss = self.criterion(val_output, y_val)

            # Print losses every 10 epochs
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}"
                )

            # Early stopping based on validation loss
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0  # Reset patience counter on improvement
            else:
                patience_counter += 1

            # Stop early if no improvement for 'patience' epochs
            if patience_counter >= self.patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}, best validation loss: {best_val_loss:.6f}"
                )
                break

    def plot_results(self):
        """Plot actual vs. predicted stock prices"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_train).cpu().numpy()
            actual_values = self.y_train.cpu().numpy()

            predictions = self.result_scaler.inverse_transform(predictions).flatten()
            actual_values = self.result_scaler.inverse_transform(
                actual_values
            ).flatten()

        plt.figure(figsize=(12, 6))
        plt.plot(
            range(len(actual_values)),
            actual_values,
            label="Actual Prices",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            range(len(predictions)),
            predictions,
            label="Predicted Prices",
            color="red",
            linestyle="dashed",
            linewidth=2,
        )
        plt.xticks(range(0, len(predictions), 1))
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.title("Train: Actual vs Predicted")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

    def test(self, show_plot=True):
        """Perform testing on test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(self.X_test).cpu().numpy()
            actual_test_close = self.y_test.cpu().numpy()

            test_pred = self.result_scaler.inverse_transform(test_pred).flatten()
            actual_test_close = self.result_scaler.inverse_transform(
                actual_test_close
            ).flatten()

            test_indices = range(len(test_pred))

            if show_plot:
                plt.figure(figsize=(12, 6))
                plt.plot(
                    test_indices, actual_test_close, label="Actual Close", color="blue"
                )
                plt.plot(
                    test_indices,
                    test_pred,
                    label="Predicted Close",
                    color="red",
                    linestyle="--",
                )
                plt.xticks(range(0, len(test_pred), 1))
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
                plt.title("Test Results")
                plt.xlabel("Time Steps")
                plt.ylabel("Price")
                plt.legend()
                plt.show()

        mse = mean_squared_error(actual_test_close, test_pred)
        mae = mean_absolute_error(actual_test_close, test_pred)
        rmse = np.sqrt(mse)

        # Direction Accuracy
        actual_direction = np.sign(
            np.diff(actual_test_close)
        )  # Calculate actual price direction
        predicted_direction = np.sign(
            np.diff(test_pred)
        )  # Calculate predicted price direction
        direction_accuracy = np.mean(
            actual_direction == predicted_direction
        )  # Compare directions

        # R-squared
        r2 = r2_score(actual_test_close, test_pred)

        print("\nTest Results:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"Direction Accuracy: {direction_accuracy:.4f}")
        print(f"R-squared: {r2:.4f}")

        # Calculate the overall score with weights (adjust weights as needed)
        w1, w2, w3, w4 = (
            0.2,
            0.3,
            0.3,
            0.2,
        )  # You can adjust these weights based on your priorities

        normalized_rmse = 1 / (1 + rmse)
        normalized_mse = 1 / (1 + mse)
        normalized_mae = 1 / (1 + mae)

        # Overall score calculation
        score = (
            w1 * normalized_rmse
            + w2 * direction_accuracy
            + w3 * max(0, r2)
            + w4 * (normalized_mse + normalized_mae)
        )

        print(f"Overall Model Score: {score:.4f}")

        return_rate = max(0, self.trade_signal(show_plot=False))
        print(f"Return rate: {return_rate:.4f}")
        print(f"Overall Score: {score:.4f}")
        print(f"Return rate * Overall Score: {score * return_rate:.4f}")
        return score * return_rate

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
            actual_open = self.df["Open"].iloc[-len(test_pred) :].values
            dates = self.df.index[-len(test_pred) :]

        # Create trading dataframe
        trade_df = pd.DataFrame(
            {
                "date": dates,
                "close": actual_close,
                "open": actual_open,
                "pred": test_pred,
            }
        ).set_index("date")

        # Generate trading signals
        trade_df["trend"] = trade_df[
            "pred"
        ].diff()  # Current prediction - previous prediction
        trade_df["signal"] = np.where(trade_df["trend"] > 0, 1, -1)

        # Trading parameters
        initial_capital = 10000.0
        cash = initial_capital
        shares = 0
        position = 0  # 0: no position, 1: long position
        trade_df = trade_df.assign(
            action="hold",
            portfolio_value=initial_capital,
            drawdown=0.0,
            trade_price=np.nan,
        )

        # Risk management parameters
        stop_loss_pct = 0.95  # 5% stop loss
        position_size = 0.5  # 50% of capital per trade

        # Trading simulation
        for i in range(1, len(trade_df)):
            current_open = trade_df["open"].iloc[i]
            prev_signal = trade_df["signal"].iloc[i - 1]

            # Update portfolio value
            current_value = cash + shares * trade_df["close"].iloc[i]
            trade_df.iloc[i, trade_df.columns.get_loc("portfolio_value")] = (
                current_value
            )

            # Stop-loss check
            if position == 1:
                entry_price = trade_df.at[trade_df.index[i - 1], "trade_price"]
                if current_open < entry_price * stop_loss_pct:
                    prev_signal = -1  # Trigger stop loss

            # Trading logic
            if prev_signal == 1 and position == 0:  # Buy signal
                max_shares = (cash * position_size * 0.99) / current_open  # 1% fee
                if max_shares > 0:
                    shares += max_shares
                    cash -= max_shares * current_open
                    position = 1
                    trade_df.iloc[i, trade_df.columns.get_loc("action")] = "buy"
                    trade_df.iloc[i, trade_df.columns.get_loc("trade_price")] = (
                        current_open
                    )

            elif prev_signal == -1 and position == 1:  # Sell signal
                sell_value = shares * current_open * 0.99  # 1% fee
                cash += sell_value
                shares = 0
                position = 0
                trade_df.iloc[i, trade_df.columns.get_loc("action")] = "sell"
                trade_df.iloc[i, trade_df.columns.get_loc("trade_price")] = current_open

        # Force liquidation on last day
        if shares > 0:
            cash += shares * trade_df["open"].iloc[-1] * 0.99
            trade_df.iloc[-1, trade_df.columns.get_loc("portfolio_value")] = cash
            trade_df.iloc[-1, trade_df.columns.get_loc("action")] = "sell"

        # Performance metrics
        returns = (cash - initial_capital) / initial_capital

        # Calculate max drawdown
        peak_values = trade_df["portfolio_value"].cummax()
        drawdowns = (peak_values - trade_df["portfolio_value"]) / peak_values
        max_drawdown = drawdowns.max()

        # Sharpe ratio (assuming 3% risk-free rate)
        excess_returns = trade_df["portfolio_value"].pct_change().dropna() - 0.03 / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Trade statistics
        trade_actions = trade_df[trade_df["action"].isin(["buy", "sell"])]
        win_trades = trade_actions[trade_actions["portfolio_value"].diff() > 0]
        win_rate = len(win_trades) / len(trade_actions) if len(trade_actions) > 0 else 0

        print("\nPerformance Metrics:")
        print(f"MSE: {mean_squared_error(trade_df['close'], trade_df['pred']):.4f}")
        print(f"MAE: {mean_absolute_error(trade_df['close'], trade_df['pred']):.4f}")
        print(f"Initial Capital: ${initial_capital:.2f}")
        print(f"Final Value: ${cash:.2f}")
        print(f"Return: {returns*100:.2f}%")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Total Trades: {len(trade_actions)}")

        # Visualization
        if show_plot:
            plt.figure(figsize=(12, 6))

            # Price and signals plot
            ax1 = plt.subplot(3, 1, 1)
            plt.plot(trade_df["close"], label="Actual Price", alpha=0.7)
            plt.plot(
                trade_df["pred"], label="Predicted Price", linestyle="--", alpha=0.7
            )
            buy_signals = trade_df[trade_df["action"] == "buy"]
            sell_signals = trade_df[trade_df["action"] == "sell"]
            plt.scatter(
                buy_signals.index,
                buy_signals["open"],
                marker="^",
                color="g",
                s=100,
                label="Buy",
                zorder=3,
            )
            plt.scatter(
                sell_signals.index,
                sell_signals["open"],
                marker="v",
                color="r",
                s=100,
                label="Sell",
                zorder=3,
            )
            plt.title("Price and Trading Signals")
            plt.legend()

            # Portfolio value plot
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            plt.plot(trade_df["portfolio_value"], label="Portfolio", color="purple")
            plt.fill_between(
                trade_df.index,
                trade_df["portfolio_value"],
                initial_capital,
                where=(trade_df["portfolio_value"] > initial_capital),
                facecolor="green",
                alpha=0.3,
                label="Profit Zone",
            )
            plt.fill_between(
                trade_df.index,
                trade_df["portfolio_value"],
                initial_capital,
                where=(trade_df["portfolio_value"] < initial_capital),
                facecolor="red",
                alpha=0.3,
                label="Loss Zone",
            )
            plt.plot(peak_values, linestyle="--", color="darkgreen", label="Peak Value")
            plt.title(f"Portfolio Value (Final: ${cash:.2f})")
            plt.legend()

            # Drawdown plot
            ax3 = plt.subplot(3, 1, 3)
            plt.fill_between(
                trade_df.index, drawdowns * 100, 0, facecolor="red", alpha=0.3
            )
            plt.title(f"Drawdown (Max: {max_drawdown*100:.2f}%)")
            plt.ylabel("Drawdown (%)")

            plt.tight_layout()
            plt.show()

        return returns

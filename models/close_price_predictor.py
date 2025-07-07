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
        epochs=5000,
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
        transaction_fee=0.01,
        directional_weight=0.5,
        take_profit=0.2,
    ):
        """
        Initialize the stock predictor model with provided hyperparameters.

        :param df: Input DataFrame with stock data (requires 'Close' column).
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
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
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
        self.transaction_fee = transaction_fee
        self.directional_weight = directional_weight
        self.take_profit = take_profit
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
        """Train the model with early stopping based on validation loss using returns"""
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
            output = self.model(
                X_train
            )  # Predicted prices (scaled), shape: (batch_size, 1)

            pred_prices = output  # Shape: (batch_size, 1)
            pred_returns = (pred_prices[1:] - pred_prices[:-1]) / (
                pred_prices[:-1] + 1e-8
            )

            actual_returns = (y_train[1:] - y_train[:-1]) / (y_train[:-1] + 1e-8)

            # MSE Loss on returns
            train_loss = self.criterion(pred_returns, actual_returns)

            # Directional Accuracy Loss
            directional_loss = torch.mean(
                (torch.sign(pred_returns) != torch.sign(actual_returns)).float()
            )
            train_loss += self.directional_weight * directional_loss

            # L1 regularization
            if self.l1_weight_decay > 0:
                train_loss += self.l1_regularization()

            train_loss.backward()
            self.optimizer.step()

            # Validation loss calculation
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                val_output = self.model(X_val)
                val_pred_returns = (val_output[1:] - val_output[:-1]) / (
                    val_output[:-1] + 1e-8
                )
                val_actual_returns = (y_val[1:] - y_val[:-1]) / (y_val[:-1] + 1e-8)
                val_loss = self.criterion(val_pred_returns, val_actual_returns)

                # Directional Accuracy Loss
                val_directional_loss = torch.mean(
                    (
                        torch.sign(val_pred_returns) != torch.sign(val_actual_returns)
                    ).float()
                )
                val_loss += self.directional_weight * val_directional_loss

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
        ax1 = plt.gca()
        ax1.plot(
            range(len(actual_values)),
            actual_values,
            label="Actual Prices",
            color="blue",
            linewidth=2,
        )
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Actual Stock Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2 = ax1.twinx()
        ax2.plot(
            range(len(predictions)),
            predictions,
            label="Predicted Prices",
            color="red",
            linestyle="dashed",
            linewidth=2,
        )
        ax2.set_ylabel("Predicted Stock Price", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax1.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.xticks(range(0, len(predictions), 1))
        plt.title("Train: Actual vs Predicted")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

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

            # if show_plot:
            #     plt.figure(figsize=(12, 6))

            #     ax1 = plt.gca()
            #     ax1.plot(
            #         test_indices,
            #         actual_test_close,
            #         label="Actual Close",
            #         color="blue",
            #         linewidth=2,
            #     )
            #     ax1.set_xlabel("Time Steps")
            #     ax1.set_ylabel("Actual Close Price", color="blue")
            #     ax1.tick_params(axis="y", labelcolor="blue")

            #     ax2 = ax1.twinx()
            #     ax2.plot(
            #         test_indices,
            #         test_pred,
            #         label="Predicted Close",
            #         color="red",
            #         linestyle="--",
            #         linewidth=2,
            #     )
            #     ax2.set_ylabel("Predicted Close Price", color="red")
            #     ax2.tick_params(axis="y", labelcolor="red")

            #     ax1.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

            #     plt.xticks(range(0, len(test_pred), 1))

            #     plt.title("Test Results")
            #     plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

            #     ax1.legend(loc="upper left")
            #     ax2.legend(loc="upper right")

            #     plt.show()

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

        test_returns = max(0, self.trade_on_test(show_plot=show_plot))
        backtest_returns = max(0, self.trade_on_train(show_plot=show_plot))
        print(f"test_returns: {test_returns:.4f}")
        print(f"backtest_returns: {backtest_returns:.4f}")
        print(f"Overall Score: {score:.4f}")
        print(f"test_returns * backtest_returns: {test_returns * backtest_returns:.4f}")
        return test_returns * backtest_returns

    def trade_on_test(self, show_plot=True):
        """Perform trading simulation based on predicted price trends with improved strategy"""
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

        return self._simulate_trading(data=trade_df, show_plot=show_plot)

    def trade_on_train(self, show_plot=True):
        """Perform backtest using training data with improved strategy"""
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.X_train).cpu().numpy().flatten()
            train_pred = self.result_scaler.inverse_transform(
                train_pred.reshape(-1, 1)
            ).flatten()
            actual_close_train = self.df["Close"].iloc[: len(train_pred)].values
            actual_open_train = self.df["Open"].iloc[: len(train_pred)].values
            dates_train = self.df.index[: len(train_pred)]

        trade_df_train = pd.DataFrame(
            {
                "date": dates_train,
                "close": actual_close_train,
                "open": actual_open_train,
                "pred": train_pred,
            }
        ).set_index("date")

        return self._simulate_trading(data=trade_df_train, show_plot=show_plot)

    def _simulate_trading(self, data, show_plot=True):
        """Simulate trading on given trading dataframe (for either train or test data)"""

        # Trading parameters
        initial_capital = 10000.0
        cash = initial_capital
        shares = 0
        position = 0  # 0: no position, 1: long position
        buy_in_price = float("inf")

        # Calculate standard percentage change but handle negative prices correctly
        pred_values = data["pred"].values
        data["trend"] = np.zeros(len(data))
        
        for i in range(1, len(pred_values)):
            current = pred_values[i]
            previous = pred_values[i-1]
            
            # Handle the case where previous is zero
            if previous == 0:
                # Avoid division by zero
                data.iloc[i, data.columns.get_loc("trend")] = 1 if current > previous else -1
            else:
                # Calculate standard percentage change
                percentage_change = (current - previous) / abs(previous)
                data.iloc[i, data.columns.get_loc("trend")] = percentage_change
        
        # Generate trading signals based on the updated conditions
        data["signal"] = np.where(
            (data["trend"] >= self.transaction_fee),
            1,  # Buy signal
            np.where(
                (data["trend"] < -self.transaction_fee),
                -1,  # Sell signal
                0,  # Hold signal
            ),
        )
        trade_df = data.assign(
            action="hold",
            portfolio_value=initial_capital,
            drawdown=0.0,
            trade_price=np.nan,
        )

        # Trading simulation
        for i in range(1, len(trade_df)):
            current_open = trade_df["open"].iloc[i]
            # previous_close = trade_df["close"].iloc[i - 1]
            signal = trade_df["signal"].iloc[i]
            trend = trade_df["trend"].iloc[i]

            # fixed position_size
            # position_size = 0.2 * initial_capital
            position_size = 0.2 * initial_capital * (trend / 0.02)  # kelly

            buy_in_amount = position_size if cash >= position_size else cash

            if position == 1:
                if ((current_open - buy_in_price) / buy_in_price) > self.take_profit:
                    signal = -1

            # Trading logic
            if signal == 1:  # Buy signal
                max_shares = (
                    buy_in_amount * (1 - self.transaction_fee)
                ) / current_open  # 1% fee
                if max_shares > 0:
                    shares += max_shares
                    cash -= buy_in_amount
                    if position == 0:
                        buy_in_price = current_open
                    position = 1
                    trade_df.iloc[i, trade_df.columns.get_loc("action")] = "buy"
                    trade_df.iloc[i, trade_df.columns.get_loc("trade_price")] = (
                        current_open
                    )

            elif signal == -1 and position == 1:  # Sell signal
                sell_value = (
                    shares * current_open * (1 - self.transaction_fee)
                )  # 1% fee
                cash += sell_value
                shares = 0
                position = 0
                trade_df.iloc[i, trade_df.columns.get_loc("action")] = "sell"
                trade_df.iloc[i, trade_df.columns.get_loc("trade_price")] = current_open
                buy_in_price = float("inf")

            # Update portfolio value
            current_value = cash + shares * trade_df["close"].iloc[i]
            trade_df.iloc[i, trade_df.columns.get_loc("portfolio_value")] = (
                current_value
            )

        # Force liquidation on last day
        if shares > 0:
            cash += shares * trade_df["open"].iloc[-1] * (1 - self.transaction_fee)
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

        if show_plot:
            plt.figure(figsize=(12, 6))

            # Price and signals plot
            ax1 = plt.subplot(3, 1, 1)

            # Plot Actual Price with the first y-axis
            ax1.plot(trade_df["close"], label="Actual Price", alpha=0.7, color="blue")

            # Create a second y-axis to plot Predicted Price
            ax2 = ax1.twinx()
            ax2.plot(
                trade_df["pred"],
                label="Predicted Price",
                linestyle="--",
                alpha=0.7,
                color="orange",
            )

            # Plot buy and sell signals
            buy_signals = trade_df[trade_df["action"] == "buy"]
            sell_signals = trade_df[trade_df["action"] == "sell"]
            ax1.scatter(
                buy_signals.index,
                buy_signals["open"],
                marker="^",
                color="g",
                s=100,
                label="Buy",
                zorder=3,
            )
            ax1.scatter(
                sell_signals.index,
                sell_signals["open"],
                marker="v",
                color="r",
                s=100,
                label="Sell",
                zorder=3,
            )

            ax1.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            # Set titles and labels
            ax1.set_title("Price and Trading Signals")
            ax1.set_ylabel("Actual Price")
            ax2.set_ylabel("Predicted Price")

            # Adding legends
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            # Portfolio value plot
            ax3 = plt.subplot(3, 1, 2, sharex=ax1)
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
            ax4 = plt.subplot(3, 1, 3)
            plt.fill_between(
                trade_df.index, drawdowns * 100, 0, facecolor="red", alpha=0.3
            )
            plt.title(f"Drawdown (Max: {max_drawdown*100:.2f}%)")
            plt.ylabel("Drawdown (%)")

            plt.tight_layout()
            plt.show()

        return returns

    def predict_tomorrow_signal(self):
        self.model.eval()
        with torch.no_grad():
            all_data = self.df[self.features].values

            last_n_days = all_data[-self.n_steps :]
            last_sequence_scaled = self.scaler.transform(last_n_days).reshape(
                1, self.n_steps, -1
            )
            last_sequence = torch.tensor(last_sequence_scaled, dtype=torch.float32).to(
                self.device
            )
            predicted_close_scaled = self.model(last_sequence).cpu().numpy()
            predicted_close = self.result_scaler.inverse_transform(
                predicted_close_scaled
            ).flatten()[0]

            previous_n_days = all_data[-(self.n_steps + 1) : -1]
            previous_sequence_scaled = self.scaler.transform(previous_n_days).reshape(
                1, self.n_steps, -1
            )
            previous_sequence = torch.tensor(
                previous_sequence_scaled, dtype=torch.float32
            ).to(self.device)
            previous_predicted_scaled = self.model(previous_sequence).cpu().numpy()
            previous_predicted_close = self.result_scaler.inverse_transform(
                previous_predicted_scaled
            ).flatten()[0]

            # Calculate percentage change using standard formula but handle negative/zero values
            if previous_predicted_close == 0:
                # Avoid division by zero
                percentage_change = 1 if predicted_close > previous_predicted_close else -1
            else:
                # Standard percentage change calculation
                percentage_change = (predicted_close - previous_predicted_close) / abs(previous_predicted_close)

            if percentage_change >= self.transaction_fee:
                signal = "Buy"
            elif percentage_change < -self.transaction_fee:
                signal = "Sell"
            else:
                signal = "Hold"

            last_day_date = pd.to_datetime(self.df.index[-1])
            next_day_date = last_day_date + pd.Timedelta(days=1)

            print(
                f"--------Next Day ({next_day_date.strftime('%Y-%m-%d')}) Trade Advice:--------"
            )
            print(f"Predicted Close: {predicted_close:.2f}")
            print(f"Previous Predicted Close: {previous_predicted_close:.2f}")
            print(f"Predicted Percentage Change: {percentage_change:.4f}")
            print(f"Trade Signal: {signal}")
            return predicted_close, signal

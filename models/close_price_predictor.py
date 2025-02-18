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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer for the output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockPredictor:
    """Handles data processing, training, prediction, and visualization"""

    def __init__(self, df, device, n_steps=10, hidden_size=50, num_layers=3, dropout=0,
                 lr=0.0001, epochs=10000, test_ratio=0.1, patience=50, l2_weight_decay=0):
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

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.result_scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train, self.X_test, self.y_test, self.feature_size = self.prepare_data()
        self.initialize_model()

    def prepare_data(self):
        """Prepare training and testing data with time series split"""
        data = self.df.dropna()  # Remove rows with NaN values

        feature_columns = [col for col in self.df.columns if col not in ['index', 'Date']]  # 包含Close

        # Split data into train and test segments
        train_size = int(len(data) * (1 - self.test_ratio))
        test_start_idx = train_size - self.n_steps  # Maintain window size for test data
        train_data = data[:train_size]
        test_data = data[test_start_idx:]  # Include last n_steps from training
        
        # Fit scaler only on training data (exclude non-numeric columns)
        self.scaler.fit(train_data[feature_columns].values)  # Fit scaler only on numeric columns
        train_data_scaled = self.scaler.transform(train_data[feature_columns].values)
        test_data_scaled = self.scaler.transform(test_data[feature_columns].values)

        self.result_scaler.fit(train_data['Close'].values.reshape(-1, 1))
        train_return_scaled = self.result_scaler.transform(train_data['Close'].values.reshape(-1, 1))
        test_return_scaled = self.result_scaler.transform(test_data['Close'].values.reshape(-1, 1))
        
        def create_sequences(scaled_data, target_data):
            X, y = [], []
            for i in range(len(scaled_data) - self.n_steps - 1):
                X.append(scaled_data[i:i+self.n_steps])
                y.append(target_data[i+self.n_steps])
            return np.array(X), np.array(y)
        
        # Create sequences
        X_train, y_train = create_sequences(train_data_scaled, train_return_scaled)
        X_test, y_test = create_sequences(test_data_scaled, test_return_scaled)
        
        # Move data to device
        return (torch.tensor(X_train, dtype=torch.float32).to(self.device),
                torch.tensor(y_train, dtype=torch.float32).to(self.device),
                torch.tensor(X_test, dtype=torch.float32).to(self.device),
                torch.tensor(y_test, dtype=torch.float32).to(self.device),
                X_train.shape[2])

    def initialize_model(self):
        """Initialize model components"""
        self.model = LSTMModel(self.feature_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)  # Move model to device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay)  # L2 Regularization

    def train(self):
        print('Training model...')
        best_loss = float('inf')
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
                print(f"Early stopping at epoch {epoch+1} due to no improvement in loss")
                break

    def backtest(self):
        """Perform backtesting on test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(self.X_test).cpu().numpy().flatten()  # Move to CPU for plotting
        
            # Inverse scaling
            test_pred = self.result_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            actual_test_close = self.df['Close'].iloc[len(self.df) - len(self.y_test):].values  
            
            # Calculate metrics
            mse = mean_squared_error(actual_test_close, test_pred)
            mae = mean_absolute_error(actual_test_close, test_pred)
            rmse = np.sqrt(mse)
            
            print("\nBacktest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(actual_test_close, label='Actual Close', color='blue')
            plt.plot(test_pred, label='Predicted Close', color='red', linestyle='--')
            plt.title('Backtest Results')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

        return test_pred

    def predict_next_close(self):
        self.model.eval()
        feature_columns = [col for col in self.df.columns if col not in ['index', 'Date']]
        last_n_days = self.df[feature_columns].values[-self.n_steps:]
        last_n_days_scaled = self.scaler.transform(last_n_days)
        input_tensor = torch.tensor(last_n_days_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predicted_scaled = self.model(input_tensor).item()
        return self.result_scaler.inverse_transform([[predicted_scaled]])[0][0]

    def plot_results(self):
        """Plot training results"""
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.X_train).cpu().numpy().flatten()  # Move to CPU for plotting
        
            # Inverse scaling
            train_pred_original = self.result_scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            actual_train_close = self.df['Close'].iloc[:len(self.X_train)].values 
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual_train_close, label='Actual Close', color='blue')
            plt.plot(train_pred_original, label='Predicted Close', color='orange', linestyle='--')
            plt.title('Training Results')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

    # def generate_trading_signals(self):
    #     """Generate buy/hold/sell signals based on predicted returns, price change, and trend direction."""
    #     self.model.eval()
        
    #     # Rolling prediction using all test data
    #     predicted_prices = []
    #     actual_prices = self.scaler.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1)).flatten()

    #     with torch.no_grad():
    #         for i in range(len(self.X_test)):
    #             input_tensor = self.X_test[i].unsqueeze(0).to(self.device)  # Move to device
    #             predicted_scaled = self.model(input_tensor).item()
    #             predicted_price = self.scaler.inverse_transform([[predicted_scaled]])[0][0]
    #             predicted_prices.append(predicted_price)

    #     predicted_prices = np.array(predicted_prices)
        
    #     # Calculate predicted daily return (percentage change)
    #     predicted_change = (predicted_prices[1:] - actual_prices[:-1]) / actual_prices[:-1]

    #     # Generate buy signal: if predicted return > 1% and the trend is upward (predicted price continues to rise)
    #     buy_signals = np.full_like(predicted_change, np.nan)
    #     for i in range(1, len(predicted_change)-1):
    #         if predicted_change[i] > 0.01 and np.all(predicted_change[i:i+3] > 0):  # Trend is upwards for the next 3 days
    #             buy_signals[i] = actual_prices[i]
        
    #     # Generate sell signal: if predicted return < -1% or the trend is downward
    #     sell_signals = np.full_like(buy_signals, np.nan, dtype=float)
    #     for i in range(1, len(buy_signals)):
    #         if not np.isnan(buy_signals[i-1]):  # If there was a buy signal the previous day
    #             # Calculate the price change the next day
    #             price_change = (actual_prices[i] - actual_prices[i-1]) / actual_prices[i-1]
    #             # If the price drops more than 1% or trend is downward, generate a sell signal
    #             if price_change < -0.01 or np.all(predicted_change[i:i+3] < 0):  # Trend is downward for the next 3 days
    #                 sell_signals[i] = actual_prices[i]
        
    #     # Generate hold signals: any day that is neither a buy nor sell
    #     hold_signals = np.full_like(buy_signals, np.nan, dtype=float)
    #     for i in range(len(buy_signals)):
    #         if np.isnan(buy_signals[i]) and np.isnan(sell_signals[i]):
    #             hold_signals[i] = actual_prices[i]
        
    #     # Plot the price curves
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(actual_prices, label='Actual Close Price', color='blue', linewidth=1.5)
    #     plt.plot(predicted_prices, label='Predicted Close Price', color='red', linestyle='--', alpha=0.7)

    #     # Mark buy points (green ▲)
    #     plt.scatter(np.where(~np.isnan(buy_signals))[0], buy_signals[~np.isnan(buy_signals)], 
    #                 marker='^', color='green', label='Buy Signal', s=100)

    #     # Mark sell points (red ▼)
    #     plt.scatter(np.where(~np.isnan(sell_signals))[0], sell_signals[~np.isnan(sell_signals)], 
    #                 marker='v', color='red', label='Sell Signal', s=100)

    #     # Mark hold points (yellow ◯)
    #     plt.scatter(np.where(~np.isnan(hold_signals))[0], hold_signals[~np.isnan(hold_signals)], 
    #                 marker='o', color='yellow', label='Hold Signal', s=100)

    #     # Add grid, display ticks by day
    #     plt.xticks(range(0, len(actual_prices), 1))  # Display tick for each day
    #     plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    #     # Add title and labels
    #     plt.title('Trading Signals Based on Predicted Price Movements with Trend Analysis')
    #     plt.xlabel('Time Steps (Days)')
    #     plt.ylabel('Stock Price')
    #     plt.legend()
    #     plt.show()

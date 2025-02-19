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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockPredictor:
    """Handles data processing, training, prediction, and visualization"""

    def __init__(self, df, device, n_steps=10, hidden_size=50, num_layers=3,
                 lr=0.0001, epochs=10000, test_ratio=0.1, dropout=0, 
                 l2_weight_decay=0, early_stopping_patience=50):
        
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
        :param early_stopping_patience: Number of epochs to wait before stopping if no improvement.
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
        self.early_stopping_patience = early_stopping_patience

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1,1))
        self.result_scaler = MinMaxScaler(feature_range=(-1,1))
        self.X_train, self.y_train, self.X_test, self.y_test, self.feature_size = self.prepare_data()
        print('Feature size:', self.feature_size)
        self.initialize_model()

    def prepare_data(self):
        """Prepare training and testing data with all available features"""
        data = self.df.dropna()  # Remove rows with NaN values
        
        # Select all columns except 'Date' as features
        feature_columns = [col for col in self.df.columns if col not in ['index', 'Date']]  

        # Split data into train and test segments
        train_size = int(len(data) * (1 - self.test_ratio))
        test_start_idx = train_size - self.n_steps  
        train_data = data[:train_size]
        test_data = data[test_start_idx:]  

        # Normalize features based on training data
        self.scaler.fit(train_data[feature_columns].values)
        train_data_scaled = self.scaler.transform(train_data[feature_columns].values)
        test_data_scaled = self.scaler.transform(test_data[feature_columns].values)

        self.result_scaler.fit(train_data['Return'].values.reshape(-1, 1))
        train_return_scaled = self.result_scaler.transform(train_data['Return'].values.reshape(-1, 1))
        test_return_scaled = self.result_scaler.transform(test_data['Return'].values.reshape(-1, 1))


        def create_sequences(scaled_data, target_data):
            X, y = [], []
            for i in range(len(scaled_data) - self.n_steps - 1):
                X.append(scaled_data[i:i+self.n_steps])
                y.append(target_data[i+self.n_steps])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data_scaled, train_return_scaled.flatten())
        X_test, y_test = create_sequences(test_data_scaled, test_return_scaled.flatten())

        return (torch.tensor(X_train, dtype=torch.float32).to(self.device),
                torch.tensor(y_train, dtype=torch.float32).to(self.device),
                torch.tensor(X_test, dtype=torch.float32).to(self.device),
                torch.tensor(y_test, dtype=torch.float32).to(self.device),
                X_train.shape[2])

    def initialize_model(self):
        """Initialize the LSTM model"""
        self.model = LSTMModel(self.feature_size, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight_decay)

    def train(self):
        """Train the model with early stopping"""
        print('Training model...')

        best_loss = float('inf')
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

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def backtest(self):
        """Perform backtesting on the test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred_return = self.model(self.X_test).cpu().numpy().flatten()  
            predicted_test_return_original = self.result_scaler.inverse_transform(test_pred_return.reshape(-1, 1)).flatten()
            print('---------------predicted_test_return_original')
            print(predicted_test_return_original)

            actual_test_close = self.df['Close'].iloc[len(self.df) - len(self.y_test):].values  
            actual_test_return = self.df['Return'].iloc[len(self.df) - len(self.y_test):].values  

            predicted_test_close = actual_test_close[:-1] * (1 + predicted_test_return_original[1:])

            # Compute backtest metrics
            mse = mean_squared_error(actual_test_close[1:], predicted_test_close)
            mae = mean_absolute_error(actual_test_close[1:], predicted_test_close)
            rmse = np.sqrt(mse)

            print("\nBacktest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            # Plot actual vs. predicted close prices
            plt.figure(figsize=(12, 6))
            plt.plot(actual_test_close[1:], label='Actual Close', color='blue')
            plt.plot(predicted_test_close, label='Predicted Close', color='red', linestyle='--')

            # Plot actual and predicted returns on a secondary axis
            ax2 = plt.gca().twinx()
            
            actual_pos_returns = np.where(actual_test_return > 0, actual_test_return, np.nan)
            actual_neg_returns = np.where(actual_test_return <= 0, actual_test_return, np.nan)

            ax2.plot(actual_pos_returns, label='Actual Return (Positive)', color='purple', linestyle='-')
            ax2.plot(actual_neg_returns, label='Actual Return (Negative)', color='brown', linestyle='-')

            predicted_pos_returns = np.where(predicted_test_return_original > 0, predicted_test_return_original, np.nan)
            predicted_neg_returns = np.where(predicted_test_return_original <= 0, predicted_test_return_original, np.nan)

            ax2.plot(predicted_pos_returns, label='Predicted Return (Positive)', color='green', linestyle='-.')
            ax2.plot(predicted_neg_returns, label='Predicted Return (Negative)', color='red', linestyle='-.')

            plt.title('Backtest Results')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            ax2.set_ylabel('Return')

            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')
            plt.show()

            return predicted_test_close


    def plot_results(self):
        """Plot training results with predicted and actual close prices and returns"""
        self.model.eval()
        with torch.no_grad():
            train_pred_return = self.model(self.X_train).cpu().numpy().flatten()
            predicted_test_return_original = self.result_scaler.inverse_transform(train_pred_return.reshape(-1, 1)).flatten()
            print('---------------predicted_test_return_original')
            print(predicted_test_return_original)

            actual_train_close = self.df['Close'].iloc[:len(self.X_train)].values  
            actual_train_return = self.df['Return'].iloc[:len(self.X_train)].values  

            predicted_train_close = actual_train_close[:-1] * (1 + predicted_test_return_original[1:])

            plt.figure(figsize=(12, 6))
            plt.plot(actual_train_close, label='Actual Close', color='blue')
            plt.plot(np.concatenate(([actual_train_close[0]], predicted_train_close)), label='Predicted Close', color='orange', linestyle='--')

            # Plot actual and predicted returns on a secondary axis
            ax2 = plt.gca().twinx()
            
            actual_pos_returns = np.where(actual_train_return > 0, actual_train_return, np.nan)
            actual_neg_returns = np.where(actual_train_return <= 0, actual_train_return, np.nan)

            ax2.plot(actual_pos_returns, label='Actual Return (Positive)', color='purple', linestyle='-')
            ax2.plot(actual_neg_returns, label='Actual Return (Negative)', color='brown', linestyle='-')

            predicted_pos_returns = np.where(predicted_test_return_original > 0, predicted_test_return_original, np.nan)
            predicted_neg_returns = np.where(predicted_test_return_original <= 0, predicted_test_return_original, np.nan)

            ax2.plot(predicted_pos_returns, label='Predicted Return (Positive)', color='green', linestyle='-.')
            ax2.plot(predicted_neg_returns, label='Predicted Return (Negative)', color='red', linestyle='-.')

            plt.title('Training Results (Close Price and Return)')
            plt.xlabel('Time Steps')
            plt.ylabel('Price')
            ax2.set_ylabel('Return')

            plt.legend(loc='upper left')
            ax2.legend(loc='upper right')

            plt.show()

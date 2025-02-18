import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LSTMModel(nn.Module):
    """LSTM model for predicting Close price"""

    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output only Close price

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockPredictor:
    """Handles data processing, training, prediction, and visualization"""

    def __init__(self, df, n_steps=60, hidden_size=64, num_layers=2, 
                 lr=0.001, epochs=100, test_ratio=0.2):
        self.df = df.copy()
        self.n_steps = n_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.test_ratio = test_ratio

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train, self.y_train, self.X_test, self.y_test, self.feature_size = self.prepare_data()
        self.initialize_model()

    def prepare_data(self):
        """Prepare training and testing data with time series split"""
        data = self.df[['Close']].values
        
        # Split data into train and test segments
        train_size = int(len(data) * (1 - self.test_ratio))
        test_start_idx = train_size - self.n_steps  # Maintain window size for test data
        
        train_data = data[:train_size]
        test_data = data[test_start_idx:]  # Include last n_steps from training
        
        # Fit scaler only on training data
        self.scaler.fit(train_data)
        train_data_scaled = self.scaler.transform(train_data)
        test_data_scaled = self.scaler.transform(test_data)
        
        def create_sequences(scaled_data):
            X, y = [], []
            for i in range(len(scaled_data) - self.n_steps - 1):
                X.append(scaled_data[i:i+self.n_steps])
                y.append(scaled_data[i+self.n_steps])
            return np.array(X), np.array(y)
        
        # Create sequences
        X_train, y_train = create_sequences(train_data_scaled)
        X_test, y_test = create_sequences(test_data_scaled)
        
        return (torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32),
                X_train.shape[2])

    def initialize_model(self):
        """Initialize model components"""
        self.model = LSTMModel(self.feature_size, self.hidden_size, self.num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """Train the model"""
        print('Training model...')
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

    def backtest(self):
        """Perform backtesting on test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(self.X_test).numpy().flatten()
        
        # Inverse scaling
        test_pred = self.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        actual_test = self.scaler.inverse_transform(self.y_test.numpy().reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actual_test, test_pred)
        mae = mean_absolute_error(actual_test, test_pred)
        rmse = np.sqrt(mse)
        
        print("\nBacktest Results:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(actual_test, label='Actual Close', color='blue')
        plt.plot(test_pred, label='Predicted Close', color='red', linestyle='--')
        plt.title('Backtest Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        return test_pred

    def predict_next_close(self):
        """Predict next day's closing price"""
        self.model.eval()
        
        # Use most recent data
        last_n_days = self.df[['Close']].values[-self.n_steps:]
        last_n_days_scaled = self.scaler.transform(last_n_days)
        input_tensor = torch.tensor(last_n_days_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            predicted_scaled = self.model(input_tensor).item()
        
        return self.scaler.inverse_transform([[predicted_scaled]])[0][0]

    def plot_results(self):
        """Plot training results"""
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(self.X_train).numpy().flatten()
        
        # Inverse scaling
        train_pred = self.scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        actual_train = self.scaler.inverse_transform(self.y_train.numpy().reshape(-1, 1)).flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual_train, label='Actual Close', color='blue')
        plt.plot(train_pred, label='Predicted Close', color='orange', linestyle='--')
        plt.title('Training Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
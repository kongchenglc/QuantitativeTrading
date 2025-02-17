import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    """LSTM model for predicting Close price"""

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output only Close price

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Take the last timestep output


class StockClosePredictor:
    """Handles data processing, training, prediction, and visualization"""

    def __init__(self, df, n_steps=20, hidden_size=64, num_layers=2, lr=0.001, epochs=100):
        self.df = df.copy()
        self.n_steps = n_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize Close prices
        self.X_train, self.y_train, self.feature_size = self.prepare_data()
        self.initialize_model()

    def prepare_data(self):
        """Prepare training data from Close prices"""
        data = self.df[['Close']].values
        data_scaled = self.scaler.fit_transform(data)  # Normalize Close prices

        X, y = [], []
        for i in range(len(data_scaled) - self.n_steps - 1):
            X.append(data_scaled[i:i + self.n_steps])  # Past n_steps Close prices
            y.append(data_scaled[i + self.n_steps])  # Predict next Close price

        X = np.array(X)
        y = np.array(y)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1), X.shape[2]

    def initialize_model(self):
        """Initialize LSTM model, loss function, and optimizer"""
        self.model = LSTMModel(self.feature_size, self.hidden_size, self.num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """Train the LSTM model"""
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(self.X_train)
            loss = self.criterion(output, self.y_train)

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

    def predict_all(self):
        """Predict Close prices on training data for visualization"""
        self.model.eval()
        predicted_scaled = self.model(self.X_train).detach().numpy().flatten()

        # Inverse transform Close prices
        predicted_closes = self.scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()
        actual_closes = self.df["Close"].values[self.n_steps + 1:]

        return actual_closes, predicted_closes

    def predict_next_close(self):
        """Predict the next day's Close price"""
        self.model.eval()

        last_n_days = self.df[['Close']].values[-self.n_steps:]
        last_n_days_scaled = self.scaler.transform(last_n_days)  # Normalize
        input_tensor = torch.tensor(last_n_days_scaled, dtype=torch.float32).unsqueeze(0)

        predicted_scaled = self.model(input_tensor).item()
        predicted_close = self.scaler.inverse_transform([[predicted_scaled]])[0, 0]

        return predicted_close

    def plot_results(self):
        """Plot actual vs. predicted Close prices"""
        actual_closes, predicted_closes = self.predict_all()
        predicted_next_close = self.predict_next_close()

        plt.figure(figsize=(12, 6))
        plt.plot(actual_closes, label="Actual Close", color="blue", linestyle="-")
        plt.plot(predicted_closes, label="Predicted Close (Train)", color="orange", linestyle="--")
        plt.scatter(len(actual_closes), predicted_next_close, color="red", label="Predicted Next Close", zorder=3)

        plt.legend()
        plt.title("Stock Close Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.show()

        print(f"Predicted Next Day Close: {predicted_next_close:.2f}")


# ---- Run LSTM Prediction ----
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/cleaned_data.csv")

    # Initialize predictor
    predictor = StockClosePredictor(df, n_steps=20, hidden_size=64, num_layers=2, lr=0.001, epochs=100)

    # Train the model
    predictor.train()

    # Plot Close price predictions
    predictor.plot_results()

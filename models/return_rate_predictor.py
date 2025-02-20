import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
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
        epochs=2000,
        patience=30,
        lr=0.0001,
        n_steps=10,
        hidden_size=50,
        num_layers=3,
        dropout=0,
        l1_weight_decay=0,
        l2_weight_decay=0,
        test_ratio=0.1,
        features=None,  # Add features as a parameter
    ):
        """
        Initialize the stock predictor model with provided hyperparameters.

        :param df: Input DataFrame with stock data.
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
        self.l1_weight_decay = l1_weight_decay
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

    def l1_regularization(self):
        """Calculate L1 regularization term"""
        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
        return self.l1_weight_decay * l1_norm

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

            if self.l1_weight_decay > 0:
                loss += self.l1_regularization()

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

    def plot_results(self):
        """Plot training results with predicted and actual returns"""
        self.model.eval()
        with torch.no_grad():
            # Get the predicted returns from the model
            train_pred_return = self.model(self.X_train).cpu().numpy().flatten()
            print('-------------train_pred_return')
            print(train_pred_return)
            predicted_train_return_original = self.result_scaler.inverse_transform(
                train_pred_return.reshape(-1, 1)
            ).flatten()
            print('-------------predicted_train_return_original')
            print(predicted_train_return_original)

            # Get the actual returns from the dataframe
            actual_train_return = self.df["Return"].iloc[: len(self.X_train)].values
            print('-------------actual_train_return')
            print(actual_train_return)
            # Create a single plot to compare returns
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot actual returns and predicted returns
            ax.plot(
                actual_train_return,
                label="Actual Return",
                color="blue",
                linestyle="-",
                linewidth=2,
            )
            ax.plot(
                predicted_train_return_original,
                label="Predicted Return",
                color="red",
                linestyle="--",
                linewidth=2,
            )

            # Add a baseline where the return equals zero
            ax.axhline(
                0,
                color="black",
                linestyle="-",
                linewidth=1,
                label="Baseline (Return = 0)",
            )

            # Set title and labels for the axes
            ax.set_title("Training Results: Actual vs Predicted Returns", fontsize=16)
            ax.set_xlabel("Time Steps", fontsize=14)
            ax.set_ylabel("Return", fontsize=14)
            ax.grid(True)  # Add gridlines to improve readability

            # Add the legend
            ax.legend(loc="upper left", fontsize=12)

            # Adjust layout to avoid overlapping of elements
            plt.tight_layout()

            # Display the plot
            plt.show()

    def test(self, show_plot=True):
        """Perform backtesting on the test set"""
        self.model.eval()
        with torch.no_grad():
            test_pred_return = self.model(self.X_test).cpu().numpy().flatten()
            predicted_test_return_original = self.result_scaler.inverse_transform(
                test_pred_return.reshape(-1, 1)
            ).flatten()

            actual_test_return = (
                self.df["Return"].iloc[len(self.df) - len(self.y_test) :].values
            )
            
            print('-------------test_pred_return')
            print(test_pred_return)
            
            print('-------------actual_test_return')
            print(actual_test_return)
            
            print('-------------predicted_test_return_original')
            print(predicted_test_return_original)

            mse = mean_squared_error(actual_test_return, predicted_test_return_original)
            mae = mean_absolute_error(
                actual_test_return, predicted_test_return_original
            )
            rmse = np.sqrt(mse)

            print("\nBacktest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            actual_direction = np.sign(actual_test_return)
            predicted_direction = np.sign(predicted_test_return_original)
            direction_accuracy = np.mean(actual_direction == predicted_direction)

            r2 = r2_score(actual_test_return, predicted_test_return_original)

            print("\nTest Results:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            print(f"Direction Accuracy: {direction_accuracy:.4f}")
            print(f"R-squared: {r2:.4f}")

            w1, w2, w3, w4 = 0.2, 0.3, 0.3, 0.2
            normalized_rmse = 1 / (1 + rmse)
            normalized_mse_mae = 1 / (1 + mse + mae)
            score = (
                w1 * normalized_rmse
                + w2 * direction_accuracy
                + w3 * r2
                - w4 * normalized_mse_mae
            )

            print(f"Overall Model Score: {score:.4f}")

            if show_plot:
                # Create a subplot for returns
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot actual vs. predicted returns
                ax.plot(
                    actual_test_return,
                    label="Actual Return",
                    color="blue",
                    linestyle="-",
                )
                ax.plot(
                    predicted_test_return_original,
                    label="Predicted Return",
                    color="red",
                    linestyle="--",
                )

                # Set the plot style
                ax.set_title("Backtest Results: Actual vs Predicted Returns")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Return")
                ax.axhline(
                    0,
                    color="black",
                    linestyle="-",
                    linewidth=1,
                    label="Baseline (Return = 0)",
                )  # Baseline for comparison
                ax.grid(True)  # Add grid lines
                ax.legend(loc="upper left")

                # Show the plot
                plt.tight_layout()
                plt.show()

            return score

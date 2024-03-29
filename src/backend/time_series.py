import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesForecastingModel:
    def __init__(self, sequence_length=7, input_size=4, hidden_size=64, num_layers=2, output_size=4):
        """
        Initialize the TimeSeriesForecastingModel.

        Args:
            sequence_length (int): Length of input sequences.
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output features.
        """
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = LSTM(input_size, hidden_size, num_layers, output_size, self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.load_data('src/backend/DailyDelhiClimateTrain.csv')
        



    def load_data(self, train_file):
        """
        Load and preprocess data.

        Args:
            train_file (str): Path to the training data CSV file.
        """
        self.data = pd.read_csv(train_file)

        self.data['date'] = pd.to_datetime(self.data['date'])

        self.scaler = MinMaxScaler()
        self.data[['meantemp', 'humidity', 'wind_speed', 'meanpressure']] = self.scaler.fit_transform(self.data[['meantemp', 'humidity', 'wind_speed', 'meanpressure']])

        X, y = self.create_sequences(self.data[['meantemp', 'humidity', 'wind_speed', 'meanpressure']].values, self.sequence_length)

        self.X_tensor = torch.Tensor(X)
        self.y_tensor = torch.Tensor(y)


    def create_sequences(self, data, sequence_length):
        """
        Create input-output sequences from the data.

        Args:
            sequence_length (int): Length of input sequences.

        Returns:
            numpy.ndarray: Input sequences.
            numpy.ndarray: Output sequences.
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def train_model(self, num_epochs, batch_size, learning_rate):
        """
        Train the model.

        Args:
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for optimizer.
        """
        for epoch in range(num_epochs):
            for i in range(0, len(self.X_tensor), batch_size):
                inputs = self.X_tensor[i:i+batch_size].to(self.device)
                targets = self.y_tensor[i:i+batch_size].to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        """
        pass

    def predict(self, prediction_lenghts = 30):
        """
        Make predictions using the trained model.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.X_tensor[-1].unsqueeze(0).to(self.device)
            for i in range(prediction_lenghts):
                pred = self.model(inputs)
                inputs = torch.cat((inputs[:,1:,:], pred.unsqueeze(0)), dim=1)

            return inputs.squeeze(0).cpu().numpy()

    def invert_scaling(self, values):
        """
        Invert scaling of predicted values.
        """
        return self.scaler.inverse_transform(values)


if __name__ == "__main__":
    model = TimeSeriesForecastingModel()
    model.train_model(num_epochs=100, batch_size=32, learning_rate=0.001)
    values = model.predict()
    inverted_values =model.invert_scaling(values)
    print(inverted_values)
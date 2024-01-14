# Imports
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import ta  # Technical Analysis Library
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from math import sqrt
import time

# Load environment variables
APCA_API_KEY_ID = os.getenv("ALPACA_API_KEY")
APCA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
# target_stock = 'SRNE'
target_stock = 'NVDA'

# Initialize Alpaca API
api = tradeapi.REST(APCA_API_KEY_ID, APCA_API_SECRET_KEY, 'https://paper-api.alpaca.markets')

# Fetch Data
def fetch_data(symbol, start_date, end_date):
    data = api.get_bars(symbol, TimeFrame.Minute, start_date, end_date, adjustment='raw').df

    # Add technical indicators and future close column
    data['close_future'] = data['close'].shift(-15)
    data['rsi'] = ta.momentum.rsi(data['close'])
    data['ema'] = ta.trend.ema_indicator(data['close'])
    data['cmf'] = ta.volume.chaikin_money_flow(data['high'], data['low'], data['close'], data['volume'])
    data['vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])
    data['bollinger_high'] = ta.volatility.bollinger_hband(data['close'])
    data['bollinger_low'] = ta.volatility.bollinger_lband(data['close'])
    data['macd'] = ta.trend.macd(data['close'])
    ichimoku = ta.trend.IchimokuIndicator(data['high'], data['low'])
    data['ichimoku_a'] = ichimoku.ichimoku_a()
    data['ichimoku_b'] = ichimoku.ichimoku_b()
    data['ichimoku_base'] = ichimoku.ichimoku_base_line()
    data['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    data['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'])

    data.dropna(inplace=True)
    return data

# train_data = fetch_data(target_stock, '2020-06-01', '2020-07-03')
# test_data = fetch_data(target_stock, '2020-07-02', '2020-07-03')

train_data = fetch_data(target_stock, '2019-12-18', '2019-12-31')
test_data = fetch_data(target_stock, '2020-01-01', '2023-12-26')

# train_data = fetch_data(target_stock, '2023-10-01', '2023-11-01')
# test_data = fetch_data(target_stock, '2023-11-02', '2023-12-26')

# Assuming train_data and test_data are pandas DataFrames already defined and have a 'close' column.

# Print the columns of the DataFrame to verify its structure
print(f"Number of Train Data: {train_data.columns}")

# Define the window size and the number of features
WINDOW_SIZE = 15
NUM_FEATURES = len(train_data.columns) - 1  # Adjust based on the actual number of features

# Now print NUM_FEATURES to verify it's correct
print(f"Calculated NUM_FEATURES: {NUM_FEATURES}")

# Standardize the training data and normalize the target variable
ss = StandardScaler()
mm = MinMaxScaler()

# Make sure 'ds' column is created before running the rolling window loop
train_data['ds'] = train_data.index
test_data['ds'] = test_data.index

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # This should give you a shape of (batch_size, output_size)
        return out

# Training function
def train_model(model, optimiser, loss_fn, X_train, y_train, n_epochs, window_num):
    y_train = y_train.view(-1, 1)
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()
        outputs = model(X_train)
        if outputs.shape != y_train.shape:
            outputs = outputs.view(*y_train.shape)

        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimiser.step()
        end_time = time.time()

        iteration_time = end_time - start_time
        iterations_per_second = 1 / iteration_time

        if epoch % 100 == 0:
            print(f'Window {window_num}, Epoch {epoch}/{n_epochs}, Loss: {loss.item():.6f}, Iterations/s: {iterations_per_second:.2f}')

# Select the device for training
# Check for available devices in order of preference: CUDA, MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Initialize the LSTM model
hidden_size = 50
num_layers = 1
output_size = 1

# Print the LSTM configuration
# print(f"Number of Features: {NUM_FEATURES}")
# print(f"Hidden Size: {hidden_size}")
# print(f"Number of Layers: {num_layers}")
# print(f"Output Size: {output_size}")

model = LSTM(NUM_FEATURES, hidden_size, num_layers, output_size).to(device)

# Training parameters
n_epochs = 100
learning_rate = 0.001

loss_fn = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Rolling window prediction and training
predictions = []

# Rolling window prediction and training
for i in range(0, len(test_data), WINDOW_SIZE):
    window_num = i // WINDOW_SIZE + 1
    print(f"Processing window {window_num}/{len(test_data)//WINDOW_SIZE + (len(test_data) % WINDOW_SIZE > 0)}")

    end_idx = i + WINDOW_SIZE

    # Prepare training data
    train_window = train_data.copy()
    # Drop non-numeric columns
    train_window_numeric = train_window.select_dtypes(include=[np.number])

    X_train_scaled = ss.fit_transform(train_window_numeric.drop(columns=['close_future']))
    y_train_scaled = mm.fit_transform(train_window_numeric[['close_future']])

    # Convert to tensors
    X_train_tensors = Variable(torch.Tensor(X_train_scaled)).to(device)
    y_train_tensors = Variable(torch.Tensor(y_train_scaled)).to(device)

    # Reshape for LSTM input
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    y_train_tensors_final = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1, y_train_tensors.shape[1]))

    # Train the model
    train_model(model, optimiser, loss_fn, X_train_tensors_final, y_train_tensors_final, n_epochs, window_num)

    # Prepare test data
    test_window = test_data.iloc[i:end_idx]
    # Drop non-numeric columns
    test_window_numeric = test_window.select_dtypes(include=[np.number])

    X_test_scaled = ss.transform(test_window_numeric.drop(columns=['close_future']))
    X_test_tensors = Variable(torch.Tensor(X_test_scaled)).to(device)
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    # Predict on the next window of test data
    model.eval()
    with torch.no_grad():
        test_prediction = model(X_test_tensors_final)
        test_prediction = test_prediction.cpu().data.numpy()
        test_prediction = mm.inverse_transform(test_prediction)
        predictions.extend(test_prediction.flatten().tolist())

    # Add the window to the training data for the next iteration
    train_data = pd.concat([train_data, test_window])

# Function to run predictions on training data
# Function to run predictions on training data
def predict_model(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(data_loader):
            X_batch = X_batch.view(X_batch.size(0), 1, -1).to(device)
            y_pred = model(X_batch)
            y_pred = y_pred.cpu().data.numpy()
            y_pred = mm.inverse_transform(y_pred.reshape(-1, 1))
            predictions.extend(y_pred.flatten().tolist())
    return predictions

# Convert the training dataset to tensors
train_data_scaled = ss.fit_transform(train_data.select_dtypes(include=[np.number]).drop(columns=['close']))
train_targets_scaled = mm.fit_transform(train_data.select_dtypes(include=[np.number])[['close']])

train_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(train_data_scaled),
    torch.Tensor(train_targets_scaled)
)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

# Get predictions on training data
train_predictions = predict_model(model, train_loader)

# After generating predictions for the test set
test_predictions = predictions[:len(test_data)]  # assuming 'predictions' contains your test set predictions

# Calculate error metrics for the predictions
actuals = test_data['close_future'][:len(predictions)]
mae = mean_absolute_error(actuals, predictions)
rmse = sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)
print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}')

# Creating a ledger DataFrame
ledger = pd.DataFrame({
    'Timestamp': train_data.index.tolist() + test_data.index.tolist(),
    'Actual Future Close': train_data['close_future'].tolist() + test_data['close_future'][:len(predictions)].tolist(),
    'Predicted Future Close': train_predictions + predictions,
})

# Plotting the actual vs predicted values
plt.figure(figsize=(15, 7))
plt.xticks(rotation=45)

# Plot training actual future close
plt.plot(train_data.index, train_data['close_future'], label='Train Actual Future Close', color='blue')

# Plot training predicted future close
# Ensure train_predictions aligns correctly with train_data indices
plt.plot(train_data.index[:len(train_predictions)], train_predictions, label='Train Predicted Future Close', color='orange', alpha=0.7)

# Plot test actual future close
plt.plot(test_data.index, test_data['close_future'], label='Test Actual Future Close', color='green')

# Plot test predicted future close
plt.plot(test_data.index[:len(predictions)], predictions, label='Test Predicted Future Close', color='red', alpha=0.7)

# Add divider line between training and test data
divider_timestamp = test_data.index[0]
plt.axvline(x=divider_timestamp, color='black', linestyle='--', linewidth=2, label='Train/Test Divider')

plt.title('Train and Test Actual Future Close vs Predicted Future Close')
plt.xlabel('Time')
plt.ylabel('Future Close Price')
plt.legend()
# plt.show()

# Creating a ledger DataFrame
ledger = pd.DataFrame({
    'Timestamp': train_data.index.tolist() + test_data.index.tolist(),
    'Actual Future Close': train_data['close_future'].tolist() + test_data['close_future'][:len(predictions)].tolist(),
    'Predicted Future Close': train_predictions + predictions,
})

plt.figure(figsize=(15, 7))
plt.xticks(rotation=45)

# Plot actual future close
plt.plot(test_data.index, test_data['close_future'], label='Actual Future Close', color='green')

# Plot predicted future close
plt.plot(test_data.index[:len(predictions)], predictions, label='Predicted Future Close', color='red', alpha=0.7)

plt.title('Test Actual Future Close vs Predicted Future Close')
plt.xlabel('Time')
plt.ylabel('Future Close Price')
plt.legend()
plt.show()
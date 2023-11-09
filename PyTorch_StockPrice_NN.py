#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:29:14 2023

@author: paviprathiraja

PyTorch for machine learning in finance, specifically for predictive analysis.
Basic neural network to predict stock prices based on historical data.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical stock price data
# For demonstration, let's assume 'data.csv' contains two columns: 'Date' and 'Close'
data = pd.read_csv('data.csv')
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
closing_prices_normalized = scaler.fit_transform(closing_prices)

# Prepare data for training
seq_length = 10  # Length of sequences for each input data point
X, y = [], []

for i in range(len(closing_prices_normalized) - seq_length):
    X.append(closing_prices_normalized[i:i + seq_length])
    y.append(closing_prices_normalized[i + seq_length])

X, y = np.array(X), np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Define a simple neural network
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
input_size = 1  # Number of features (closing price)
hidden_size = 50  # Number of hidden units
output_size = 1  # Output size (predicted closing price)
model = StockPredictor(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Training the model
num_epochs = 130
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    mse = mean_squared_error(y_test_tensor, test_outputs)
    print(f'Mean Squared Error on Test Data: {mse:.4f}')

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(test_outputs.numpy())
actual_prices = scaler.inverse_transform(y_test_tensor.numpy())

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
#plt.xlim (100, 150)
plt.show()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.xlim (200, 250)
plt.show()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.xlim (0, 50)
plt.show()

import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to create features and target variable
def create_features_target(data, target_col, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])  # Extracting only the first column (adjusted close prices)
        y.append(data[i+window_size, 0])     # Extracting the next day's adjusted close price
    return np.array(X), np.array(y)

# Function to create and train a simple feedforward neural network
def create_and_train_nn(X_train, y_train, epochs=100, batch_size=32):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Code modified from https://keras.io/guides/training_with_built_in_methods/

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse, predictions

# Main function
def main():
    # Parameters
    symbols = ['AAPL', 'MSFT', 'AMD', 'NVDA']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    window_size = 10

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, symbol in enumerate(symbols):
        # Fetch historical stock data
        stock_data = fetch_stock_data(symbol, start_date, end_date)

        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data['Adj Close'].values.reshape(-1,1))

        # Create features and target variable
        X, y = create_features_target(scaled_data, 'Adj Close', window_size)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train a simple feedforward neural network
        model = create_and_train_nn(X_train, y_train)

# Code modified from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

        # Evaluate the model
        rmse, predictions = evaluate_model(model, X_test, y_test)

        # Plot actual vs predicted prices
        row = i // 2
        col = i % 2
        axs[row, col].plot(y_test, label='Actual')
        axs[row, col].plot(predictions, label='Predicted')
        axs[row, col].set_title(f'{symbol} - RMSE: {rmse:.4f}')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Price')
        axs[row, col].legend()
        

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
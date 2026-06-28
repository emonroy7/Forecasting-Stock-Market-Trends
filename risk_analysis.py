import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Function to fetch historical stock price data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess data and generate sequences
def preprocess_data(data, sequence_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    sequences = []
    targets = []
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i+sequence_length])
        targets.append(scaled_data[i+sequence_length])

    return np.array(sequences), np.array(targets), scaler

# Function to build LSTM model
def build_model(sequence_length, input_dim):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, input_dim)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Code modified from https://faroit.com/keras-docs/1.0.4/getting-started/sequential-model-guide/

# Function to train LSTM model
def train_model(model, X_train, y_train, epochs, batch_size):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Function to perform risk and return analysis
def analyze_risk_and_return(predictions, actual_values, scaler):
    # Denormalize the predicted and actual values
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(actual_values.reshape(-1, 1))

    # Calculate percentage change between consecutive days
    predicted_change = np.diff(predicted_prices, axis=0) / predicted_prices[:-1] * 100
    actual_change = np.diff(actual_prices, axis=0) / actual_prices[:-1] * 100

    # Calculate standard deviation of percentage change (risk)
    predicted_std = np.std(predicted_change)
    actual_std = np.std(actual_change)

    # Calculate returns
    returns = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100

    return predicted_std, actual_std, returns[0]

# Code modified from https://www.kaggle.com/code/santosh1974/basic-stock-price-prediction-using-lstm
def main():
    # Parameters
    symbols = ['AAPL', 'MSFT', 'AMD', 'INTC', 'NVDA']
    start_date = '2021-01-01'
    end_date = '2022-01-01'
    sequence_length = 50
    epochs = 5
    batch_size = 32

    # Fetch historical stock price data for all symbols
    stock_data = {}
    for symbol in symbols:
        stock_data[symbol] = fetch_stock_data(symbol, start_date, end_date)

    # Perform analysis for each stock
    risks = []
    returns = []
    for symbol in symbols:
        data = stock_data[symbol]
        X, y, scaler = preprocess_data(data, sequence_length)

        # Split data into training and testing sets
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Reshape data for LSTM input
        input_dim = X_train.shape[2]

        # Build LSTM model
        model = build_model(sequence_length, input_dim)

        # Train LSTM model
        train_model(model, X_train, y_train, epochs, batch_size)

        # Make predictions
        predictions = model.predict(X_test)

        # Perform risk and return analysis
        predicted_std, actual_std, ret = analyze_risk_and_return(predictions, y_test, scaler)
        risks.append(actual_std)
        returns.append(ret)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    ax1.bar(symbols, risks, color='b', alpha=0.5, label='Risk (Standard Deviation)')
    ax2.bar(symbols, returns, color='r', alpha=0.5, label='Returns (%)')

    ax1.set_xlabel('Stocks')
    ax1.set_ylabel('Risk (Standard Deviation)')
    ax2.set_ylabel('Returns (%)')

    ax1.set_title('Risk and Returns of Stocks')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

if __name__ == "__main__":
    main()
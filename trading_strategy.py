import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import matplotlib.pyplot as plt

# Load historical price data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Calculate technical indicators (example: moving averages)
def calculate_technical_indicators(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

# Define features and target variable
def define_features_target(data):
    data['Price_Up'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 if price will go up, 0 otherwise
    data.dropna(inplace=True)  # Remove NaN values

    X = data[['MA50', 'MA200']]  # Features
    y = data['Price_Up']  # Target variable
    return X, y

# Train Random Forest Classifier model 
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test
# Code modified from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


# Evaluate model
def evaluate_model(model, X_test, y_test, data, ticker):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Example of using the model for trading decisions
    last_row = data.tail(1)
    prediction_today = model.predict([[last_row['MA50'].values[0], last_row['MA200'].values[0]]])

    plt.plot(data.index, data['Close'], label=ticker + ' Close Price')
    plt.plot(data.index, data['MA50'], label=ticker + ' 50-day MA')
    plt.plot(data.index, data['MA200'], label=ticker + ' 200-day MA')

    plt.text(data.index[-1], data['Close'].iloc[-1], f"Accuracy: {accuracy:.2f}\nPrediction: {'Up' if prediction_today == 1 else 'Down'}")

def main():
    symbols = ['AAPL', 'MSFT', 'AMD', 'NVDA']
    start_date = "2010-01-01"
    end_date = "2022-01-01"

    plt.figure(figsize=(12, 6))

    for symbol in symbols:
        # Fetch historical price data
        data = fetch_stock_data(symbol, start_date, end_date)

        # Calculate technical indicators
        data = calculate_technical_indicators(data)

        # Define features and target variable
        X, y = define_features_target(data)

        # Train machine learning model
        model, X_test, y_test = train_model(X, y)

        # Evaluate model
        evaluate_model(model, X_test, y_test, data, symbol)  # Pass 'data' to evaluate_model

    plt.title('Historical Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

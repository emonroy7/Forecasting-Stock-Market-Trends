import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_model_and_predict(ticker):
    # Fetch historical stock data
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Prepare the data for machine learning
    stock_data['Next Close'] = stock_data['Close'].shift(-1)  # Create a column for next day's close price
    stock_data.dropna(inplace=True)  # Drop rows with NaN values

    # Define features and target variable
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = stock_data['Next Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    #Code modified from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    # Evaluate the model
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print("Training RMSE for {}: {:.4f}".format(ticker, train_rmse))
    print("Testing RMSE for {}: {:.4f}".format(ticker, test_rmse))

    # Make predictions for future stock prices
    future_dates = pd.date_range(start=end_date, periods=30)  # Predicting next 30 days
    future_data = pd.DataFrame(index=future_dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    #Code modified from https://pandas.pydata.org/docs/user_guide/timeseries.html

    # Fill future data with the last available historical data
    last_data_row = stock_data.iloc[-1]
    future_data.fillna(last_data_row, inplace=True)

    future_predictions = model.predict(future_data)

    # Align y_test with future_dates
    y_test_aligned = y_test[-len(future_dates):]

    return future_dates, future_predictions, test_preds, y_test_aligned

def main():
    # Symbols to analyze
    symbols = ['AAPL', 'MSFT', 'AMD', 'INTC', 'NVDA']

    for symbol in symbols:
        future_dates, future_predictions, test_preds, y_test = train_model_and_predict(symbol)

        fig, ax = plt.subplots()
        ax.plot(future_dates, future_predictions, label='Predicted {}'.format(symbol))
        ax.plot(future_dates, y_test.values, label='Actual {}'.format(symbol), linestyle='--')
        ax.plot(future_dates, np.abs(future_predictions - y_test.values), label='Error {}'.format(symbol))
        plt.xlabel('Date')
        plt.ylabel('Close Price / Error')
        plt.title('Stock Price Predictions and Errors for {}'.format(symbol))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
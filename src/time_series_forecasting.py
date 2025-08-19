# src/time_series_forecasting.py

import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.pyplot as plt
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ensure the project's root directory is in the system path for imports
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.data_cleaning import clean_price_data

def arima_forecasting(data_path: str):
    """
    Develops and evaluates an ARIMA time series forecasting model for a financial asset.
    """
    # 1. Load the cleaned data
    try:
        tsla_df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Please check your data directory.")
        return

    # Check for the correct price column
    if 'Close' not in tsla_df.columns:
        print(f"Error: The 'Close' column was not found in the DataFrame.")
        print(f"Available columns are: {tsla_df.columns.tolist()}")
        return
    
    tsla_prices = tsla_df['Close'].asfreq('D').ffill()

    # 2. Divide the dataset into chronological training and testing sets
    train_size = int(len(tsla_prices) * 0.8)
    train, test = tsla_prices.iloc[:train_size], tsla_prices.iloc[train_size:]

    print(f"Training data size: {len(train)}")
    print(f"Testing data size: {len(test)}")

    # 3. Optimize Model Parameters using auto_arima
    print("\nSearching for optimal ARIMA parameters...")
    model = pm.auto_arima(
        train,
        start_p=1, start_q=1,
        test='adf',
        max_p=5, max_q=5,
        m=1,
        d=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    print(f"\nBest ARIMA model parameters: {model.order}")

    # 4. Use the model to forecast over the test set period and future
    n_periods_test = len(test)
    n_periods_future = 252 # 1 year of trading days
    
    # Get test forecast and confidence intervals
    test_forecast, test_conf_int = model.predict(n_periods=n_periods_test, return_conf_int=True)
    
    # Get future forecast and confidence intervals
    future_forecast, future_conf_int = model.predict(n_periods=n_periods_future, return_conf_int=True)
    
    # 5. Visualize the forecast against the test data
    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data', color='gray')
    plt.plot(test.index, test_forecast, label='ARIMA Test Forecast', color='red')
    
    # Plotting future forecast
    future_index = pd.date_range(start=test.index[-1], periods=n_periods_future, freq='B')
    plt.plot(future_index, future_forecast, label='ARIMA Future Forecast', color='green', linestyle='--')
    
    # Plot confidence intervals
    plt.fill_between(future_index, future_conf_int[:, 0], future_conf_int[:, 1], color='green', alpha=0.1)
    
    plt.title("TSLA Stock Price: ARIMA Forecast with Confidence Intervals")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_lstm_dataset(data, look_back=1):
    """
    Creates a dataset for the LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def forecast_lstm(model, last_sequence, n_periods):
    """
    Generates a recursive forecast for LSTM model.
    """
    forecast = []
    current_sequence = last_sequence.copy()
    for _ in range(n_periods):
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], 1))
        forecast.append(next_pred[0, 0])
        current_sequence = np.append(current_sequence[1:], next_pred)
    return np.array(forecast)

def lstm_forecasting(data_path: str):
    """
    Develops and evaluates an LSTM time series forecasting model.
    """
    try:
        tsla_df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        print("Data loaded successfully for LSTM.")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Please check your data directory.")
        return

    # Check for the correct price column
    if 'Close' not in tsla_df.columns:
        print(f"Error: The 'Close' column was not found in the DataFrame.")
        print(f"Available columns are: {tsla_df.columns.tolist()}")
        return

    tsla_prices = tsla_df['Close'].values.reshape(-1, 1)

    # Normalize the data for the LSTM model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(tsla_prices)

    # Divide the dataset into chronological training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_scaled, test_scaled = scaled_data[:train_size], scaled_data[train_size:]

    # Create datasets with look-back
    look_back = 60 # Using 60 trading days as the look-back window
    X_train, y_train = create_lstm_dataset(train_scaled, look_back)
    X_test, y_test = create_lstm_dataset(test_scaled, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("\nTraining LSTM model...")
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

    # Make predictions on the test set
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Generate future forecast
    n_periods_future = 252
    last_train_sequence = scaled_data[-(look_back + 1):-1]
    future_forecast_scaled = forecast_lstm(model, last_train_sequence, n_periods_future)
    future_forecast = scaler.inverse_transform(future_forecast_scaled.reshape(-1, 1))
    
    # Generate the test forecast plot
    plt.figure(figsize=(15, 6))
    plt.plot(tsla_df.index, tsla_prices, label='Actual Prices')
    
    # Plot test forecast
    test_index = tsla_df.index[train_size + look_back + 1: train_size + look_back + 1 + len(test_predict)]
    plt.plot(test_index, test_predict, label='LSTM Test Forecast', color='orange')
    
    # Plot future forecast
    future_index = pd.date_range(start=tsla_df.index[-1], periods=n_periods_future, freq='B')
    plt.plot(future_index, future_forecast, label='LSTM Future Forecast', color='purple', linestyle='--')
    
    plt.title("TSLA Stock Price: LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

# You can add this to your main script to run both models
if __name__ == '__main__':
    data_path = 'data/processed/tsla_clean.csv'
    # Call the ARIMA function from the previous step
    arima_forecasting(data_path) 
    
    # Call the new LSTM function
    lstm_forecasting(data_path)
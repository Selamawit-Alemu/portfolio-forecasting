import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.pyplot as plt
import os
import sys
import joblib  # <-- You need to add this import

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

    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, 'arima_model.joblib')
    joblib.dump(model, model_path)
    print(f"ARIMA model saved successfully to {model_path}")
    # **END OF ADDED BLOCK**
    
    # 4. Use the model to forecast over the test set period
    n_periods = len(test)
    forecast_arima = model.predict(n_periods=n_periods)
    
    # Create a DataFrame for the forecast
    forecast_index = pd.date_range(start=test.index[0], periods=n_periods, freq='B')
    forecast_df = pd.DataFrame(forecast_arima, index=forecast_index, columns=['Forecast'])

    # 5. Visualize the forecast against the test data
    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Test Data', color='gray')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='ARIMA Forecast', color='red')
    plt.title("TSLA Stock Price: ARIMA Forecast vs. Actual")
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

    # **ADD THIS BLOCK**
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    scaler_path = os.path.join(model_dir, 'lstm_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"LSTM scaler saved successfully to {scaler_path}")
    # **END OF ADDED BLOCK**
    
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

    # **ADD THIS BLOCK**
    model_path = os.path.join(model_dir, 'lstm_model.keras')
    model.save(model_path)
    print(f"LSTM model saved successfully to {model_path}")
    # **END OF ADDED BLOCK**
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invert predictions to original scale
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Generate the test forecast plot
    test_predict_plot = np.empty_like(tsla_prices)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(tsla_prices) - 1, :] = test_predict

    plt.figure(figsize=(15, 6))
    plt.plot(tsla_df.index, tsla_prices, label='Actual Prices')
    plt.plot(tsla_df.index, test_predict_plot, label='LSTM Test Forecast', color='orange')
    plt.title("TSLA Stock Price: LSTM Forecast vs. Actual")
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
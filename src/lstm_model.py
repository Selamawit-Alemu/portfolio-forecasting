# src/lstm_model.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model as KerasModel

def prepare_lstm_data(
    series: pd.Series,
    lookback: int = 60
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scales a time series and prepares it for an LSTM model using a sliding window.

    This function transforms a 1D time series into a 3D tensor suitable for
    training an LSTM. It also returns the fitted scaler for inverse transformation
    of future predictions.

    Args:
        series (pd.Series): The time series data to be prepared.
        lookback (int): The number of previous time steps to use as input
                        for a single prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
            - X (np.ndarray): 3D array of shape (samples, lookback, 1)
            - y (np.ndarray): 1D array of shape (samples,)
            - scaler (MinMaxScaler): The fitted scaler object.
    """
    data = series.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    # Reshape X for LSTM input: (samples, lookback, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_and_train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.1
) -> KerasModel:
    """
    Builds, compiles, and trains a Sequential LSTM model.

    The model architecture consists of two LSTM layers with dropout, and a
    Dense output layer. It uses an EarlyStopping callback to prevent overfitting.

    Args:
        X_train (np.ndarray): The 3D training features.
        y_train (np.ndarray): The 1D training targets.
        epochs (int): Number of epochs to train.
        batch_size (int): Training batch size.
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        KerasModel: The trained Keras model object.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training LSTM model...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model

def forecast_lstm(
    model: KerasModel,
    data: np.ndarray,
    n_periods: int,
    scaler: MinMaxScaler
) -> pd.Series:
    """
    Generates a forecast for future time steps using a trained LSTM model.

    This function predicts the next n_periods, using the last `lookback`
    values from the provided data as the initial input.

    Args:
        model (KerasModel): The trained LSTM model.
        data (np.ndarray): The last `lookback` values from the training data.
        n_periods (int): The number of future periods to forecast.
        scaler (MinMaxScaler): The scaler used to transform the data.

    Returns:
        pd.Series: A Series containing the inverse-transformed forecast.
    """
    forecast = []
    current_input = data.reshape(1, data.shape[0], 1)

    for _ in range(n_periods):
        # Predict the next value
        predicted_value_scaled = model.predict(current_input, verbose=0)[0][0]
        
        # Add the prediction to the list
        forecast.append(predicted_value_scaled)
        
        # Update the input sequence for the next prediction
        new_input = np.append(current_input[:, 1:, :], [[predicted_value_scaled]], axis=1)
        current_input = new_input
        
    # Inverse transform the forecast to original scale
    forecast_scaled = np.array(forecast).reshape(-1, 1)
    forecast_original_scale = scaler.inverse_transform(forecast_scaled)

    return pd.Series(forecast_original_scale.flatten())

if __name__ == "__main__":
    # --- Example Usage ---
    # This block demonstrates how the functions would be used in a full pipeline.
    print("--- LSTM Model Workflow Example ---")

    # 1. Create dummy time series data
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    prices = [100 + np.sin(i / 10) * 10 + np.random.randn() for i in range(200)]
    dummy_data = pd.DataFrame({'Close': prices}, index=dates)

    # 2. Define train/test split
    split_date = '2020-06-01'
    train_series = dummy_data['Close'][dummy_data.index < split_date]
    test_series = dummy_data['Close'][dummy_data.index >= split_date]
    
    # 3. Prepare data for LSTM
    print("\nPreparing data...")
    lookback_window = 30
    X_train, y_train, scaler = prepare_lstm_data(train_series, lookback=lookback_window)
    X_test, y_test_scaled, _ = prepare_lstm_data(test_series, lookback=lookback_window)

    # 4. Build and train model
    print("\nBuilding and training LSTM model...")
    model = build_and_train_lstm_model(X_train, y_train)

    # 5. Forecast
    print("\nForecasting on test data...")
    # Get the last 'lookback' values from the training set to start the forecast
    initial_forecast_input = X_test[0].flatten()
    forecasted_values = forecast_lstm(model, initial_forecast_input, len(y_test_scaled), scaler)
    
    print("\nForecasted Values (first 5):")
    print(forecasted_values.head())
    print("\nActual Values (first 5):")
    print(scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()[:5])

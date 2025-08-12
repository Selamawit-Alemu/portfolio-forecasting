import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_lstm_data(df, feature_col='Close', lookback=60):
    """
    Prepare data for LSTM:
    - Scale the feature column between 0 and 1
    - Create sequences of length `lookback` for features and targets

    Args:
        df (pd.DataFrame): DataFrame with DateTimeIndex and feature_col present
        feature_col (str): Column to use as feature (default 'Close')
        lookback (int): Number of previous time steps to use as input

    Returns:
        X (np.array): 3D array (samples, lookback, features)
        y (np.array): 1D array (samples,)
        scaler (MinMaxScaler): fitted scaler to inverse transform later
    """
    data = df[[feature_col]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def create_train_test_lstm(train_df, test_df, feature_col='Close', lookback=60):
    """
    Prepare LSTM train/test datasets from train and test DataFrames.

    Args:
        train_df (pd.DataFrame): training data with DateTimeIndex
        test_df (pd.DataFrame): testing data with DateTimeIndex
        feature_col (str): column for feature (default 'Close')
        lookback (int): lookback window size

    Returns:
        X_train, y_train, scaler_train, X_test, y_test
    """
    X_train, y_train, scaler_train = prepare_lstm_data(train_df, feature_col, lookback)

    test_scaled = scaler_train.transform(test_df[[feature_col]].values)
    X_test, y_test = [], []
    for i in range(lookback, len(test_scaled)):
        X_test.append(test_scaled[i - lookback:i, 0])
        y_test.append(test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, scaler_train, X_test, y_test


def build_lstm_model(input_shape, units=50, dropout=0.2):
    """
    Build LSTM model.

    Args:
        input_shape (tuple): shape of input data (timesteps, features)
        units (int): number of LSTM units
        dropout (float): dropout rate

    Returns:
        model (keras.Model)
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units=units),
        Dropout(dropout),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, val_split=0.1):
    """
    Train LSTM model with EarlyStopping.

    Args:
        model (keras.Model): compiled LSTM model
        X_train (np.array): training features
        y_train (np.array): training targets
        epochs (int): number of epochs to train
        batch_size (int): batch size
        val_split (float): validation split ratio

    Returns:
        history (keras.callbacks.History)
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=2
    )
    return history


def predict_and_plot(model, X_test, y_test, scaler, title='LSTM Forecast vs Actual'):
    """
    Predict on test data, inverse scale, plot predictions vs actuals, and return performance metrics.

    Args:
        model (keras.Model): trained model
        X_test (np.array): test features
        y_test (np.array): true test targets
        scaler (MinMaxScaler): scaler used for inverse transformation
        title (str): plot title

    Returns:
        dict: performance metrics {'MAE', 'RMSE'}
    """
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Corrected line: Calculate RMSE manually by taking the square root of MSE

    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    return {'MAE': mae, 'RMSE': rmse}
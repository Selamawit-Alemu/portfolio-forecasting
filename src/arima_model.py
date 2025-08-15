# src/arima_model.py

import pmdarima as pm
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple

def check_stationarity(series: pd.Series) -> bool:
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    A key assumption of ARIMA models is that the time series is stationary.
    This function helps to verify that condition.

    Args:
        series (pd.Series): The time series data to test.

    Returns:
        bool: True if the series is stationary (p-value <= 0.05), False otherwise.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    
    if p_value <= 0.05:
        print(f"ADF p-value: {p_value:.4f}. Series is likely stationary.")
        return True
    else:
        print(f"ADF p-value: {p_value:.4f}. Series is likely non-stationary.")
        return False

def fit_arima_model(train_series: pd.Series, seasonal: bool = False) -> pm.ARIMA:
    """
    Automatically fits an optimal ARIMA model to the training data.

    This function uses pmdarima's auto_arima to find the best-fitting ARIMA
    parameters (p, d, q) based on an information criterion (AIC, BIC, etc.).

    Args:
        train_series (pd.Series): The training data for the ARIMA model.
        seasonal (bool): Whether to fit a seasonal ARIMA model. Defaults to False.

    Returns:
        pm.ARIMA: The fitted ARIMA model object.
    """
    model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        stepwise=True,
        suppress_warnings=True,
        trace=False  # Set to True for verbose output
    )
    print("ARIMA model summary:")
    print(model.summary())
    return model

def forecast_arima(model: pm.ARIMA, n_periods: int) -> pd.DataFrame:
    """
    Generates a forecast and confidence intervals using a fitted ARIMA model.

    Args:
        model (pm.ARIMA): The fitted ARIMA model object.
        n_periods (int): The number of periods to forecast into the future.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast, lower bound,
                      and upper bound of the confidence interval.
    """
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_df = pd.DataFrame(
        {'forecast': forecast, 'lower_bound': conf_int[:, 0], 'upper_bound': conf_int[:, 1]}
    )
    return forecast_df

if __name__ == "__main__":
    # Example usage block to demonstrate the functions
    print("--- Example ARIMA Model Workflow ---")
    
    # 1. Create dummy data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({'price': [100 + i + (i % 5) for i in range(100)]}, index=dates)
    train_data = dummy_data['price'][:80]
    test_data = dummy_data['price'][80:]

    # 2. Check stationarity (Task 1 from project instructions)
    print("\nChecking stationarity of the series...")
    is_stationary = check_stationarity(train_data)
    if not is_stationary:
        print("Series is non-stationary. Differencing is required.")

    # 3. Fit ARIMA model (Task 2)
    print("\nFitting ARIMA model...")
    fitted_model = fit_arima_model(train_data)

    # 4. Forecast (Task 3)
    print("\nForecasting future prices...")
    n_forecast_periods = len(test_data)
    forecast_results = forecast_arima(fitted_model, n_periods=n_forecast_periods)
    print("Forecast results (first 5 periods):")
    print(forecast_results.head())
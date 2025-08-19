import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity 
    of a given time series.

    Args:
        series (pd.Series): The time series data to test.

    Returns:
        tuple: ADF test results, containing:
            - test statistic
            - p-value
            - number of lags used
            - number of observations used
            - critical values for significance levels
            - maximized information criterion value

    Prints:
        - ADF statistic
        - p-value
        - Stationarity conclusion based on p-value (threshold = 0.05).
    """
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("Series is non-stationary; differencing is recommended.")
    else:
        print("Series is stationary.")
    return result


def fit_arima_model(train_series, seasonal=False, stepwise=True, suppress_warnings=True):
    """
    Fit an ARIMA (or SARIMA) model using pmdarima's auto_arima.

    Args:
        train_series (pd.Series): Training time series data.
        seasonal (bool, optional): Whether to include seasonal terms. Default is False.
        stepwise (bool, optional): Whether to use the stepwise algorithm for model selection. Default is True.
        suppress_warnings (bool, optional): Whether to suppress warnings during fitting. Default is True.

    Returns:
        pmdarima.arima.ARIMA: The fitted ARIMA model.

    Prints:
        - Model summary including coefficients and diagnostics.
    """
    model = pm.auto_arima(
        train_series,
        seasonal=seasonal,
        stepwise=stepwise,
        suppress_warnings=suppress_warnings,
        trace=True
    )
    print(model.summary())
    model.fit(train_series)
    return model


def forecast_and_plot(model, train_series, test_series):
    """
    Forecast values using a fitted ARIMA model and visualize performance 
    against actual test data.

    Args:
        model (pmdarima.arima.ARIMA): Fitted ARIMA model.
        train_series (pd.Series): Training time series data.
        test_series (pd.Series): Testing/validation time series data.

    Displays:
        Matplotlib plot comparing:
            - Training data
            - Test data
            - Forecasted values with confidence intervals
    """
    n_periods = len(test_series)
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

    plt.figure(figsize=(14,6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test')
    plt.plot(test_series.index, forecast, label='Forecast')
    plt.fill_between(test_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title('ARIMA Forecast vs Actual')
    plt.show()

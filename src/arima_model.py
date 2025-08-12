import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def check_stationarity(series):
    """
    Perform Augmented Dickey-Fuller test to check stationarity of a time series.
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
    Fits an ARIMA model using auto_arima on the training series.
    Returns the fitted model.
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
    Forecasts using the fitted model over test series period and plots train, test and forecast.
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

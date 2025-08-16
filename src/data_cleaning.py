# src/data_cleaning.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the financial time series DataFrame by handling missing values and
    ensuring correct data types for financial analysis.
    
    Transformations applied:
    - Fills missing 'Adj Close' with 'Close' if the former is absent.
    - Removes rows with non-numeric price placeholders.
    - Converts relevant columns (Open, High, Low, Close, Adj Close, Volume) to numeric.
    - Sets the 'Date' column as a proper datetime index.
    - Handles missing values using forward-fill (ffill), a standard practice for time-series data.
    - Drops any remaining NaNs at the beginning of the series that couldn't be filled.
    
    Args:
        df (pd.DataFrame): The raw DataFrame from the CSV file.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with a datetime index.
    """
    if 'Adj Close' not in df.columns:
        print("Warning: 'Adj Close' column not found. Using 'Close' for cleaning.")
        df['Adj Close'] = df['Close']

    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
    
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')

    df = df[df.index.notna()]
    df = df.ffill()
    df = df.dropna()

    return df

def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the daily log returns from the 'Adj Close' price series.
    Log returns are used for financial analysis as they are stationary.
    
    Transformations applied:
    - Calculates log returns as log(P_t / P_{t-1}).
    - Removes the first row which will have a NaN value.
    - Replaces infinite values (from zero prices) with NaN and drops them.
    
    Args:
        df (pd.DataFrame): The cleaned financial DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with a new 'Log_Returns' column.
    """
    if 'Adj Close' not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column to calculate returns.")

    df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def adf_test(series: pd.Series, title=''):
    """
    Performs and prints the Augmented Dickey-Fuller (ADF) test for stationarity.
    
    Transformation applied:
    - This is a statistical test, not a data transformation, used to prove
      that the log returns series is stationary, a key prerequisite for
      most time series forecasting models.
    """
    print(f'Augmented Dickey-Fuller Test for: {title}')
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Statistic: {result[0]:.2f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.2f}')
    if result[1] <= 0.05:
        print("Result: The series is stationary.\n")
    else:
        print("Result: The series is NOT stationary.\n")

def calculate_var_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculates Value at Risk (VaR) and the Sharpe Ratio.
    
    Transformations applied:
    - VaR is calculated as the negative of the 5th percentile of the returns.
    - Sharpe Ratio annualizes the daily mean and standard deviation of returns.
    
    Args:
        returns (pd.Series): A Series of log returns.
        risk_free_rate (float): The risk-free rate of return (default is 0.0 for simplicity).
        
    Returns:
        dict: A dictionary containing 'VaR' and 'Sharpe_Ratio'.
    """
    if returns.empty:
        return {'VaR': np.nan, 'Sharpe_Ratio': np.nan}
    
    var_95 = -np.percentile(returns.dropna(), 5)
    
    annualized_returns = returns.mean() * 252
    annualized_std = returns.std() * np.sqrt(252)
    
    if annualized_std == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annualized_returns - risk_free_rate) / annualized_std
        
    return {
        'VaR': var_95,
        'Sharpe_Ratio': sharpe_ratio
    }
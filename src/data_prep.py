# src/data_prep.py

import pandas as pd
from typing import Tuple

def split_time_series(
    df: pd.DataFrame, 
    split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a time series DataFrame into training and testing sets based on a date.

    This function assumes the DataFrame's index is a DatetimeIndex. It separates
    the data chronologically into a training set (before the split date) and
    a testing set (on or after the split date).

    Args:
        df (pd.DataFrame): The input time series DataFrame with a DatetimeIndex.
        split_date (str): The cutoff date in 'YYYY-MM-DD' format.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and
                                           testing DataFrames.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")
    
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]

    return train_df, test_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates key financial features from a DataFrame of price data.

    Features include daily returns and a 21-day rolling volatility.

    Args:
        df (pd.DataFrame): A DataFrame containing 'Close' prices.

    Returns:
        pd.DataFrame: The original DataFrame with new 'Daily_Return' and 
                      'Rolling_Volatility' columns.
    """
    df['Daily_Return'] = df['Close'].pct_change()
    df['Rolling_Volatility'] = df['Daily_Return'].rolling(window=21).std()
    return df

if __name__ == "__main__":
    # This block demonstrates how the functions would be used in practice.
    # It assumes you have a cleaned DataFrame with a DatetimeIndex.

    # Example setup with dummy data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    prices = [100 + i + (i % 5) for i in range(100)]
    dummy_data = pd.DataFrame({'Close': prices}, index=dates)

    # Split the data
    train_set, test_set = split_time_series(dummy_data, split_date='2020-03-01')
    
    print("Train set head:")
    print(train_set.head())
    print("\nTest set head:")
    print(test_set.head())

    # Engineer features
    data_with_features = engineer_features(dummy_data)
    print("\nData with new features:")
    print(data_with_features.head())
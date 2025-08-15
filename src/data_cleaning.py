# src/data_cleaning.py

import pandas as pd
from typing import List

def clean_financial_data(df: pd.DataFrame, price_cols: List[str] = ['Close', 'High', 'Low', 'Open']) -> pd.DataFrame:
    """
    Cleans and preprocesses a financial time series DataFrame.

    This function performs several key cleaning steps:
    1. Removes rows with non-numeric values in the specified price columns.
    2. Converts specified price columns and the 'Volume' column to a numeric type.
    3. Ensures the DataFrame has a proper datetime index.
    4. Handles any remaining missing values by dropping them.

    Args:
        df (pd.DataFrame): The raw DataFrame with financial data.
        price_cols (List[str]): A list of column names that contain price data.
                                 Defaults to ['Close', 'High', 'Low', 'Open'].

    Returns:
        pd.DataFrame: A cleaned DataFrame with a datetime index and numeric data.
    """
    # 1. Convert columns to numeric, coercing errors
    for col in price_cols + ['Volume']:
        # Use .loc to avoid a SettingWithCopyWarning
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            # Drop rows with invalid dates early
            df.loc[:, 'Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date')
        else:
            # If no 'Date' column, try converting the existing index
            df.index = pd.to_datetime(df.index, errors='coerce')
    
    # 3. Drop any rows with NaN values resulting from coercion or missing data
    df = df.dropna(subset=price_cols + ['Volume'])

    return df
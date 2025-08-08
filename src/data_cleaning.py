# src/data_cleaning.py

import pandas as pd

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the financial time series DataFrame:
    - Removes any rows with non-numeric placeholders (e.g., column names)
    - Converts columns to numeric
    - Parses datetime index
    """
    # Remove rows where 'Close' is not a number (e.g., header-like rows)
    df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]

    # Convert all price columns to numeric
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Reset index if needed and convert to datetime
    df = df.reset_index()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date')

    # Drop any remaining NaT or NaN
    df = df.dropna()

    return df

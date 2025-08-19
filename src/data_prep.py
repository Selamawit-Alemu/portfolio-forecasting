import os
import pandas as pd

def split_time_series(df, date_column, split_date, save_path=None):
    """
    Splits a time series DataFrame into training and testing sets based on a date.

    Parameters:
        df (pd.DataFrame): The input time series DataFrame.
        date_column (str): The column name containing datetime information.
        split_date (str): The cutoff date (e.g., '2024-01-01').
        save_path (str, optional): Directory to save the split CSV files.

    Returns:
        (train_df, test_df): Tuple of training and testing DataFrames.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    train_df = df[df[date_column] < split_date]
    test_df = df[df[date_column] >= split_date]

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        train_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(save_path, 'test.csv'), index=False)

    return train_df, test_df

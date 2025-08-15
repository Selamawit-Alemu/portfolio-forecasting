# src/data_loader.py

import yfinance as yf
import pandas as pd
from typing import List, Optional

def fetch_historical_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    Fetches and combines historical financial data for multiple tickers.

    This function downloads historical 'Adj Close' prices for a list of tickers,
    merges them into a single DataFrame, and handles potential errors gracefully.

    Args:
        tickers (List[str]): A list of ticker symbols (e.g., ['TSLA', 'BND', 'SPY']).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        Optional[pd.DataFrame]: A DataFrame with the 'Adj Close' prices for each ticker,
                                or None if the data fetching fails.
    """
    try:
        # Use yf.download to get data for all tickers at once
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker'
        )
        
        # Check if any data was returned
        if data.empty:
            print("Warning: No data found for the specified tickers or dates.")
            return None

        # Extract only the 'Adj Close' prices for each ticker
        adj_close_df = pd.DataFrame()
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                adj_close_df[ticker] = data[ticker]['Adj Close']
            else:
                print(f"Warning: No 'Adj Close' data found for {ticker}. Skipping.")

        # Ensure the index is a DatetimeIndex
        adj_close_df.index = pd.to_datetime(adj_close_df.index)

        return adj_close_df.sort_index()

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    TICKERS = ['TSLA', 'BND', 'SPY']
    START_DATE = '2015-07-01'
    END_DATE = '2025-07-31'

    price_data = fetch_historical_data(TICKERS, START_DATE, END_DATE)

    if price_data is not None:
        print("Data fetched successfully:")
        print(price_data.head())
    else:
        print("Data fetching failed.")
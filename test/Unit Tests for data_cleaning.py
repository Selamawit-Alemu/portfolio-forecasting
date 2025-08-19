import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# Import the functions to be tested.
# Assuming the file is named data_cleaning.py in the same directory.
from data_cleaning import clean_price_data, calculate_log_returns, calculate_var_sharpe

class TestDataCleaning(unittest.TestCase):
    """
    A class to test the functions in the data_cleaning.py module.
    """

    def setUp(self):
        """
        Set up a mock DataFrame for use across all test cases.
        This provides a consistent, predictable dataset to test against.
        """
        self.raw_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Open': [100, 101, 102, 103, 104],
            'High': [102, 103, 104, 105, 106],
            'Low': [99, 100, 101, 102, 103],
            'Adj Close': [100, 101, 103.01, 102.5, 105.0]
        })
        self.raw_data_missing = self.raw_data.copy()
        self.raw_data_missing.loc[2, 'Adj Close'] = np.nan
        self.raw_data_missing.set_index('Date', inplace=True)
        self.raw_data.set_index('Date', inplace=True)

    def test_clean_price_data(self):
        """
        Test that `clean_price_data` correctly cleans the input DataFrame.
        It should handle datetime conversion, set the index, and sort by date.
        """
        cleaned_df = clean_price_data(self.raw_data.copy())
        
        # Check if the index is a DatetimeIndex
        self.assertIsInstance(cleaned_df.index, pd.DatetimeIndex)
        
        # Check if the data is sorted by date
        self.assertTrue(cleaned_df.index.is_monotonic_increasing)
        
        # Check if NaN values are dropped
        cleaned_with_missing = clean_price_data(self.raw_data_missing.copy())
        self.assertEqual(len(cleaned_with_missing), len(self.raw_data_missing) - 1)
        self.assertTrue(pd.notna(cleaned_with_missing).all().all())

    def test_calculate_log_returns(self):
        """
        Test that `calculate_log_returns` correctly calculates log returns.
        It should also drop the first row which will have a NaN return.
        """
        cleaned_df = clean_price_data(self.raw_data.copy())
        returns_df = calculate_log_returns(cleaned_df.copy())
        
        # Check if the 'Log_Returns' column was added
        self.assertIn('Log_Returns', returns_df.columns)
        
        # Check if the first row (with NaN) was dropped
        self.assertEqual(len(returns_df), len(cleaned_df) - 1)
        
        # Check a known calculation for correctness (using the second value)
        # log(101 / 100) ≈ 0.00995
        self.assertAlmostEqual(returns_df['Log_Returns'].iloc[0], np.log(101/100), places=4)

    def test_calculate_var_sharpe(self):
        """
        Test that `calculate_var_sharpe` returns the correct VaR and Sharpe Ratio.
        This test uses a predictable set of returns to verify the calculations.
        """
        # Create a mock series of returns for predictable testing
        mock_returns = pd.Series([-0.02, -0.01, 0.01, 0.02, 0.05])
        
        # Manually calculate expected values
        expected_var = -0.01  # 5th percentile is -0.01
        
        mean_return = mock_returns.mean()
        std_dev = mock_returns.std()
        
        # Mock annual calculations with 252 trading days
        expected_annual_return = mean_return * 252
        expected_annual_std = std_dev * np.sqrt(252)
        expected_sharpe = (expected_annual_return - 0.04) / expected_annual_std

        metrics = calculate_var_sharpe(mock_returns, risk_free_rate=0.04)
        
        self.assertAlmostEqual(metrics['VaR'], expected_var, places=4)
        self.assertAlmostEqual(metrics['Sharpe_Ratio'], expected_sharpe, places=4)

if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
import numpy as np

# Import the functions to be tested.
# Assuming the file is named portfolio_optimization.py in the same directory.
from portfolio_optimization import find_optimal_portfolios

class TestPortfolioOptimization(unittest.TestCase):
    """
    A class to test the functions in the portfolio_optimization.py module.
    """

    def test_find_optimal_portfolios(self):
        """
        Test that `find_optimal_portfolios` correctly identifies the max Sharpe
        and min volatility portfolios from a given set of portfolio data.
        """
        # Create a mock DataFrame that mimics the output of a Monte Carlo simulation.
        # We'll hard-code the optimal values to make the test deterministic.
        portfolio_data = pd.DataFrame({
            'Returns': [0.10, 0.20, 0.30, 0.40, 0.50],
            'Volatility': [0.15, 0.12, 0.13, 0.18, 0.25],
            # Sharpe Ratio = Returns / Volatility (assuming risk-free rate is 0 for simplicity in this mock)
            'Sharpe Ratio': [0.667, 1.667, 2.308, 2.222, 2.0],
            'TSLA_Weight': [0.2, 0.4, 0.6, 0.8, 0.9],
            'BND_Weight': [0.7, 0.5, 0.3, 0.1, 0.05],
            'SPY_Weight': [0.1, 0.1, 0.1, 0.1, 0.05],
        })

        # The max Sharpe portfolio should be at index 2 (Sharpe = 2.308)
        # The min volatility portfolio should be at index 1 (Volatility = 0.12)
        
        # Call the function to be tested
        optimal_portfolios = find_optimal_portfolios(portfolio_data)
        
        # Check that the max Sharpe portfolio is correctly identified
        max_sharpe_portfolio = optimal_portfolios['Max_Sharpe_Portfolio']
        self.assertAlmostEqual(max_sharpe_portfolio['Sharpe Ratio'], 2.308, places=3)
        self.assertAlmostEqual(max_sharpe_portfolio['Returns'], 0.30, places=3)

        # Check that the min volatility portfolio is correctly identified
        min_vol_portfolio = optimal_portfolios['Min_Volatility_Portfolio']
        self.assertAlmostEqual(min_vol_portfolio['Volatility'], 0.12, places=3)
        self.assertAlmostEqual(min_vol_portfolio['Returns'], 0.20, places=3)
        
    def test_find_optimal_portfolios_edge_cases(self):
        """
        Test that the function handles edge cases gracefully, such as empty or
        single-row DataFrames.
        """
        # Test Case 1: Empty DataFrame
        empty_df = pd.DataFrame(columns=['Returns', 'Volatility', 'Sharpe Ratio'])
        optimal_empty = find_optimal_portfolios(empty_df)
        self.assertIsNone(optimal_empty['Max_Sharpe_Portfolio'])
        self.assertIsNone(optimal_empty['Min_Volatility_Portfolio'])

        # Test Case 2: Single-row DataFrame
        single_row_df = pd.DataFrame({
            'Returns': [0.25],
            'Volatility': [0.10],
            'Sharpe Ratio': [2.5]
        })
        optimal_single = find_optimal_portfolios(single_row_df)
        self.assertAlmostEqual(optimal_single['Max_Sharpe_Portfolio']['Sharpe Ratio'], 2.5)
        self.assertAlmostEqual(optimal_single['Min_Volatility_Portfolio']['Volatility'], 0.10)
        
if __name__ == '__main__':
    unittest.main()

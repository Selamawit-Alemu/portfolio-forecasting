import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sys
import os

# Assuming your src directory is correctly in the path.
# This ensures Streamlit can find your data_cleaning.py script.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

# Import your custom functions
from data_cleaning import clean_price_data, calculate_log_returns, calculate_var_sharpe

# --- Data Loading and Preparation (run once and cache) ---
@st.cache_data
def load_and_process_data():
    """
    Loads raw data, cleans it, calculates returns and metrics.
    Caches the output to improve performance on subsequent runs.
    """
    try:
        tsla_raw = pd.read_csv('data/raw/TSLA.csv')
        bnd_raw = pd.read_csv('data/raw/BND.csv')
        spy_raw = pd.read_csv('data/raw/SPY.csv')

        tsla_cleaned = clean_price_data(tsla_raw)
        bnd_cleaned = clean_price_data(bnd_raw)
        spy_cleaned = clean_price_data(spy_raw)
        
        # Combine the cleaned 'Adj Close' prices into a single DataFrame
        # This is where we are explicit about using 'Adj Close'
        all_close_prices_df = pd.DataFrame({
            'TSLA': tsla_cleaned['Adj Close'],
            'BND': bnd_cleaned['Adj Close'],
            'SPY': spy_cleaned['Adj Close']
        }).dropna()
        
        # Normalize the combined price DataFrame to a base value of 1
        # Now, normalized_prices_df is explicitly the normalized closing prices
        normalized_prices_df = all_close_prices_df / all_close_prices_df.iloc[0]

        tsla_returns = calculate_log_returns(tsla_cleaned.copy())
        bnd_returns = calculate_log_returns(bnd_cleaned.copy())
        spy_returns = calculate_log_returns(spy_cleaned.copy())
        
        # Combine log returns into a single DataFrame for unified plots
        all_returns_df = pd.DataFrame({
            'TSLA': tsla_returns['Log_Returns'],
            'BND': bnd_returns['Log_Returns'],
            'SPY': spy_returns['Log_Returns']
        }).dropna()
        
        # Calculate key metrics
        tsla_metrics = calculate_var_sharpe(tsla_returns['Log_Returns'])
        bnd_metrics = calculate_var_sharpe(bnd_returns['Log_Returns'])
        spy_metrics = calculate_var_sharpe(spy_returns['Log_Returns'])
        
        return tsla_metrics, bnd_metrics, spy_metrics, all_returns_df, normalized_prices_df

    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'TSLA.csv', 'BND.csv', and 'SPY.csv' are in the 'data/raw' directory.")
        return None, None, None, None, None

tsla_metrics, bnd_metrics, spy_metrics, all_returns_df, normalized_prices_df = load_and_process_data()

# Check if data loaded successfully
if tsla_metrics is None:
    st.stop()

# --- Dashboard Layout ---
st.set_page_config(layout="wide", page_title="Portfolio Forecasting EDA")

st.title("Financial Portfolio EDA Dashboard 📈")

st.markdown("""
This dashboard provides an exploratory data analysis of three key assets: **Tesla (TSLA)**, the **Vanguard Total Bond Market ETF (BND)**, and the **S&P 500 ETF (SPY)**.
The analysis focuses on understanding the risk and return characteristics of each asset.
""")

st.header("Key Insights from the Analysis")
st.markdown("""
Based on the data and metrics, here are the key takeaways for each asset:

* **TSLA (Tesla):** The **most volatile** asset with the highest daily risk. Its high VaR indicates a significant potential for daily losses.
* **SPY (S&P 500 ETF):** Provides the **best risk-adjusted return**. With the highest Sharpe Ratio, SPY has historically delivered the most return for each unit of risk taken.
* **BND (Total Bond Market ETF):** A **low-risk, low-volatility** asset. Its low VaR and low returns make it a stable component for a diversified portfolio.
""")

st.write("---") 

st.header("Risk and Return Metrics")

# Create columns for a clean, side-by-side display of metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("TSLA")
    st.metric("VaR (95%)", f"{tsla_metrics['VaR']:.2%}")
    st.metric("Sharpe Ratio", f"{tsla_metrics['Sharpe_Ratio']:.2f}")

with col2:
    st.subheader("BND")
    st.metric("VaR (95%)", f"{bnd_metrics['VaR']:.2%}")
    st.metric("Sharpe Ratio", f"{bnd_metrics['Sharpe_Ratio']:.2f}")

with col3:
    st.subheader("SPY")
    st.metric("VaR (95%)", f"{spy_metrics['VaR']:.2%}")
    st.metric("Sharpe Ratio", f"{spy_metrics['Sharpe_Ratio']:.2f}")

st.write("---")

st.header("Interactive Visuals")

# Normalized Closing Price Plot
st.subheader("Normalized Closing Price Performance")
st.markdown("This chart shows the performance of each asset starting from a normalized base of $1.00, allowing for a direct comparison of growth over time.")

fig_normalized = px.line(
    normalized_prices_df,
    title="Normalized Closing Price Performance Over Time",
    labels={"value": "Normalized Price", "index": "Date"}
)
st.plotly_chart(fig_normalized, use_container_width=True)


# Daily Log Returns Plot
st.subheader("Daily Log Returns")
fig_returns = px.line(all_returns_df, title="Daily Log Returns Over Time")
st.plotly_chart(fig_returns, use_container_width=True)

# Returns Distribution Plot
st.subheader("Returns Distribution")
fig_hist = px.histogram(all_returns_df, barmode='overlay', nbins=50, title="Distribution of Daily Log Returns")
fig_hist.update_layout(xaxis_title="Log Returns", yaxis_title="Frequency")
st.plotly_chart(fig_hist, use_container_width=True)

# Correlation Heatmap
st.subheader("Asset Correlation Heatmap")
st.markdown("A correlation of **+1.0** means assets move in the same direction, while **-1.0** means they move in opposite directions. For diversification, you want to combine assets that are not perfectly correlated.")

correlation_matrix = all_returns_df.corr()
fig_corr = px.imshow(
    correlation_matrix,
    x=correlation_matrix.columns,
    y=correlation_matrix.index,
    color_continuous_scale='Viridis',
    title='Interactive Correlation Heatmap'
)
for i, row in enumerate(correlation_matrix.values):
    for j, val in enumerate(row):
        fig_corr.add_annotation(
            x=correlation_matrix.columns[j],
            y=correlation_matrix.index[i],
            text=f'{val:.2f}',
            showarrow=False,
            font=dict(color='black')
        )
st.plotly_chart(fig_corr, use_container_width=True)

# app.py (continued from previous turn)

st.subheader("Rolling Volatility (Risk)")
st.markdown("This chart shows the 30-day rolling standard deviation of daily log returns. It's a key measure of an asset's risk, allowing you to see how volatility changes over time.")

# Calculate the 30-day rolling standard deviation for each asset's log returns
rolling_volatility = all_returns_df.rolling(window=30).std() * np.sqrt(252) # Annualize the volatility

# Create the interactive plot
fig_vol = px.line(
    rolling_volatility,
    title="30-Day Rolling Annualized Volatility",
    labels={"value": "Annualized Volatility (%)", "index": "Date"}
)
st.plotly_chart(fig_vol, use_container_width=True)



st.subheader("Cumulative Returns")
st.markdown("This chart shows the hypothetical growth of an initial investment in each asset, reflecting the total compounded return over the period.")

# Calculate cumulative returns from the log returns
# Exponentiate the log returns and compute the cumulative product
cumulative_returns_df = np.exp(all_returns_df.cumsum())

# Create an interactive line chart with Plotly
fig_cumulative = px.line(
    cumulative_returns_df,
    title="Cumulative Returns of Portfolio Assets",
    labels={"value": "Cumulative Return", "index": "Date"}
)
st.plotly_chart(fig_cumulative, use_container_width=True)

st.subheader("Returns Distribution: Box Plot")
st.markdown("A box plot provides a clear, side-by-side comparison of the distribution of returns for each asset, including the median, quartiles, and outliers.")

# Create the interactive box plot
fig_box = px.box(
    all_returns_df,
    title="Returns Distribution by Asset",
    labels={"value": "Daily Log Returns"}
)
st.plotly_chart(fig_box, use_container_width=True)

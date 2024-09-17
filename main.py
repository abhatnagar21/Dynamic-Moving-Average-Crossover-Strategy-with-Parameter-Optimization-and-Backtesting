import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Helper functions for metrics
def calculate_sharpe(returns, risk_free_rate=0):
    # Sharpe Ratio: (mean return - risk free rate) / std deviation of returns
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)  # Annualized

def calculate_drawdown(cumulative_returns):
    # Drawdown: Calculate percentage drawdown from peak
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_cagr(final_value, initial_value, n_years):
    # CAGR: Compound Annual Growth Rate
    return (final_value / initial_value) ** (1 / n_years) - 1

# Download stock data
ticker = 'AAPL'
data = yf.download(ticker, start="2015-01-01", end="2023-01-01")

# Define initial balance and parameters
initial_balance = 100000
params = {'short_window': [20, 50, 100], 'long_window': [100, 200, 300]}
best_performance = -float('inf')
best_params = None
n_years = (data.index[-1] - data.index[0]).days / 365.25

# Grid search for best parameters
for param in ParameterGrid(params):
    # Calculate short and long-term moving averages
    data['SMA_short'] = data['Close'].rolling(window=param['short_window']).mean()
    data['SMA_long'] = data['Close'].rolling(window=param['long_window']).mean()

    # Generate buy/sell signals
    data['Signal'] = 0
    data['Signal'] = np.where(data['SMA_short'] > data['SMA_long'], 1, -1)

    # Strategy returns
    data['Returns'] = data['Close'].pct_change()
    data['Strategy Returns'] = data['Signal'].shift(1) * data['Returns']

    # Calculate cumulative returns
    data['Cumulative Strategy Returns'] = (1 + data['Strategy Returns']).cumprod() * initial_balance

    # Final portfolio value
    final_portfolio_value = data['Cumulative Strategy Returns'].iloc[-1]

    # Calculate performance metrics
    sharpe_ratio = calculate_sharpe(data['Strategy Returns'])
    max_drawdown = calculate_drawdown(data['Cumulative Strategy Returns'])
    cagr = calculate_cagr(final_portfolio_value, initial_balance, n_years)

    # Check if this is the best performance
    if final_portfolio_value > best_performance:
        best_performance = final_portfolio_value
        best_params = param
        best_sharpe = sharpe_ratio
        best_drawdown = max_drawdown
        best_cagr = cagr

# Print the best parameters and their performance
print(f"Best Parameters: {best_params}")
print(f"Best Final Portfolio Value: ${best_performance:.2f}")
print(f"Best Sharpe Ratio: {best_sharpe:.2f}")
print(f"Best Maximum Drawdown: {best_drawdown:.2%}")
print(f"Best CAGR: {best_cagr:.2%}")

# Visualization
fig = go.Figure()

# Plot stock price and SMAs
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_short'], name=f"SMA {best_params['short_window']}", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_long'], name=f"SMA {best_params['long_window']}", line=dict(color='green')))

# Highlight buy/sell signals
buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

# Update layout
fig.update_layout(title=f'{ticker} Strategy Performance', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
fig.show()

# Portfolio value over time
plt.figure(figsize=(10,6))
plt.plot(data.index, data['Cumulative Strategy Returns'], label='Strategy Returns', color='blue')
plt.title(f"Portfolio Value Over Time - {ticker}")
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid()
plt.show()

# Performance metrics
print(f'Final Portfolio Value: ${final_portfolio_value:.2f}')
print(f'Sharpe Ratio: {best_sharpe:.2f}')
print(f'Max Drawdown: {best_drawdown:.2%}')
print(f'CAGR: {best_cagr:.2%}')

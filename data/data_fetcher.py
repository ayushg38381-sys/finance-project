import yfinance as yf
import pandas as pd

tickers = ["AAPL","MSFT","NVDA"]

data = yf.download(tickers,start="2020-01-01",end="2025-01-01")

close_prices = data["Close"]

print("Close Price")
print(close_prices.head())

returns=close_prices.pct_change()
print("\nDaily Returns")
print(returns.head())
volatility = returns.std()
print("\nVolatility")
print(volatility)
correlation_matrix = returns.corr()
print("\ncorrelation Matrix")
print(correlation_matrix)
weights = [0.4,0.3,0.3]
portfolio_returns = (returns["AAPL"] * 0.4 + returns["MSFT"] * 0.3 +returns["NVDA"] * 0.3)

print("\nPortfolio Returns")
print(portfolio_returns.head())
portfolio_volatility = portfolio_returns.std()

print("\nPortfolio Volatility")
print(portfolio_volatility)
annual_return = portfolio_returns.mean() * 252
print("\nAnnual Expected Return")
print(annual_return)
annual_volatility = portfolio_volatility * (252 ** 0.5)

print("\nAnnual Volatility")
print(annual_volatility)
risk_free_rate = 0.04

sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
print("\nSharpe Ratio")
print(sharpe_ratio)
cumulative_returns = (1 + portfolio_returns).cumprod()
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min()
print("\nMaximum Drawdown")
print(max_drawdown)
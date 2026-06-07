import numpy as np
from scipy.stats import norm

portfolio_mean = portfolio_returns.mean()

print("\nDaily Mean Return")
print(portfolio_mean)
z_95 = norm.ppf(0.95)

var_95 = (
    z_95 * portfolio_volatility
    - portfolio_mean
)

print("\n95% VaR")
print(var_95)
z_99 = norm.ppf(0.99)

var_99 = (
    z_99 * portfolio_volatility
    - portfolio_mean
)

print("\n99% VaR")
print(var_99)
historical_var_95 = portfolio_returns.quantile(0.05)
historical_var_99 = portfolio_returns.quantile(0.01)
print("\nHistorical VaR 95%")
print(historical_var_95)

print("\nHistorical VaR 99%")
print(historical_var_99)
cvar_95 = portfolio_returns[
    portfolio_returns <= historical_var_95
].mean()

cvar_99 = portfolio_returns[
    portfolio_returns <= historical_var_99
].mean()
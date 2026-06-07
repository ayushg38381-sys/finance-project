import yfinance as yf
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

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
benchmark = yf.download(
    "SPY" ,
    start="2020-01-01",
    end="2025-01-01"
)
benchmark_close = benchmark["Close"]
benchmark_returns = benchmark_close.pct_change().squeeze()
print("\nBenchmark Returns")
print(benchmark_returns.head())
covariance = portfolio_returns.cov(benchmark_returns)
market_variance = benchmark_returns.var()
beta = covariance / market_variance
print("\nPortfolio Beta")
print(beta)
print(portfolio_returns.shape)
print(benchmark_returns.shape)
market_return = benchmark_returns.mean() * 252
risk_free_rate = 0.04
expected_return = ( risk_free_rate
    + beta * (market_return - risk_free_rate))
alpha = annual_return - expected_return
print("\nMarket Return")
print(market_return)
print("\nPortfolio Alpha")
print(alpha)
cov_matrix = returns.cov()
print("\nCovariance Matrix")
print(cov_matrix)
import numpy as np

weights = np.array([ 0.4, 0.3, 0.3])
portfolio_variance = np.dot(
    weights.T,
    np.dot(cov_matrix, weights)
)
print("\nPortfolio Variance")
print(portfolio_variance)
portfolio_std_matrix = np.sqrt(portfolio_variance)
print("\nPortfolio std From Matrix")
print(portfolio_std_matrix)
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

print("\nCVaR 95%")
print(cvar_95)

print("\nCVaR 99%")
print(cvar_99)
num_days = 30

random_returns = np.random.normal(
    portfolio_mean,
    portfolio_volatility,
    num_days
)

print("\nRandom Future Returns")
print(random_returns[:5])
initial_value = 100
future_path = initial_value * np.cumprod(
    1 + random_returns
)
print("\nFuture Portfolio Path")
print(future_path[:5])
num_simulations = 1000
num_days = 30

final_values = []
for i in range(num_simulations):
    random_returns = np.random.normal(
        portfolio_mean,
        portfolio_volatility,
        num_days
    )
    future_path = 100 * np.cumprod(
        1 + random_returns
    )
    final_values.append(
    future_path[-1]
    )
final_values = np.array(final_values)
print(len(final_values))
print("\nFirst 10 Final Portfolio Values")
print(final_values[:10])
print("\nWorst Outcome")
print(min(final_values))

print("\nBest Outcome")
print(max(final_values))

print("\nAverage Outcome")
print(np.mean(final_values))
mc_var_95 = np.percentile(final_values, 5)
mc_var_99 = np.percentile(final_values, 1)
print("\nMonte Carlo VaR 95%")
print(mc_var_95)

print("\nMonte Carlo VaR 99%")
print(mc_var_99)
mc_cvar_95 = np.mean(
    [x for x in final_values if x <= mc_var_95]
)

mc_cvar_99 = np.mean(
    [x for x in final_values if x <= mc_var_99]
)

print("\nMonte Carlo CVaR 95%")
print(mc_cvar_95)

print("\nMonte Carlo CVaR 99%")
print(mc_cvar_99)
crisis_shock = -0.30
stressed_portfolio = 100 * (1 + crisis_shock)
print("\n2008 style stress Test")
print(stressed_portfolio)
stress_scenarios = {
    "2008 Financial Crisis": -0.30,
    "COVID Crash": -0.35,
    "Interest Rate Shock": -0.15,
    "Mild Recession": -0.10
}
print("\nStress Testing Results")
for scenario, shock in stress_scenarios.items():
    stressed_value = 100 * (1+shock)
    print(
        scenario,
        "->",
        stressed_value
    )
    mean_returns = returns.mean()
    def portfolio_performance(weights):
        portfolio_return = np.sum(
            mean_returns * weights
        ) * 252
        portfolio_vol = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    cov_matrix * 252,
                    weights
                )
            )
        )
        sharpe = portfolio_return / portfolio_vol

        return portfolio_return, portfolio_vol, sharpe
test_weights = np.array([
            0.33,
            0.33,
            0.34
        ])
print("\nPortfolio Performance")

print(
    portfolio_performance(
        test_weights
    )
)
def negative_sharpe(weights):
    return -portfolio_performance(weights)[2]
num_assets = len(mean_returns)
constraints = (
    {
        'type':'eq',
        'fun':lambda x: np.sum(x) -1
    }
)
bounds = tuple(
    (0,1)
    for asset in range(num_assets)
)
initial_weights = np.array(
    [1/num_assets] * num_assets
)
optimal = minimize(
    negative_sharpe,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
print("\nOptimal Weights")
print(optimal.x)

print("\nOptimal Portfolio")
print(
    portfolio_performance(
        optimal.x
    )
)
def portfolio_variance(weights):

    return portfolio_performance(weights)[1]
min_var = minimize(
    portfolio_variance,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)
print("\nMinimum Variance Weights")
print(min_var.x)

print("\nMinimum Variance Portfolio")
print(
    portfolio_performance(
        min_var.x
    )
)

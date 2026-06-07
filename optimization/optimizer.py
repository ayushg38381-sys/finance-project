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
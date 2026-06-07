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

def get_monte_carlo_results():
    return{
        "worst_outcome": float(min(final_values)),
        "best_outcome": float(max(final_values)),
        "average_outcome": float(np.mean(final_values)),
        "final_values": final_values.tolist()
    }  

mc_cvar_95 = np.mean(
    [x for x in final_values if x <= mc_var_95]
)

mc_cvar_99 = np.mean(
    [x for x in final_values if x <= mc_var_99]
)
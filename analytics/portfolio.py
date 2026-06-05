import numpy as np

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
def portfolio_variance(weights):

    return portfolio_performance(weights)[1]
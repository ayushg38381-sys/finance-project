from fastapi import FastAPI
from fastapi import Query
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from risk.correlation import calculate_correlation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


app = FastAPI()

def get_returns(stock1, stock2, stock3):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    return returns

@app.get("/stocks")
def get_stocks(
    stock1: str,
    stock2: str,
    stock3: str
):
    return {
        "stocks": [
            stock1,
            stock2,
            stock3
        ]
    }


@app.get("/")
def home():
    return {
        "message": "Portfolio Risk Dashboard API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "project": "Portfolio Risk Dashboard"
    }


@app.get("/portfolio")
def portfolio():

    annual_return = 0.42440596745532777
    volatility = 0.33603654649460546

    sharpe_ratio = annual_return / volatility

    return {
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio
    }


@app.get("/risk")
def risk():

    return {
        "annual_return": 0.42440596745532777,
        "annual_volatility": 0.33603654649460546,
        "sharpe_ratio": 1.1439409536411802,
        "max_drawdown": -0.3997290381851315,
        "beta": 1.3698092025170763,
        "alpha": 0.22600295221779682
    }


@app.get("/var")
def var():

    return {
        "var95": 0.033134625258636474,
        "var99": 0.047560708585146945,
        "historical_var95": -0.03299529878746904,
        "historical_var99": -0.053338393581653376,
        "cvar95": -0.04689983858061192,
        "cvar99": -0.07174915503062163
    }


@app.get("/monte-carlo")
def monte_carlo():

    return {
        "worst_outcome": 75.83447842424923,
        "best_outcome": 150.12046899844827,
        "average_outcome": 105.18055475746797,
        "monte_carlo_var95": 86.99942930441459,
        "monte_carlo_var99": 80.3892616477146,
        "monte_carlo_cvar95": 82.57702318326916,
        "monte_carlo_cvar99": 77.27963513307893
    }


@app.get("/optimization")
def optimization():

    return {
        "optimal_weights": [
            0.130593406,
            0.0,
            0.869406594
        ],
        "expected_return": 0.707099518805297,
        "volatility": 0.4946082407824585,
        "sharpe_ratio": 1.4296153207772728
    }


@app.get("/min-variance")
def min_variance():

    return {
        "weights": [
            0.424618014,
            0.575381986,
            0.0
        ],
        "expected_return": 0.269720611289701,
        "volatility": 0.2901932616125039,
        "sharpe_ratio": 0.9294516688325447
    }

@app.get("/portfolio-data")
def portfolio_data(
    stock1: str,
    stock2: str,
    stock3: str,
):
    
    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    portfolio_returns = returns.mean(axis=1)

    annual_return = portfolio_returns.mean() * 252

    annual_volatility = portfolio_returns.std() * (252**0.5)

    sharpe_ratio = annual_return / annual_volatility

    return {
        "annual_return": float(annual_return),
        "volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio)
    }    
@app.get("/dynamic-risk")
def dynamic_risk(
    stock1: str,
    stock2: str,
    stock3: str
):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    portfolio_returns = returns.mean(axis=1)

    annual_return = portfolio_returns.mean() * 252

    annual_volatility = portfolio_returns.std() * (252**0.5)

    sharpe_ratio = annual_return / annual_volatility

    market_data = yf.download(
    "SPY",
    start="2020-01-01",
    end="2025-01-01"
    )

    market = market_data["Close"].squeeze()

    market_returns = market.pct_change().dropna()

    print("Portfolio shape:", portfolio_returns.shape)
    print("Market shape:", market_returns.shape)
    print("Market type:", type(market_returns))

    common_dates = portfolio_returns.index.intersection(
    market_returns.index
)

    portfolio_aligned = portfolio_returns.loc[common_dates]
    market_aligned = market_returns.loc[common_dates]

    beta = (
        portfolio_aligned.cov(market_aligned)
        / market_aligned.var()
    )

    risk_free_rate = 0.04

    market_return = market_aligned.mean() * 252

    alpha = (
        annual_return
        - (
            risk_free_rate
            + beta * (market_return - risk_free_rate)
        )
    )

    cumulative_returns = (1 + portfolio_returns).cumprod()

    running_max = cumulative_returns.cummax()

    drawdown = (
        cumulative_returns - running_max
    ) / running_max

    max_drawdown = drawdown.min()

    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "beta": float(beta),
        "alpha": float(alpha)
    }
@app.get("/dynamic-var")
def dynamic_var(
    stock1: str,
    stock2: str,
    stock3: str
):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    portfolio_returns = returns.mean(axis=1)

    historical_var_95 = portfolio_returns.quantile(0.05)

    historical_var_99 = portfolio_returns.quantile(0.01)

    cvar_95 = portfolio_returns[
        portfolio_returns <= historical_var_95
    ].mean()

    cvar_99 = portfolio_returns[
        portfolio_returns <= historical_var_99
    ].mean()

    return {

        "var95": float(abs(historical_var_95)),
        "var99": float(abs(historical_var_99)),

        "historical_var95": float(historical_var_95),
        "historical_var99": float(historical_var_99),

        "cvar95": float(cvar_95),
        "cvar99": float(cvar_99)
    }
@app.get("/dynamic-monte-carlo")
def dynamic_monte_carlo(
    stock1: str,
    stock2: str,
    stock3: str
):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    portfolio_returns = returns.mean(axis=1)

    portfolio_mean = portfolio_returns.mean()

    portfolio_volatility = portfolio_returns.std()

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

    return {

        "worst_outcome": float(np.min(final_values)),

        "best_outcome": float(np.max(final_values)),

        "average_outcome": float(np.mean(final_values)),

        "simulated_values":
            final_values.tolist()
    }
@app.get("/dynamic-optimization")
def dynamic_optimization(
    stock1: str,
    stock2: str,
    stock3: str
):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    mean_returns = returns.mean()

    cov_matrix = returns.cov()

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

    def negative_sharpe(weights):

        portfolio_return, portfolio_vol, sharpe = (
            portfolio_performance(weights)
        )

        concentration_penalty = np.sum(
            weights ** 2
        )

        return -sharpe + concentration_penalty

    num_assets = len(mean_returns)

    constraints = (
        {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1
        },
    )

    bounds = tuple(
        (0.1, 0.6)
        for asset in range(num_assets)
    )

    initial_weights = np.array(
        [1/num_assets] * num_assets
    )

    optimal = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    portfolio_return, portfolio_vol, sharpe = (
        portfolio_performance(optimal.x)
    )

    return {
        "optimal_weights": optimal.x.tolist(),
        "expected_return": float(portfolio_return),
        "volatility": float(portfolio_vol),
        "sharpe_ratio": float(sharpe)
    }

@app.get("/efficient-frontier")
def efficient_frontier(
    stock1: str,
    stock2: str,
    stock3: str,
):
    
    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    mean_returns = returns.mean()

    cov_matrix = returns.cov()
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpes = []

    for i in range(5000):
        weights = np.random.random(3)
        weights = weights / np.sum(weights)
        portfolio_return = np.sum(
            mean_returns * weights
        ) * 252
        portfolio_volatility = np.sqrt(
            np.dot(
                weights.T,
                np.dot(
                    cov_matrix * 252,
                    weights
                )
            )
        )
        sharpe = portfolio_return / portfolio_volatility  
        portfolio_returns.append(
            float(portfolio_return)
        )

        portfolio_volatilities.append(
            float(portfolio_volatility)
        )

        portfolio_sharpes.append(
            float(sharpe)
        ) 

    return {
    "returns": portfolio_returns,
    "volatilities": portfolio_volatilities,
    "sharpes": portfolio_sharpes
    }   

@app.get("/correlation")
def correlation(
    stock1: str,
    stock2: str,
    stock3: str
):
    tickers = [
        stock1,
        stock2,
        stock3
    ]
    corr_matrix = calculate_correlation(
        tickers
    )
    return{
        "correlation_matrix":
        corr_matrix.to_dict()
    }
@app.get("/benchmark")
def benchmark(
    stock1: str,
    stock2: str,
    stock3: str
):

    tickers = [stock1, stock2, stock3]

    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    portfolio_returns = returns.mean(axis=1)

    market = yf.download(
        "SPY",
        start="2020-01-01",
        end="2025-01-01"
    )["Close"]

    market_returns = market.pct_change().dropna()

    common_dates = portfolio_returns.index.intersection(
        market_returns.index
    )

    portfolio_returns = portfolio_returns.loc[common_dates]

    market_returns = market_returns.loc[common_dates]

    portfolio_cumulative = (
        1 + portfolio_returns
    ).cumprod()

    market_cumulative = (
        1 + market_returns
    ).cumprod()

    return {
        "dates": portfolio_cumulative.index.astype(str).tolist(),
        "portfolio": portfolio_cumulative.tolist(),
        "market": market_cumulative.squeeze().tolist()
    }

@app.get("/dynamic-stress-test")
def dynamic_stress_test(
    stock1: str,
    stock2: str,
    stock3: str,
    shock1: float,
    shock2: float,
    shock3: float
):
    shock_vector = np.array([
        shock1 / 100,
        shock2 / 100,
        shock3 / 100
    ])

    opt = dynamic_optimization(
    stock1,
    stock2,
    stock3
    ) 
    
    weights = np.array(
    opt["optimal_weights"]
    )

    print("Stocks:", stock1, stock2, stock3)
    print("Weights:", weights)
    print("Shock Vector:", shock_vector)

    portfolio_loss = np.sum(
        weights * shock_vector
    )

    return {
        "portfolio_loss": float(portfolio_loss),
        "remaining_value": float(
            1 + portfolio_loss
        )
    }

@app.get("/market-regime")
def market_regime(
    stock1: str,
    stock2: str,
    stock3: str
):

    returns = get_returns(
        stock1,
        stock2,
        stock3
    )

    portfolio_returns = returns.mean(axis=1)

    volatility = (
        portfolio_returns
        .rolling(20)
        .std()
    )

    momentum = (
        portfolio_returns
        .rolling(20)
        .mean()
    )

    features = pd.DataFrame({
        "return": portfolio_returns,
        "volatility": volatility,
        "momentum": momentum
    }).dropna()

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(
        scaled_features
    )

    current_cluster = int(
        clusters[-1]
    )

    regime_names = {
        0: "Bull Market",
        1: "Bear Market",
        2: "High Volatility",
        3: "Recovery Phase"
    }

    distances = kmeans.transform(
        scaled_features
    )

    latest_distance = distances[-1]

    confidence = (
        1
        - latest_distance.min()
        /
        latest_distance.sum()
    ) * 100

    return {
        "regime":
            regime_names[current_cluster],

        "cluster":
            current_cluster,

        "confidence":
            float(confidence)
    }
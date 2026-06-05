from fastapi import FastAPI

app = FastAPI()

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

    annual_return = 0.4244
    volatility = 0.3360

    sharpe_ratio = (
        annual_return /
        volatility
    )

    return {
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio
    }
@app.get("/square")
def square(x: int):

    return {
        "number": x,
        "square": x * x
    }
@app.get("/expected-return")
def expected_return(
    investment: float,
    annual_return: float
):

    future_value = (
        investment *
        (1 + annual_return)
    )

    return {
        "investment": investment,
        "annual_return": annual_return,
        "future_value": future_value
    }
@app.get("/risk")
def risk():

    return {
        "annual_return": 0.424405,
        "annual_volatility": 0.336036,
        "sharpe_ratio": 1.14394,
        "max_drawdown": -0.39972,
        "beta": 1.3698,
        "alpha": 0.2260
    }
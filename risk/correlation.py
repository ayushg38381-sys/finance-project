import yfinance as yf

def calculate_correlation(tickers):
    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01"
    )

    close_prices = data["Close"]
    returns = close_prices.pct_change().dropna()
    corr_matrix = returns.corr()
    return corr_matrix
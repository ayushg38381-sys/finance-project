import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide"
)

st.title("📊 Portfolio Risk Dashboard")

stock1 = st.text_input(
    "Stock 1",
    "AAPL"
)

stock2 = st.text_input(
    "Stock 2",
    "MSFT"
)

stock3 = st.text_input(
    "Stock 3",
    "NVDA"
)

analyze = st.button("Analyze Portfolio")

if analyze:

    selected_stocks = requests.get(
        f"http://127.0.0.1:8000/stocks?"
        f"stock1={stock1}&"
        f"stock2={stock2}&"
        f"stock3={stock3}"
    ).json()

    st.success(
        f"Selected Stocks: {selected_stocks['stocks']}"
    )



st.title("📊 Portfolio Risk Dashboard")

if analyze:

    portfolio = requests.get(
        f"http://127.0.0.1:8000/portfolio-data?"
        f"stock1={stock1}&"
        f"stock2={stock2}&"
        f"stock3={stock3}"
    ).json()

else:

    portfolio = requests.get(
        "http://127.0.0.1:8000/portfolio"
    ).json()

if analyze:

    risk = requests.get(
        f"http://127.0.0.1:8000/dynamic-risk?"
        f"stock1={stock1}&"
        f"stock2={stock2}&"
        f"stock3={stock3}"
    ).json()

else:

    risk = requests.get(
        "http://127.0.0.1:8000/risk"
    ).json()

if analyze:

    var = requests.get(
        f"http://127.0.0.1:8000/dynamic-var?"
        f"stock1={stock1}&"
        f"stock2={stock2}&"
        f"stock3={stock3}"
    ).json()

else:

    var = requests.get(
        "http://127.0.0.1:8000/var"
    ).json()

monte = requests.get(
    f"http://127.0.0.1:8000/dynamic-monte-carlo?"
    f"stock1={stock1}&"
    f"stock2={stock2}&"
    f"stock3={stock3}"
).json()

opt = requests.get(
    f"http://127.0.0.1:8000/dynamic-optimization?"
    f"stock1={stock1}&"
    f"stock2={stock2}&"
    f"stock3={stock3}"
).json()

benchmark = requests.get(
    f"http://127.0.0.1:8000/benchmark?"
    f"stock1={stock1}&"
    f"stock2={stock2}&"
    f"stock3={stock3}"
).json()

if analyze:

    url = (
        f"http://127.0.0.1:8000/correlation?"
        f"stock1={stock1}&"
        f"stock2={stock2}&"
        f"stock3={stock3}"
    )

    corr = requests.get(url).json()

    

else:
    st.info("Enter stocks and click Analyze Portfolio")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Annual Return",
        round(portfolio["annual_return"], 4)
    )

with col2:
    st.metric(
        "Volatility",
        round(portfolio["volatility"], 4)
    )

with col3:
    st.metric(
        "Sharpe Ratio",
        round(portfolio["sharpe_ratio"], 4)
    )

st.divider()

st.subheader("Risk Metrics")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric(
        "Beta",
        round(risk["beta"], 4)
    )

with r2:
    st.metric(
        "Alpha",
        round(risk["alpha"], 4)
    )

with r3:
    st.metric(
        "Max Drawdown",
        round(risk["max_drawdown"], 4)
    )

st.divider()

st.subheader("Value at Risk")

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.metric("VaR 95%", round(var["var95"], 4))

with v2:
    st.metric("VaR 99%", round(var["var99"], 4))

with v3:
    st.metric("CVaR 95%", round(var["cvar95"], 4))

with v4:
    st.metric("CVaR 99%", round(var["cvar99"], 4))

st.divider()

st.subheader("Monte Carlo Simulation")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Worst Outcome",
        round(monte["worst_outcome"], 2)
    )

with m2:
    st.metric(
        "Average Outcome",
        round(monte["average_outcome"], 2)
    )

with m3:
    st.metric(
        "Best Outcome",
        round(monte["best_outcome"], 2)
    )

st.divider()

st.subheader("Portfolio Optimization")

weights = opt["optimal_weights"]

o1, o2, o3 = st.columns(3)

with o1:
    st.metric(
        f"{stock1} Weight %",
        round(weights[0] * 100, 2)
    )

with o2:
    st.metric(
        f"{stock2} Weight %",
        round(weights[1] * 100, 2)
    )

with o3:
    st.metric(
        f"{stock3} Weight %",
        round(weights[2] * 100, 2)
    )

st.divider()

st.subheader("Optimal Portfolio Performance")

p1, p2, p3 = st.columns(3)

with p1:
    st.metric(
        "Expected Return",
        round(opt["expected_return"], 4)
    )

with p2:
    st.metric(
        "Volatility",
        round(opt["volatility"], 4)
    )

with p3:
    st.metric(
        "Sharpe Ratio",
        round(opt["sharpe_ratio"], 4)
    )

st.divider()

st.subheader("Portfolio vs S&P 500")

benchmark_df = pd.DataFrame({
    "Date": benchmark["dates"],
    "Portfolio": benchmark["portfolio"],
    "S&P500": benchmark["market"]
})

fig = px.line(
    benchmark_df,
    x="Date",
    y=["Portfolio", "S&P500"],
    title="Portfolio Performance vs S&P500"
)

st.plotly_chart(fig)

st.divider()

st.subheader("Portfolio Allocation")

weights = opt["optimal_weights"]

allocation_df = pd.DataFrame({
    "Stock": [stock1, stock2, stock3],
    "Weight": weights
})

fig = px.pie(
    allocation_df,
    names="Stock",
    values="Weight"
)

fig.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

fig.update_layout(
    title="Portfolio Allocation",
     height=600
)

st.plotly_chart(fig)

st.divider()

st.subheader("Monte Carlo Distribution")

simulation_df = pd.DataFrame(
    monte["simulated_values"],
    columns=["value"]
)

hist_fig = px.histogram(
    simulation_df,
    x="value",
    nbins=40,
    title="Distribution of Simulated Portfolio Values"
)
hist_fig.update_layout(
    height=500
)

st.plotly_chart(hist_fig)


st.divider()

st.subheader("Correlation Heatmap")

corr_df = pd.DataFrame(
    corr["correlation_matrix"]
)

heatmap = px.imshow(
    corr_df,
    text_auto=".2f",
    title="Stock Correlation Matrix",
    aspect="auto"
)

heatmap.update_layout(
    width=800,
    height=600
)

st.plotly_chart(heatmap)

st.divider()

st.subheader("🤖 AI Risk Assessment")

risk_level = "High" if risk["beta"] > 1 else "Moderate"

weights = opt["optimal_weights"]

stocks = [stock1, stock2, stock3]

max_index = weights.index(max(weights))

highest_stock = stocks[max_index]

highest_weight = max(weights)

risk_text = f"""
# Portfolio Analysis

- Portfolio Beta: {risk['beta']:.2f}

- Portfolio Alpha: {risk['alpha']:.2f}

- Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}

- Highest Allocation: {highest_stock} ({highest_weight:.1%})

- Risk Level: {risk_level}

- Recommendation:
  Consider reducing concentration risk and increasing diversification.
"""

st.markdown(risk_text)

st.divider()

st.subheader("🤖 AI Portfolio Insights")

weights = opt["optimal_weights"]

stocks = [stock1, stock2, stock3]

max_index = weights.index(max(weights))

highest_stock = stocks[max_index]

highest_weight = max(weights)

st.markdown(f"""
• Annual Return: {portfolio['annual_return']:.2%}

• Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}

• Portfolio Beta: {risk['beta']:.2f}

• Portfolio Alpha: {risk['alpha']:.2f}

• Highest Allocation: {highest_stock} ({highest_weight:.1%})

• Optimized Expected Return: {opt['expected_return']:.2%}
""")

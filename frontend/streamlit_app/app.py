"""Interactive Streamlit UI for the FPT Stock Forecasting project."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from components import render_metric_card
from streamlit.components.v1 import html

BASE_DIR = Path(__file__).parent
CSS_PATH = BASE_DIR / "assets" / "css" / "theme.css"
JS_PATH = BASE_DIR / "assets" / "js" / "animations.js"


def load_static_assets() -> None:
    """Inject custom CSS/JS for modern styling."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    if JS_PATH.exists():
        html(f"<script>{JS_PATH.read_text()}</script>", height=0)


def get_dummy_metrics() -> list[tuple[str, str, float]]:
    """Return placeholder KPI metrics."""
    return [
        ("Latest Close", "₫118.45K", 1.24),
        ("Projected 30D CAGR", "8.6%", 0.82),
        ("Model Confidence", "92%", -0.45),
    ]


def get_dummy_history(periods: int = 120) -> pd.DataFrame:
    """Generate synthetic historical prices for prototyping."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods, freq="B")
    base = 95 + np.linspace(-6, 6, len(dates)) + np.sin(np.linspace(0, 6, len(dates))) * 4
    df = pd.DataFrame({"date": dates, "close": base})
    df["volume"] = (1.2 + np.cos(np.linspace(0, 4, len(dates)))) * 1_000_000
    return df


def get_dummy_forecast(horizon: int) -> pd.DataFrame:
    """Return placeholder multi-step forecast."""
    start_price = 118
    dates = pd.date_range(
        start=pd.Timestamp.today() + pd.offsets.BDay(1), periods=horizon, freq="B"
    )
    growth_trend = np.linspace(0, 4, len(dates))
    noise = np.sin(np.linspace(0, 3, len(dates)))
    prices = start_price + growth_trend + noise
    returns = np.diff(np.insert(prices, 0, start_price)) / start_price
    return pd.DataFrame(
        {"date": dates, "forecast_price": prices, "expected_return": returns},
    )


def render_header() -> None:
    description = (
        "Monitor market momentum, preview smart forecasts, "
        "and prepare trading plans – all from a single modern console."
    )
    html_content = (
        '<div class="glass-card">'
        '<div class="hero-title">'
        "<h1>FPT Stock Intelligence</h1>"
        f"<p>{description}</p>"
        "</div>"
        '<div class="glow-divider"></div>'
        "<div>"
        '<span class="tag-pill">ElasticNet V6</span>'
        '<span class="tag-pill">Selective Features</span>'
        '<span class="tag-pill">100-Step Forecast</span>'
        "</div>"
        "</div>"
    )
    st.markdown(html_content, unsafe_allow_html=True)


def render_metrics(metrics: list[tuple[str, str, float]]) -> None:
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics, strict=False):
        with col:
            render_metric_card(*metric)


def render_price_section(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    st.subheader("Market Pulse & Projection")
    combined = pd.concat(
        [
            history.rename(columns={"close": "price"}).assign(label="Historical"),
            forecast.rename(columns={"forecast_price": "price"}).assign(label="Forecast"),
        ]
    )
    fig = px.line(
        combined,
        x="date",
        y="price",
        color="label",
        color_discrete_sequence=["#8c6bff", "#45d0ff"],
        template="plotly_dark",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fbff"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fbff"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_table(forecast: pd.DataFrame) -> None:
    st.subheader("Upcoming Forecast Snapshots")
    preview = forecast.head(10).copy()
    preview["date"] = preview["date"].dt.strftime("%d %b %Y")
    preview["expected_return"] = (preview["expected_return"] * 100).map("{:.2f}%".format)
    preview.rename(
        columns={
            "date": "Session",
            "forecast_price": "Projected Close (₫K)",
            "expected_return": "Δ vs Prev",
        },
        inplace=True,
    )
    st.dataframe(
        preview,
        use_container_width=True,
        column_config={
            "Projected Close (₫K)": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
    )


def render_cta() -> None:
    cta_description = (
        "Swap the dummy data with live predictions from the FastAPI backend "
        "and unlock automated monitoring."
    )
    html_content = (
        '<div class="glass-card cta-card">'
        "<h3>Ready to connect the real-time API?</h3>"
        f"<p>{cta_description}</p>"
        "<ul>"
        "<li>POST /api/v1/predict/single</li>"
        "<li>POST /api/v1/predict/multi</li>"
        "<li>POST /api/v1/predict/full</li>"
        "</ul>"
        "</div>"
    )
    st.markdown(html_content, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="FPT Stock Intelligence",
        page_icon="📈",
        layout="wide",
    )
    load_static_assets()

    # Sidebar controls
    st.sidebar.title("Prototype Controls")
    horizon = st.sidebar.slider("Forecast horizon (business days)", 30, 100, 60, step=10)
    st.sidebar.markdown(
        f"""
        <div class="slider-label-row">
            <span class="slider-label">30d</span>
            <span class="slider-label slider-label-current">{horizon}d</span>
            <span class="slider-label">100d</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Baseline parameters are fixed to preserve Kaggle MSE results.")

    render_header()

    metrics = get_dummy_metrics()
    render_metrics(metrics)

    history = get_dummy_history()
    forecast = get_dummy_forecast(horizon)
    render_price_section(history, forecast)
    render_forecast_table(forecast)
    render_cta()


if __name__ == "__main__":
    main()

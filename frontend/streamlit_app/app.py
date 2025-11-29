"""Interactive Streamlit UI for the FPT Stock Forecasting project."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from components import render_metric_card
from streamlit.components.v1 import html

BASE_DIR = Path(__file__).parent
CSS_PATH = BASE_DIR / "assets" / "css" / "theme.css"
JS_PATH = BASE_DIR / "assets" / "js" / "animations.js"

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Change this if your API runs on different port


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


def fetch_realtime_prediction(
    api_url: str, n_steps: int = 100, historical_days: int = 120
) -> dict | None:
    """
    Fetch realtime prediction from API

    Args:
        api_url: Base URL of the API
        n_steps: Number of days to forecast
        historical_days: Number of historical days to fetch

    Returns:
        Response dict or None if error
    """
    try:
        response = requests.post(
            f"{api_url}/api/v1/predict/realtime",
            json={"n_steps": n_steps, "historical_days": historical_days},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def render_cta() -> None:
    cta_description = (
        "Use the sidebar to fetch real-time data from internet and get live predictions. "
        "The system automatically fetches FPT stock data and uses the trained model "
        "to forecast future prices."
    )
    html_content = (
        '<div class="glass-card cta-card">'
        "<h3>Realtime Prediction Available</h3>"
        f"<p>{cta_description}</p>"
        "<ul>"
        "<li>POST /api/v1/predict/realtime - Fetch data from internet and predict</li>"
        "<li>POST /api/v1/predict/multi - Predict with custom data</li>"
        "<li>POST /api/v1/predict/full - Full 100-day prediction</li>"
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
    st.sidebar.title("Prediction Controls")

    # Data source selection
    use_realtime = st.sidebar.checkbox("Use Realtime Data from Internet", value=False)

    if use_realtime:
        st.sidebar.subheader("API Configuration")
        api_url = st.sidebar.text_input("API URL", value=API_BASE_URL)
        historical_days = st.sidebar.slider("Historical Days to Fetch", 20, 365, 120, step=10)

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

    if not use_realtime:
        st.sidebar.caption("Baseline parameters are fixed to preserve Kaggle MSE results.")
        st.sidebar.info("💡 Enable 'Use Realtime Data' to fetch live FPT stock data from internet")

    render_header()

    # Fetch realtime data or use dummy data
    if use_realtime:
        st.info("🔄 Fetching realtime data from internet...")
        with st.spinner("Fetching FPT stock data and generating predictions..."):
            result = fetch_realtime_prediction(
                api_url, n_steps=horizon, historical_days=historical_days
            )

        if result:
            # Show fetch status
            if result.get("fetched_new_data", False):
                st.success(
                    "✅ Fetched new data from internet! "
                    f"Total: {result['fetched_data_count']} days. "
                    f"Latest date: {result['latest_date']}"
                )
                if result.get("previous_last_date"):
                    st.info(f"📊 Previous last date in dataset: {result['previous_last_date']}")
            else:
                st.info(
                    "ℹ️ Using existing dataset (no new data needed). "
                    f"Total: {result['fetched_data_count']} days. "
                    f"Latest date: {result['latest_date']}"
                )

            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(result["predictions"])
            predictions_df["date"] = pd.to_datetime(predictions_df["date"])

            # Calculate metrics from real data
            latest_price = predictions_df.iloc[0]["price"] if len(predictions_df) > 0 else 0
            avg_return = predictions_df["return"].mean() * 100 if len(predictions_df) > 0 else 0

            metrics = [
                ("Latest Close", f"₫{latest_price:.2f}K", 0),
                ("Avg Projected Return", f"{avg_return:.2f}%", 0),
                ("Forecast Days", f"{result['n_steps']}d", 0),
            ]
            render_metrics(metrics)

            # For history, we'll show a simple trend
            # (in real app, you may want to fetch full historical data)
            history_dates = pd.date_range(
                end=pd.Timestamp(result["latest_date"]),
                periods=min(120, result["fetched_data_count"]),
                freq="B",
            )
            history = pd.DataFrame(
                {
                    "date": history_dates,
                    "close": [latest_price]
                    * len(history_dates),  # Placeholder - in real app, fetch full history
                }
            )

            forecast = predictions_df.rename(
                columns={"price": "forecast_price", "return": "expected_return"}
            )
            render_price_section(history, forecast)
            render_forecast_table(forecast)
        else:
            st.error("Failed to fetch realtime data. Showing dummy data instead.")
            metrics = get_dummy_metrics()
            render_metrics(metrics)
            history = get_dummy_history()
            forecast = get_dummy_forecast(horizon)
            render_price_section(history, forecast)
            render_forecast_table(forecast)
    else:
        # Use dummy data
        metrics = get_dummy_metrics()
        render_metrics(metrics)
        history = get_dummy_history()
        forecast = get_dummy_forecast(horizon)
        render_price_section(history, forecast)
        render_forecast_table(forecast)

    render_cta()


if __name__ == "__main__":
    main()

"""Interactive Streamlit UI for the FPT Stock Forecasting project."""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from components import render_metric_card
from streamlit.components.v1 import html

# CRITICAL: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="FPT Stock Intelligence",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).parent.parent.parent  # Go up to project root
CSS_PATH = BASE_DIR / "frontend" / "streamlit_app" / "assets" / "css" / "theme.css"
JS_PATH = BASE_DIR / "frontend" / "streamlit_app" / "assets" / "js" / "animations.js"
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# API Configuration
API_BASE_URL = os.getenv(
    "API_BASE_URL", "http://localhost:8000"
)  # Can be overridden via environment variable


def check_dataset_exists() -> Path | None:
    """
    Check if dataset file exists in data/raw/ directory.

    Returns:
        Path to dataset file if found (contains 'train' in filename), None otherwise
    """
    if not DATA_RAW_DIR.exists():
        return None

    csv_files = list(DATA_RAW_DIR.glob("*.csv"))
    train_files = [f for f in csv_files if "train" in f.name.lower()]

    if train_files:
        return train_files[0]
    return None


def save_uploaded_dataset(uploaded_file, filename: str | None = None) -> Path | None:
    """
    Save uploaded CSV file to data/raw/ directory with 'train' + date in filename.

    Args:
        uploaded_file: Streamlit uploaded file object
        filename: Optional custom filename (if None, uses train_YYYYMMDD.csv format)

    Returns:
        Path to saved file if successful, None otherwise
    """
    try:
        # Ensure data/raw directory exists
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Generate filename with 'train' + date
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"train_{date_str}.csv"

        # Ensure filename contains 'train'
        if "train" not in filename.lower():
            date_str = datetime.now().strftime("%Y%m%d")
            base_name = Path(filename).stem
            filename = f"train_{base_name}_{date_str}.csv"

        # Full path
        save_path = DATA_RAW_DIR / filename

        # Save file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Dataset saved to: {save_path}")
        return save_path
    except Exception as e:
        st.error(f"❌ Error saving dataset: {e}")
        return None


def load_static_assets() -> None:
    """Inject custom CSS/JS for modern styling."""
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    if JS_PATH.exists():
        html(f"<script>{JS_PATH.read_text()}</script>", height=0)
    # Ensure buttons (e.g. 'Run prediction') are clearly visible
    # Fix header/footer z-index to not cover Streamlit controls
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #f97316;
            color: #0f172a;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.3rem 1.2rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #fb923c;
            color: #0f172a;
        }

        /* Simple: Just ensure glass-card doesn't interfere with Streamlit header */
        .glass-card {
            position: relative !important;
            z-index: 1 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1.5rem !important;
        }

        /* Footer styling - fixed at bottom */
        .app-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--card-bg);
            border-top: 1px solid var(--card-border);
            padding: 0.75rem 1.5rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            letter-spacing: 0.1em;
            z-index: 100;
            backdrop-filter: blur(16px);
            box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.2);
        }

        /* Add padding to main content to prevent footer overlap */
        .main .block-container {
            padding-bottom: 4rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    description = (
        "Monitor market momentum, preview smart forecasts, "
        "and prepare trading plans – all from a single modern console."
    )
    # CSS already handles spacing, but adding minimal spacing for safety
    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
    html_content = (
        '<div class="glass-card">'
        '<div class="hero-title">'
        "<h1>FPT Stock Intelligence</h1>"
        f"<p>{description}</p>"
        "</div>"
        '<div class="glow-divider"></div>'
        "<div>"
        '<span class="tag-pill">PatchTST V2</span>'
        '<span class="tag-pill">Post-processing</span>'
        '<span class="tag-pill">Smooth Bias Correction</span>'
        "</div>"
        "</div>"
    )
    st.markdown(html_content, unsafe_allow_html=True)


def render_footer() -> None:
    """Render fixed footer at the bottom of the page."""
    footer_html = '<div class="app-footer">CONQ999-AIO2025</div>'
    st.markdown(footer_html, unsafe_allow_html=True)


def render_metrics(metrics: list[tuple[str, str, float]]) -> None:
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics, strict=False):
        title, value, delta = metric
        with col:
            render_metric_card(title, value, delta)


def render_price_section(history: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Render price chart with historical and forecast data."""
    st.subheader("Market Pulse & Projection")

    # Normalize history data - handle both "time" and "date" columns
    history_clean = history.copy()
    if "time" in history_clean.columns:
        history_clean = history_clean.rename(columns={"time": "date"})
    if "date" not in history_clean.columns:
        st.error("History data must have 'time' or 'date' column!")
        return

    history_clean["date"] = pd.to_datetime(history_clean["date"])
    if "close" in history_clean.columns:
        history_clean = history_clean.rename(columns={"close": "price"})
    if "price" not in history_clean.columns:
        st.error("History data must have 'close' or 'price' column!")
        return

    history_clean = history_clean[["date", "price"]].copy()
    history_clean = history_clean.dropna(subset=["date", "price"])
    history_clean = history_clean[history_clean["price"] > 0]

    # Filter out extreme outliers (prices outside reasonable range for FPT stock)
    # FPT stock typically ranges from ~20 to ~150, so filter values outside 1-200 range
    if len(history_clean) > 0:
        before_filter = len(history_clean)
        history_clean = history_clean[
            (history_clean["price"] >= 1) & (history_clean["price"] <= 200)
        ]
        if len(history_clean) < before_filter:
            print(
                f"[DEBUG] render_price_section: Filtered out {before_filter - len(history_clean)} "
                f"historical values outside reasonable range (1-200)"
            )

    history_clean = history_clean.sort_values("date").reset_index(drop=True)

    # Debug: log history data range before rendering
    if len(history_clean) > 0:
        first_date = history_clean["date"].min()
        last_date = history_clean["date"].max()
        print(
            f"[DEBUG] render_price_section: History data has {len(history_clean)} records "
            f"from {first_date.date()} to {last_date.date()}, "
            f"price range: {history_clean['price'].min():.2f} to {history_clean['price'].max():.2f}"
        )

    # Normalize forecast data
    forecast_clean = forecast.copy()
    forecast_clean["date"] = pd.to_datetime(forecast_clean["date"])
    if "forecast_price" in forecast_clean.columns:
        forecast_clean = forecast_clean.rename(columns={"forecast_price": "price"})
    forecast_clean = forecast_clean[["date", "price"]].copy()
    forecast_clean = forecast_clean.dropna(subset=["date", "price"])
    forecast_clean = forecast_clean[forecast_clean["price"] > 0]

    # Filter out extreme outliers (prices outside reasonable range for FPT stock)
    # FPT stock typically ranges from ~20 to ~150, so filter values outside 1-200 range
    if len(forecast_clean) > 0:
        before_filter = len(forecast_clean)
        forecast_clean = forecast_clean[
            (forecast_clean["price"] >= 1) & (forecast_clean["price"] <= 200)
        ]
        if len(forecast_clean) < before_filter:
            print(
                f"[DEBUG] render_price_section: Filtered out {before_filter - len(forecast_clean)} "
                f"forecast values outside reasonable range (1-200)"
            )

    forecast_clean = forecast_clean.sort_values("date").reset_index(drop=True)

    # Debug: log forecast data range
    if len(forecast_clean) > 0:
        print(
            f"[DEBUG] render_price_section: Forecast data has {len(forecast_clean)} records "
            f"from {forecast_clean['date'].min().date()} to "
            f"{forecast_clean['date'].max().date()}, "
            f"price range: {forecast_clean['price'].min():.2f} to "
            f"{forecast_clean['price'].max():.2f}"
        )

    # Combine for range calculation
    # Use only historical data for Y-axis range to avoid outliers from forecast
    # This ensures the chart shows the full historical range properly
    combined = pd.concat([history_clean, forecast_clean], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # For Y-axis range, use historical data only to avoid forecast outliers
    # This ensures the chart displays the full historical data range correctly
    if len(history_clean) > 0:
        price_min_for_range = history_clean["price"].min()
        price_max_for_range = history_clean["price"].max()
    else:
        price_min_for_range = combined["price"].min()
        price_max_for_range = combined["price"].max()

    # Create chart using go.Figure for full control
    fig = go.Figure()

    # Add historical line
    if len(history_clean) > 0:
        # Debug: verify data before adding trace
        print(
            f"[DEBUG] render_price_section: Adding historical trace with "
            f"{len(history_clean)} points, "
            f"date range: {history_clean['date'].min().date()} to "
            f"{history_clean['date'].max().date()}"
        )

        fig.add_trace(
            go.Scatter(
                x=history_clean["date"],
                y=history_clean["price"],
                mode="lines",
                name="Historical",
                line=dict(color="#8c6bff", width=2),
                hovertemplate=(
                    "<b>Historical</b><br>Date: %{x|%Y-%m-%d}<br>" "Price: %{y:.2f}<extra></extra>"
                ),
                # Ensure all points are visible
                connectgaps=False,
            )
        )

    # Add forecast line with dashed style to distinguish from historical
    if len(forecast_clean) > 0:
        # Find the transition point (last historical date)
        if len(history_clean) > 0:
            last_historical_date = history_clean["date"].max()

            # Add vertical line at transition point using add_shape (more compatible)

            fig.add_shape(
                type="line",
                x0=last_historical_date,
                x1=last_historical_date,
                y0=0,
                y1=1,
                yref="paper",  # Use paper coordinates (0-1) for full height
                line=dict(
                    color="rgba(255,255,255,0.3)",
                    width=1,
                    dash="dot",
                ),
            )

            # Add annotation for the transition point
            fig.add_annotation(
                x=last_historical_date,
                y=1,
                yref="paper",
                text="Forecast starts",
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.7)"),
                xanchor="left",
                yanchor="bottom",
            )

        fig.add_trace(
            go.Scatter(
                x=forecast_clean["date"],
                y=forecast_clean["price"],
                mode="lines",
                name="Forecast",
                line=dict(color="#45d0ff", width=2),  # Solid line (no dash)
                hovertemplate=(
                    "<b>Forecast</b><br>Date: %{x|%Y-%m-%d}<br>" "Price: %{y:.2f}<extra></extra>"
                ),
            )
        )

    # Set axes configuration to show full data range
    if len(combined) > 0:
        date_min = combined["date"].min()
        date_max = combined["date"].max()
        date_range_days = (date_max - date_min).days
        date_padding = pd.Timedelta(days=max(30, int(date_range_days * 0.05)))

        # Use historical data range for Y-axis to avoid forecast outliers
        price_min = price_min_for_range
        price_max = price_max_for_range
        price_range = price_max - price_min
        if price_range > 0:
            price_padding = max(price_range * 0.1, price_max * 0.05)
        else:
            price_padding = price_max * 0.1 if price_max > 0 else 1

        # Debug: log axis ranges
        print(
            f"[DEBUG] render_price_section: Combined data has {len(combined)} records "
            f"from {date_min.date()} to {date_max.date()}"
        )
        print(
            f"[DEBUG] render_price_section: Using historical price range for Y-axis: "
            f"{price_min:.2f} to {price_max:.2f} (to avoid forecast outliers)"
        )
        print(
            f"[DEBUG] render_price_section: Setting X-axis range from "
            f"{(date_min - date_padding).date()} to "
            f"{(date_max + date_padding).date()}, "
            f"Y-axis range from {max(0, price_min - price_padding):.2f} to "
            f"{price_max + price_padding:.2f}"
        )

        # Note: Axis ranges are set in update_layout() below to ensure they're applied correctly

    # Prepare layout with explicit ranges if we have data
    layout_config = {
        "template": "plotly_dark",
        "margin": dict(l=10, r=10, t=30, b=10),
        "hovermode": "x unified",
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fbff"),
        ),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color="#f8fbff"),
    }

    # Add explicit axis ranges if we have data (use variables calculated above)
    if len(combined) > 0:
        layout_config["xaxis"] = dict(
            type="date",
            range=[date_min - date_padding, date_max + date_padding],
            autorange=False,  # CRITICAL: Disable auto-scaling
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        )

        layout_config["yaxis"] = dict(
            range=[max(0, price_min - price_padding), price_max + price_padding],
            autorange=False,  # CRITICAL: Disable auto-scaling
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        )
    else:
        layout_config["xaxis"] = dict(
            type="date",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        )
        layout_config["yaxis"] = dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
        )

    fig.update_layout(**layout_config)

    # Configure Plotly to show all data points without sampling
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "fpt_forecast",
            "height": 600,
            "width": 1200,
            "scale": 1,
        },
    }

    st.plotly_chart(fig, use_container_width=True, config=config)


def render_forecast_table(forecast: pd.DataFrame) -> None:
    st.subheader("Upcoming Forecast Snapshots")
    preview = forecast.copy()
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
    Fetch realtime prediction from API.

    Note: The 'historical_days' parameter is for backward compatibility only.
    When an existing dataset (FPT_train.csv) is present, the API ALWAYS returns
    ALL data from the dataset (from 2020-08-03) + newly fetched data, regardless
    of the 'historical_days' value. The parameter only affects behavior when no
    existing dataset exists.
    """
    try:
        print(f"[DEBUG] Sending request: n_steps={n_steps}, historical_days={historical_days}")
        response = requests.post(
            f"{api_url}/api/v1/predict/realtime",
            json={"n_steps": n_steps, "historical_days": historical_days},
            timeout=60,  # Increased timeout for large datasets
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        error_detail = "Unknown error"
        try:
            error_response = e.response.json()
            error_detail = error_response.get("detail", str(e))
        except Exception:
            error_detail = str(e)
        st.error(f"Error calling API: {error_detail}")
        print(f"[ERROR] API error: {e.response.status_code} - {error_detail}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        print(f"[ERROR] Request exception: {e}")
        return None


def call_prediction_api(
    api_url: str,
    endpoint: str,
    historical_data: list[dict],
    n_steps: int | None = None,
) -> dict | None:
    """Call FastAPI prediction endpoints with prepared historical data."""
    endpoint_map = {
        "single": "/api/v1/predict/single",
        "multi": "/api/v1/predict/multi",
        "full": "/api/v1/predict/full",
    }
    payload: dict[str, object] = {"historical_data": historical_data}
    if endpoint == "multi" and n_steps is not None:
        payload["n_steps"] = n_steps

    try:
        response = requests.post(f"{api_url}{endpoint_map[endpoint]}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Error calling API ({endpoint}): {exc}")
        return None


def prepare_payload_from_csv(uploaded_file) -> list[dict] | None:
    """Validate uploaded CSV and convert to list of dicts."""
    if uploaded_file is None:
        st.warning("Please upload a CSV with columns: time, open, high, low, close, volume.")
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to parse CSV: {exc}")
        return None

    required_cols = {"time", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        st.error(f"CSV missing required columns: {missing}")
        return None

    df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("time").reset_index(drop=True)
    if len(df) < 20:
        st.error("Need at least 20 rows of historical data.")
        return None

    payload = [
        {
            "time": row["time"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for _, row in df.iterrows()
    ]
    return payload


def render_prediction_results(
    history_df: pd.DataFrame, forecast_df: pd.DataFrame, headline: str
) -> None:
    """Render charts/tables for prediction output."""
    history = history_df.copy()

    # Debug: log input data
    print(
        f"[DEBUG] render_prediction_results: Received history_df with {len(history_df)} records, "
        f"columns: {list(history_df.columns)}"
    )

    # Ensure we have 'close' column for last_close calculation
    if "close" not in history.columns:
        if "price" in history.columns:
            history["close"] = history["price"]
        else:
            st.error("History data must have 'close' or 'price' column!")
            return

    last_close = float(history["close"].iloc[-1])

    # Normalize column names - ensure 'date' column exists
    if "time" in history.columns:
        history = history.rename(columns={"time": "date"})
    history["date"] = pd.to_datetime(history["date"])

    # Debug: log normalized data
    if len(history) > 0:
        first_date = history["date"].min()
        last_date = history["date"].max()
        print(
            f"[DEBUG] render_prediction_results: Normalized history has {len(history)} records "
            f"from {first_date.date()} to {last_date.date()}"
        )

    forecast = forecast_df.copy()
    forecast.rename(columns={"price": "forecast_price", "return": "expected_return"}, inplace=True)
    forecast["date"] = pd.to_datetime(forecast["date"])

    st.subheader(headline)

    latest_price = float(forecast.iloc[0]["forecast_price"])
    avg_return = float(forecast["expected_return"].mean() * 100)

    if last_close > 0:
        delta_price_pct = (latest_price - last_close) / last_close * 100
    else:
        delta_price_pct = 0.0

    delta_avg_ret = avg_return
    metrics = [
        ("Latest Close", f"₫{latest_price:.2f}K", delta_price_pct),
        ("Avg Projected Return", f"{avg_return:.2f}%", delta_avg_ret),
        ("Forecast Days", f"{len(forecast)}d", None),
    ]
    render_metrics(metrics)

    render_price_section(history, forecast)
    render_forecast_table(forecast)


def predictions_to_dataframe(endpoint: str, response: dict) -> pd.DataFrame:
    """Normalize API response into dataframe with date/price/return."""
    if endpoint == "single":
        records = [
            {
                "date": response["forecast_date"],
                "price": response["predicted_price"],
                "return": response["predicted_return"],
            }
        ]
    else:
        records = response.get("predictions", [])
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


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
    load_static_assets()

    st.sidebar.title("Prediction Controls")
    api_url = st.sidebar.text_input("API Base URL", value=API_BASE_URL)

    mode = st.sidebar.radio(
        "Prediction mode",
        ("Realtime API", "Upload CSV → API"),
        index=1,  # Default to "Upload CSV → API"
    )
    horizon = st.sidebar.slider("Forecast horizon (business days)", 30, 100, 60, step=10)

    # Initialize variables for Realtime API mode
    proceed_with_fetch = False
    has_dataset = False
    realtime_days = 120  # Default value
    upload_choice = None
    uploaded_file_realtime = None

    if mode == "Realtime API":
        # Check if dataset exists in data/raw/ (file with 'train' in name)
        dataset_file = check_dataset_exists()
        has_dataset = dataset_file is not None

        if has_dataset:
            st.sidebar.success(f"✅ Dataset found: {dataset_file.name}")
            realtime_days = st.sidebar.slider(
                "Historical days (realtime fetch)",
                20,
                365,
                120,
                step=10,
                help=(
                    "Note: When a dataset file exists, the system will use ALL data "
                    "from that file + newly fetched data, regardless of this slider value."
                ),
            )
            # User can proceed with fetching
            proceed_with_fetch = True
            upload_choice = None
            uploaded_file_realtime = None
        else:
            # No dataset found - show message box for user to choose
            st.sidebar.warning("⚠️ No dataset found in data/raw/ (file with 'train' in name).")

            # Message box for user choice
            st.sidebar.info(
                "📋 Please choose how to proceed:\n\n"
                "1. Upload a CSV file to use as dataset\n"
                "2. Use slider to fetch data from internet"
            )

            # User choice: Upload file or use slider
            upload_choice = st.sidebar.radio(
                "Choose option:",
                ("Upload CSV file", "Fetch from internet (use slider)"),
                index=None,  # No default selection
                key="realtime_choice",
            )

            if upload_choice == "Upload CSV file":
                uploaded_file_realtime = st.sidebar.file_uploader(
                    "Upload CSV file", type=["csv"], key="realtime_upload"
                )
                if uploaded_file_realtime is not None:
                    # Save uploaded file as dataset
                    if st.sidebar.button("💾 Save as Dataset and Proceed"):
                        saved_path = save_uploaded_dataset(uploaded_file_realtime)
                        if saved_path:
                            st.sidebar.success(f"✅ Saved to: {saved_path.name}")
                            # Refresh to detect new dataset
                            st.rerun()
                    proceed_with_fetch = False
                else:
                    proceed_with_fetch = False
            elif upload_choice == "Fetch from internet (use slider)":
                realtime_days = st.sidebar.slider(
                    "Historical days (realtime fetch)",
                    20,
                    365,
                    120,
                    step=10,
                    help="Number of days to fetch from internet when no dataset exists.",
                )
                proceed_with_fetch = True
                uploaded_file_realtime = None
            else:
                # No choice made yet
                proceed_with_fetch = False
                uploaded_file_realtime = None

    elif mode == "Upload CSV → API":
        # Upload mode - no dataset checking needed
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        st.sidebar.caption("CSV mode uses multi-step prediction based on the slider above.")
        run_prediction = st.sidebar.button(f"Run prediction ({horizon} days)")

    render_header()

    if mode == "Realtime API":
        # Only proceed if user has made a choice (has dataset OR chose option)
        if not proceed_with_fetch:
            if not has_dataset:
                st.info(
                    "👆 Please choose an option in the sidebar: "
                    "upload a CSV file or use slider to fetch from internet."
                )
            return

        # Show fetching message only during the API call
        with st.spinner("🔄 Fetching realtime data from FastAPI ..."):
            result = fetch_realtime_prediction(
                api_url, n_steps=horizon, historical_days=realtime_days
            )

        if result and result.get("predictions"):
            # Show success message after fetching completes
            if result.get("fetched_new_data", False):
                st.success(
                    "✅ Fetched new data from internet! "
                    f"Total: {result['fetched_data_count']} days. "
                    f"Latest date: {result['latest_date']}"
                )
            else:
                st.info(
                    "ℹ️ Using cached dataset. "
                    f"Total: {result['fetched_data_count']} days. "
                    f"Latest date: {result['latest_date']}"
                )

            # Group debug information in an expander (will be populated after data cleaning)

            predictions_df = pd.DataFrame(result["predictions"])
            predictions_df["date"] = pd.to_datetime(predictions_df["date"])
            predictions_df = predictions_df[predictions_df["price"] > 0]
            predictions_df = predictions_df.sort_values("date").reset_index(drop=True)

            if len(predictions_df) == 0:
                st.error("⚠️ All predictions are invalid (zero or negative prices).")
                return

            # Use real historical data from API response
            # This should contain ALL data from FPT_train.csv (from 2020-08-03) + newly fetched data
            if result.get("historical_data") and len(result["historical_data"]) > 0:
                history_df = pd.DataFrame(result["historical_data"])
                history_df["time"] = pd.to_datetime(history_df["time"])
                history_df = history_df.sort_values("time").reset_index(drop=True)

                # Ensure we have 'close' column
                if "close" not in history_df.columns:
                    if "price" in history_df.columns:
                        history_df["close"] = history_df["price"]
                    else:
                        st.error("Historical data missing 'close' column!")
                        return

                # Clean data - remove invalid records but keep all valid historical data
                history_df = history_df.dropna(subset=["time", "close"])
                history_df = history_df[history_df["close"] > 0]
                history_df = history_df.sort_values("time").reset_index(drop=True)

                # Add cleaned data info to expander
                if len(history_df) > 0:
                    first_date_clean = history_df["time"].min()
                    last_date_clean = history_df["time"].max()
                    # Create expander with all debug information
                    with st.expander("📊 Data Details", expanded=False):
                        if result.get("previous_last_date"):
                            st.caption(f"Previous last date: {result['previous_last_date']}")
                        st.caption(
                            f"Received {len(history_df)} historical records "
                            f"from {first_date_clean.date()} to {last_date_clean.date()}"
                        )
                    print(
                        f"[DEBUG] Cleaned historical data: {len(history_df)} records "
                        f"from {first_date_clean.date()} to {last_date_clean.date()}"
                    )
            else:
                # Fallback: create placeholder if API doesn't return historical_data
                st.warning("⚠️ API did not return historical_data. Using placeholder.")
                latest_price = predictions_df.iloc[0]["price"] if not predictions_df.empty else 0
                history_dates = pd.date_range(
                    end=pd.Timestamp(result["latest_date"]),
                    periods=min(120, result["fetched_data_count"]),
                    freq="B",
                )
                history_df = pd.DataFrame(
                    {
                        "time": history_dates,
                        "close": [latest_price] * len(history_dates),
                    }
                )

            render_prediction_results(history_df, predictions_df, "Realtime forecast")
        else:
            st.error("Failed to fetch realtime predictions from API.")

    elif mode == "Upload CSV → API":
        st.info("Upload historical OHLCV data to call FastAPI prediction endpoints.")
        if run_prediction:
            historical_payload = prepare_payload_from_csv(uploaded_file)
            if historical_payload:
                result = call_prediction_api(
                    api_url,
                    "multi",
                    historical_payload,
                    n_steps=horizon,
                )
                if result:
                    forecast_df = predictions_to_dataframe("multi", result)
                    if forecast_df.empty:
                        st.warning("API returned no predictions.")
                    else:
                        history_df = pd.DataFrame(historical_payload)
                        # Ensure 'close' column exists for consistency
                        history_df["time"] = pd.to_datetime(history_df["time"])
                        render_prediction_results(history_df, forecast_df, "API prediction result")
                else:
                    st.error("API did not return a valid response.")
        else:
            st.caption("Select a CSV file and click 'Run prediction' to call the API.")

    render_cta()

    # Render fixed footer
    render_footer()


if __name__ == "__main__":
    main()

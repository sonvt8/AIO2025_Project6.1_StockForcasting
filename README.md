# FPT Stock Forecasting (v2) — PatchTST + Post-processing + Smooth Bias

A production-style project to forecast FPT stock prices using PatchTST (time-series Transformer) enhanced by:
- Post-processing Regression (LinearRegression)
- Smooth Bias Correction (Linear 20%)

The v2 replaces the baseline ElasticNet (v1) with a stronger, inference-only PatchTST pipeline. API and UI contracts are preserved.

---

## Table of Contents

1) Project Overview
2) Architecture Diagram
3) Training Pipeline (one-time)
4) Inference Pipeline (runtime)
5) API Endpoints
6) Frontend (Streamlit) Workflow
7) Data Flow & Normalization
8) Key Components
9) Artifacts & Releases (Plan A)
10) Run the System (API + UI)
11) Testing
12) Data Format Reference
13) Versioning and Tags (Important)
14) Troubleshooting

---

## 1) Project Overview

- Model: PatchTST (NeuralForecast) with fixed hyperparameters from the baseline notebook `notebooks/baseline_patchtst_v2.ipynb`
- Post-processing: Linear Regression trained on fold predictions
- Smooth Bias Correction: Linear blending over first 20% of horizon
- Real-time data fetch (vnstock) + merge with existing dataset for charting
- FastAPI backend for prediction endpoints
- Streamlit frontend for interactive visualization

Goals:
- Predict next 1–100 business days of FPT closing prices
- Keep API contract stable for UI compatibility
- Provide reproducible artifacts via GitHub Releases

---

## 2) Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FPT STOCK FORECASTING SYSTEM                     │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PHASE (One-time)                       │
├──────────────────────────────────────────────────────────────────────────┤
│  1) Load training (data/raw/*train*.csv)                                 │
│  2) Train PatchTST (NeuralForecast)                                      │
│  3) Train post-processing (LinearRegression, TSCV folds)                 │
│  4) Export artifacts: patchtst.pt, best_params.json, post_model.pkl,     │
│     smooth_config.json                                                    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PHASE (Runtime)                       │
├──────────────────────────────────────────────────────────────────────────┤
│  FASTAPI                                                                 │
│  - Load artifacts on startup or first request                            │
│  - Endpoints: single/multi/full/realtime                                 │
│  - Realtime: fetch new data (vnstock) + merge with existing dataset      │
│  - ForecastService: validate → window(100) → PatchTST → postproc → smooth│
│    → clamp → dates(BDay) → returns(log)                                  │
│  STREAMLIT                                                               │
│  - Realtime mode or Upload CSV mode                                      │
│  - Charts (historical + forecast), metrics, table                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3) Training Pipeline (one-time)

Location: `app/utils/model_trainer.py`

- Input: `data/raw/*train*.csv` with columns time, open, high, low, close, volume
- Steps:
  1) Read and sort training series; build NeuralForecast dataset
  2) Train PatchTST with fixed hparams (no Optuna)
  3) TimeSeriesSplit folds → per-fold NF fit → collect (y_pred, y_true)
  4) Train LinearRegression post-model
  5) Export artifacts:
     - `app/models/artifacts/patchtst.pt`
     - `app/models/artifacts/best_params.json`
     - `app/models/artifacts/post_model.pkl`
     - `app/models/artifacts/smooth_config.json`

Hyperparameters (default in `app/config.py`):
```
input_size=100, patch_len=32, stride=4, learning_rate=0.0016108149,
max_steps=250, revin=True, horizon=100
```

Run training (optional — only if you want to build artifacts locally):
- From project root: `python -m app.utils.model_trainer`

---

## 4) Inference Pipeline (runtime)

Locations: `app/models/patchtst_loader.py`, `app/services/forecast_service.py`

- Loader (PatchTSTLoader):
  - Local-first load of artifacts; if missing, auto-download from GitHub Releases (Plan A)
  - Build model skeleton with saved hparams
  - Load state_dict → eval mode; freeze parameters
- ForecastService:
  - Validate and sort historical data
  - Extract last `input_size` points → tensor shape (1, 1, input_size)
  - Forward PatchTST → baseline predictions
  - Post-process via LinearRegression
  - Smooth bias correction (linear, 20%) → clamp to non-negative
  - Build forecast dates (business days) and log-returns vs last close

Output dict:
```
{
  "prices": np.ndarray[float], # 1D length n_steps
  "returns": np.ndarray[float],
  "dates": np.ndarray[datetime64]
}
```

---

## 5) API Endpoints

Base: `http://localhost:8000`

- GET `/health` → {status, message, models_loaded}
- GET `/api/v1/model/info` → PatchTST config/hparams
- POST `/api/v1/predict/single` → next-day price/return
- POST `/api/v1/predict/multi` → N-day forecast (1–100)
- POST `/api/v1/predict/full` → 100-day forecast
- POST `/api/v1/predict/realtime` → auto-fetch data + forecast, returns predictions and merged historical data for charting

See `app/api/routes.py` and run Swagger: `http://localhost:8000/docs`.

---

## 6) Frontend (Streamlit) Workflow

Location: `frontend/streamlit_app/app.py`

Modes:
- Realtime API: uses dataset in `data/raw/*train*.csv` + newly fetched vnstock data
- Upload CSV → API: user uploads OHLCV CSV (≥ 20 rows)

UI shows:
- Metrics: Latest Close, Avg Projected Return, Forecast Days
- Chart: historical vs forecast (vertical transition marker)
- Table: forecast snapshots

Run: `streamlit run frontend/streamlit_app/app.py`

---

## 7) Data Flow & Normalization

- Dataset detection: any CSV in `data/raw` with "train" in filename (case-insensitive)
- Realtime fetch:
  1) Load dataset → last date
  2) Fetch from (last_date + 1 BDay) to today via `vnstock`
  3) Merge, sort, deduplicate
- Normalization:
  - If fetched prices have max > 1000 → treat as VND, convert to thousands (÷1000) to match training
  - Filter extreme outliers outside [1, 500] (thousands VND)
  - Validate OHLC relationships (high ≥ low, close ∈ [low, high], ...)

---

## 8) Key Components

- `app/utils/model_trainer.py`: Train PatchTST + post-model + smooth config export
- `app/models/patchtst_loader.py`: Load artifacts; predict_sequence and predict_prices
- `app/services/forecast_service.py`: High-level prediction flow; returns prices/dates/returns
- `app/services/data_fetcher.py`: Smart fetch and merge realtime data (vnstock)
- `app/api/routes.py` + `app/api/schemas.py`: Endpoints + Pydantic schemas
- `frontend/streamlit_app/app.py`: Interactive UI
- `app/config.py`: API config, hparams, Releases tag and asset names

---

## 9) Artifacts & Releases (Plan A)

- Local-first: load artifacts from `app/models/artifacts/`
- If missing → auto-download from GitHub Releases according to `app/config.py`:
  - owner: `sonvt8`, repo: `AIO2025_Project6.1_StockForcasting`
  - tag: `version-2.0-patchtst`
  - assets: `patchtst.pt`, `best_params.json`, `post_model.pkl`, `smooth_config.json`

How to publish artifacts to a Release (quick):
- `git tag -a version-2.0-patchtst -m "PatchTST v2 artifacts"`
- `git push origin version-2.0-patchtst`
- Create/publish a Release for that tag and upload the 4 files above

Note: Do not commit artifact binaries to the repo; they are ignored by `.gitignore`.

---

## 10) Run the System (API + UI)

Prepare environment:
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

Start FastAPI:
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start Streamlit:
```
streamlit run frontend/streamlit_app/app.py
```

Optional Docker Compose (backend+frontend):
```
docker compose up --build
# API:     http://localhost:8000
# UI:      http://localhost:8501
```

---

## 11) Testing

- End-to-end API tests:
  - `python test_api.py`  (health, model info, realtime prediction)
- Data fetch tests:
  - `python test_data_fetcher.py`  (smart fetch + metadata)

---

## 12) Data Format Reference

CSV columns (train and upload modes):
- time: YYYY-MM-DD
- open, high, low, close: floats in thousands VND
- volume: non-negative number

The system auto-converts fetched prices from VND to thousands VND if needed (max price > 1000).

---

## 13) Versioning and Tags (Important)

- You do NOT need a git tag to push normal code changes. Users who clone the repo will get the latest code from the default branch.
- However, the model auto-download uses a specific Release tag defined in `app/config.py` (`GITHUB_RELEASE["tag"]`).
  - If you update model artifacts, publish a new Release tag (e.g., `version-2.1-patchtst`) with the 4 assets, and update `app/config.py` to the new tag. Then deploy/restart.
  - If you keep the same tag and upload new assets with the same names (not recommended), ensure you overwrite assets in the Release. Some users may still have older artifacts cached locally.

Recommended flow when artifacts change:
1) Train/export new artifacts locally
2) Create a new Release tag (semantic): `version-2.1-patchtst`
3) Upload 4 artifacts to that Release
4) Update `app/config.py` → `GITHUB_RELEASE["tag"] = "version-2.1-patchtst"`
5) Commit and push code; restart backend/UI

---

## 14) Troubleshooting

- Missing artifacts on startup:
  - Ensure the 4 files exist locally OR the Release assets/URL are correct.
- Realtime fetch fails (vnstock/network):
  - Use Upload CSV mode to predict with your own data.
- Shape/runtime errors during inference:
  - Ensure you’re on current code; inputs shaped (1, 1, input_size) for PatchTST; pipeline normalizes arrays to 1D before returns.
- Dataset not detected:
  - Place a CSV in `data/raw` with "train" in its filename (case-insensitive) and the required columns.

---

Happy forecasting! 🎯

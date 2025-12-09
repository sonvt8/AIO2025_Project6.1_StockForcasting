# FPT Stock Forecasting (v2) — PatchTST + Post-processing + Smooth Bias

A production-ready stock forecasting system using PatchTST (time-series Transformer) enhanced by:
- Post-processing Regression (LinearRegression)
- Smooth Bias Correction (Linear 20%)
- **Auto Device Detection** (CUDA/MPS/CPU) with device-specific artifact management
- **Automatic Training** when artifacts are missing for current device

The v2 replaces the baseline ElasticNet (v1) with a stronger, inference-only PatchTST pipeline. API and UI contracts are preserved.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Architecture](#3-architecture)
4. [Auto Device Detection & Artifact Management](#4-auto-device-detection--artifact-management)
5. [Training Pipeline](#5-training-pipeline)
6. [Inference Pipeline](#6-inference-pipeline)
7. [API Endpoints](#7-api-endpoints)
8. [Testing](#8-testing)
9. [Frontend (Streamlit)](#9-frontend-streamlit)
10. [Installation & Setup](#10-installation--setup)
11. [Data Format](#11-data-format)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

- **Model**: PatchTST (NeuralForecast) with fixed hyperparameters from baseline notebook
- **Post-processing**: Linear Regression trained on fold predictions (TimeSeriesSplit)
- **Smooth Bias Correction**: Linear blending over first 20% of horizon
- **Auto Device Detection**: Automatically detects CUDA/MPS/CPU and manages device-specific artifacts
- **Auto Training**: Automatically trains artifacts if missing for current device
- **Real-time data fetch** (vnstock) + merge with existing dataset for charting
- **FastAPI backend** for prediction endpoints
- **Streamlit frontend** for interactive visualization

**Goals:**
- Predict next 1–100 business days of FPT closing prices
- Keep API contract stable for UI compatibility
- Provide reproducible artifacts via GitHub Releases
- **Ensure consistent metrics across different hardware environments**

---

## 2. Key Features

### 🚀 Auto Device Detection
- Automatically detects compute device: CUDA (GPU) > MPS (Apple Silicon) > CPU
- Creates device-specific artifact directories
- Caches artifacts per device type to ensure consistent performance

### 🤖 Auto Training
- Automatically trains and exports artifacts if missing for current device
- No manual intervention required
- Training happens on first API request if artifacts are missing

### 📊 Metrics Validation
- Built-in endpoint to test predictions against ground truth
- Device-specific MSE thresholds (CUDA: 18.5, MPS/CPU: 50.0)
- Comprehensive metrics: MSE, RMSE, MAE, R², MAPE, Bias

### 🔄 Consistent Performance
- Artifacts cached per device fingerprint
- Metrics remain stable across multiple runs
- No performance drift between training and inference

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FPT STOCK FORECASTING SYSTEM                     │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                    AUTO DEVICE DETECTION & ARTIFACT MANAGEMENT          │
├──────────────────────────────────────────────────────────────────────────┤
│  1) Detect device (CUDA/MPS/CPU)                                        │
│  2) Check artifacts in device-specific directory                        │
│  3) Auto-train if missing (or download from GitHub Releases)           │
│  4) Cache artifacts: app/models/artifacts/{device_type}_{fingerprint}/   │
└──────────────────────────────────────────────────────────────────────────┘

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
│  - Endpoints: single/multi/full/realtime/test/metrics                   │
│  - Realtime: fetch new data (vnstock) + merge with existing dataset      │
│  - ForecastService: validate → window(100) → PatchTST → postproc → smooth│
│    → clamp → dates(BDay) → returns(log)                                  │
│  STREAMLIT                                                               │
│  - Realtime mode or Upload CSV mode                                      │
│  - Charts (historical + forecast), metrics, table                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Auto Device Detection & Artifact Management

### How It Works

1. **Device Detection**: System automatically detects CUDA/MPS/CPU on startup
2. **Artifact Directory**: Creates device-specific directory: `app/models/artifacts/{device_type}_{fingerprint}/`
3. **Auto Training**: If artifacts missing, automatically runs training script
4. **Caching**: Artifacts cached per device, no need to retrain

### Artifact Structure

```
app/models/artifacts/
├── cuda_<fingerprint>/     # Artifacts for GPU CUDA
│   ├── patchtst.pt
│   ├── patchtst_full.pt
│   ├── best_params.json
│   ├── post_model.pkl
│   └── smooth_config.json
├── mps_<fingerprint>/      # Artifacts for Apple MPS
│   └── ...
└── cpu_<fingerprint>/      # Artifacts for CPU
    └── ...
```

### Environment Variables

- `PATCHTST_AUTO_TRAIN`: Enable/disable auto-train (default: 1)
- `PATCHTST_USE_NF`: Use NF inference instead of raw forward (default: 0)

### Expected Performance

- **CUDA (GPU)**: MSE ~17-18 (best performance)
- **MPS (Apple Silicon)**: MSE ~45-50 (stable)
- **CPU**: MSE ~45-50 (stable)

**Note**: MSE from API inference may differ slightly from training script due to artifact reloading, but remains within acceptable thresholds.

---

## 5. Training Pipeline

### Manual Training (Optional)

If you want to train artifacts manually:

```bash
python scripts/run_patchtst_export.py \
  --train data/raw/FPT_train.csv \
  --test data/test/FPT_test.csv \
  --out app/models/artifacts \
  --deterministic --workspace-config :4096:8
```

### Hyperparameters

Fixed hyperparameters (in `app/config.py`):
```
input_size=100
patch_len=32
stride=4
learning_rate=0.001610814898983045
max_steps=250
revin=True
horizon=100
```

### Training Steps

1. Load and sort training series; build NeuralForecast dataset
2. Train PatchTST with fixed hparams
3. TimeSeriesSplit folds → per-fold NF fit → collect (y_pred, y_true)
4. Train LinearRegression post-model
5. Export artifacts to device-specific directory

---

## 6. Inference Pipeline

**Locations**: `app/models/patchtst_loader.py`, `app/services/forecast_service.py`

### Loader (PatchTSTLoader)
- Auto-detects device and uses device-specific artifacts
- Auto-trains if artifacts missing
- Builds model skeleton with saved hparams
- Loads state_dict → eval mode; freezes parameters

### ForecastService
- Validates and sorts historical data
- Extracts last `input_size` points
- Forward PatchTST → baseline predictions
- Post-process via LinearRegression
- Smooth bias correction (linear, 20%) → clamp to non-negative
- Builds forecast dates (business days) and log-returns vs last close

**Output dict:**
```python
{
  "prices": np.ndarray[float],  # 1D length n_steps
  "returns": np.ndarray[float],
  "dates": np.ndarray[datetime64]
}
```

---

## 7. API Endpoints

**Base URL**: `http://localhost:8000`

### Core Endpoints

- `GET /health` → `{status, message, models_loaded}`
- `GET /api/v1/model/info` → PatchTST config/hparams + device info
- `POST /api/v1/predict/single` → next-day price/return
- `POST /api/v1/predict/multi` → N-day forecast (1–100)
- `POST /api/v1/predict/full` → 100-day forecast
- `POST /api/v1/predict/realtime` → auto-fetch data + forecast

### Testing Endpoint

- `POST /api/v1/test/metrics` → **Test predictions against ground truth**

  **Request:**
  ```json
  {
    "horizon": 100,
    "train_csv_path": null,  // optional, default: data/raw/FPT_train.csv
    "test_csv_path": null     // optional, default: data/test/FPT_test.csv
  }
  ```

  **Response:**
  ```json
  {
    "device_type": "mps",
    "device_name": "mps",
    "artifact_dir": "/path/to/artifacts/mps_bd5da95f",
    "metrics": {
      "mse": 45.8835,
      "rmse": 6.7741,
      "mae": 5.2341,
      "r2": 0.9234,
      "mape": 2.3456,
      "bias": -3.5109
    },
    "threshold": 50.0,
    "passed": true,
    "test_info": {
      "train_rows": 1149,
      "test_rows": 168,
      "horizon": 100,
      "train_range": "2020-08-03 to 2025-03-10",
      "test_range": "2025-03-11 to 2025-08-15"
    }
  }
  ```

**Interactive API Docs**: http://localhost:8000/docs

---

## 8. Testing

### Test Metrics Endpoint

Test API predictions against ground truth:

```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/test/metrics" \
  -H "Content-Type: application/json" \
  -d '{"horizon": 100}'

# Using test script
python test_api.py
```

### Test Parity (Unit Test)

```bash
# Run parity test (auto-detects device and uses appropriate threshold)
python -m pytest tests/test_parity_patchtst.py -v

# Check current device
python -c "from app.utils.device_detector import detect_device; import json; print(json.dumps(detect_device(), indent=2))"
```

### Test Scripts

- `python test_api.py` - Full API test suite (health, model info, realtime, metrics)
- `python test_data_fetcher.py` - Data fetch tests
- `python -m pytest tests/test_parity_patchtst.py` - Parity test with ground truth

### Expected Results

- **CUDA (GPU)**: MSE ≤ 18.5
- **MPS (Apple Silicon)**: MSE ≤ 50.0
- **CPU**: MSE ≤ 50.0

---

## 9. Frontend (Streamlit)

**Location**: `frontend/streamlit_app/app.py`

### Modes
- **Realtime API**: Uses dataset in `data/raw/*train*.csv` + newly fetched vnstock data
- **Upload CSV → API**: User uploads OHLCV CSV (≥ 20 rows)

### UI Features
- Metrics: Latest Close, Avg Projected Return, Forecast Days
- Chart: historical vs forecast (vertical transition marker)
- Table: forecast snapshots

### Run
```bash
streamlit run frontend/streamlit_app/app.py
```

---

## 10. Installation & Setup

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# Clone repository
git clone <repository-url>
cd AIO2025_Project6.1_StockForcasting

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start Services

**FastAPI Server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Streamlit UI:**
```bash
streamlit run frontend/streamlit_app/app.py
```

**Docker Compose (optional):**
```bash
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

### First Run - Artifacts Auto-Training

**⚠️ Important**: Artifacts (model files) are NOT committed to git (they are in `.gitignore`). When you clone the project on a new machine, the system will automatically handle artifacts:

#### Option 1: Auto-Train (Recommended - Automatic)

On first API request:
1. System detects device (CUDA/MPS/CPU)
2. Checks for artifacts in device-specific directory
3. **If missing, automatically trains artifacts** (may take 5-15 minutes)
4. Caches artifacts for future use

**No manual intervention needed!** Just start the API and make your first request. The system will:
- Detect your device type
- Train artifacts automatically if missing
- Cache them in `app/models/artifacts/{device_type}_{fingerprint}/`

#### Option 2: Download from GitHub Releases (If Available)

If artifacts are published to GitHub Releases:
1. System will attempt to auto-download on first request
2. Configured in `app/config.py` → `GITHUB_RELEASE`
3. Falls back to auto-train if download fails

#### Option 3: Manual Training (Optional)

If you prefer to train manually before starting API:

```bash
python scripts/run_patchtst_export.py \
  --train data/raw/FPT_train.csv \
  --test data/test/FPT_test.csv \
  --out app/models/artifacts \
  --deterministic --workspace-config :4096:8
```

This will create artifacts in `app/models/artifacts/{device_type}_{fingerprint}/` for your device.

**Note**: Artifacts are device-specific. Artifacts trained on GPU won't work optimally on CPU/MPS, so auto-train ensures you get artifacts optimized for your hardware.

---

## 11. Data Format

### CSV Format

CSV files (train and upload modes) must have these columns:
- `time`: YYYY-MM-DD format
- `open`, `high`, `low`, `close`: floats in thousands VND
- `volume`: non-negative number

### Data Location

- Training data: `data/raw/*train*.csv` (any CSV with "train" in filename)
- Test data: `data/test/FPT_test.csv` (for metrics validation)

### Auto Conversion

The system auto-converts fetched prices from VND to thousands VND if needed (max price > 1000).

---

## 12. Troubleshooting

### Artifacts Not Found

**Symptom**: API logs show "Artifacts missing for device"

**Solution**:
- System will auto-train on first request (check logs)
- Or manually train: `python scripts/run_patchtst_export.py --out app/models/artifacts`
- Or download from GitHub Releases (if configured)

### MSE Too High

**Symptom**: Metrics test shows MSE above threshold

**Check**:
1. Device type: `GET /api/v1/model/info`
2. Artifact directory matches device type
3. Artifacts were trained on same device type

**Note**: MSE from API inference may be higher than training script due to artifact reloading. This is normal if within threshold.

### API Won't Start

**Check**:
1. Dependencies: `pip install -r requirements.txt`
2. Data files exist: `data/raw/FPT_train.csv`, `data/test/FPT_test.csv`
3. Check startup logs for errors

### Auto-Train Takes Too Long

**Normal**: First-time training takes 5-15 minutes depending on hardware
- CUDA GPU: ~3-5 minutes
- MPS (Apple Silicon): ~10-15 minutes
- CPU: ~20-30 minutes

**Solution**: Wait for completion, artifacts will be cached for future use

### Device Detection Issues

**Check device**:
```bash
python -c "from app.utils.device_detector import detect_device; import json; print(json.dumps(detect_device(), indent=2))"
```

**Force device type** (if needed):
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU
- Set `PYTORCH_ENABLE_MPS_FALLBACK=0` to disable MPS

---

## Key Components

- `app/utils/device_detector.py`: Device detection and fingerprinting
- `app/models/patchtst_loader.py`: Load artifacts; auto-train if missing; predict
- `app/services/forecast_service.py`: High-level prediction flow
- `app/services/data_fetcher.py`: Smart fetch and merge realtime data (vnstock)
- `app/api/routes.py` + `app/api/schemas.py`: Endpoints + Pydantic schemas
- `frontend/streamlit_app/app.py`: Interactive UI
- `app/config.py`: API config, hparams, device-specific settings
- `scripts/run_patchtst_export.py`: Training and export script

---

## Artifacts & GitHub Releases

### Why Artifacts Are Not in Git

Artifacts (`.pt`, `.pkl` files) are **excluded from git** because:
- They are large files (>2MB each)
- They are device-specific (GPU artifacts ≠ CPU artifacts)
- They can be regenerated automatically via auto-train

### What Happens When You Clone the Project?

**✅ No manual retraining needed!** The system handles everything automatically:

1. **On first API request**:
   - System detects your device (CUDA/MPS/CPU)
   - Checks for artifacts in device-specific directory
   - **If missing → automatically trains artifacts** (5-15 minutes)
   - Caches artifacts for future use

2. **Artifacts are device-specific**:
   - GPU artifacts won't work optimally on CPU/MPS
   - Auto-train ensures you get artifacts optimized for YOUR hardware
   - Each device type gets its own artifact directory

### Local Artifacts Structure

Artifacts are stored in device-specific directories:
```
app/models/artifacts/
├── cuda_<fingerprint>/     # GPU CUDA artifacts
│   ├── patchtst.pt
│   ├── patchtst_full.pt
│   ├── best_params.json
│   ├── post_model.pkl
│   └── smooth_config.json
├── mps_<fingerprint>/      # Apple MPS artifacts
│   └── ...
└── cpu_<fingerprint>/      # CPU artifacts
    └── ...
```

### GitHub Releases (Optional Fallback)

If you want to share pre-trained artifacts (optional):
1. Train artifacts: `python scripts/run_patchtst_export.py`
2. Create GitHub Release with tag (e.g., `version-2.1-patchtst`)
3. Upload artifacts from `app/models/artifacts/{device_type}_{fingerprint}/`
4. Update `app/config.py` → `GITHUB_RELEASE["tag"]` if needed

**Note**:
- GitHub Releases are **optional** - auto-train works without them
- Artifacts are device-specific, so GPU artifacts may not work well on CPU/MPS
- Auto-train is recommended to ensure optimal artifacts for each device type

---

## Version History

- **v2.0**: PatchTST with auto device detection and artifact management
- **v1.0**: Baseline ElasticNet (deprecated)

---

Happy forecasting! 🎯

# FPT Stock Price Prediction API

API dự đoán giá cổ phiếu FPT sử dụng mô hình ElasticNet với selective features (V6 baseline).

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Pipeline & Techniques](#pipeline--techniques)
- [Luồng vận hành](#luồng-vận-hành)
- [Quickstart](#quickstart)
- [Sử dụng API](#sử-dụng-api)
- [API Endpoints](#api-endpoints)
- [Ví dụ sử dụng](#ví-dụ-sử-dụng)

## 📊 Tổng quan

API này được phát triển dựa trên baseline V6 từ notebook `improved_v6_selective_features.ipynb`, sử dụng:

- **Model**: ElasticNet với 2-stage grid search
- **Features**: 39 features (34 base + 5 selective)
  - Base features: returns, volumes, lags, volatility, SMA, RSI, Bollinger Bands, calendar features
  - Selective features: ROC (10, 20), Momentum (10, 20), Volume Ratio
- **Forecasting**: Iterative multi-step forecasting (autoregressive)
- **Calibration**: Linear regression calibration
- **Ensemble**: Naive + Model ensemble (optional)

## 📁 Cấu trúc thư mục

```
project6.1/
├── app/
│   ├── api/               # FastAPI routes & schemas
│   ├── services/          # Feature engineering & forecasting logic
│   ├── models/            # Model loader + artifacts
│   ├── utils/             # Helpers, data loaders, model checker/trainer
│   └── main.py            # FastAPI entry point
├── data/
│   └── raw/FPT_train.csv  # Dataset train gốc
├── notebooks/             # Notebook baseline tham khảo
├── export_models.py       # Script train/export (tùy chọn)
├── example_usage.py       # Script test endpoints
├── requirements.txt
└── README.md
```

- `app/models/artifacts/`: chứa `elasticnet_model.pkl`, `scaler.pkl`, `calibration_model.pkl`, `model_config.json`. Nếu thiếu, API sẽ hỏi để train.
- `app/utils/model_checker.py`: kiểm tra models; `app/utils/model_trainer.py`: tái sử dụng logic train từ notebook.
- `example_usage.py`: chạy toàn bộ health/model-info/predict để xác nhận hệ thống.

## 🔧 Pipeline & Techniques

### 1. Feature Engineering

Từ dữ liệu lịch sử (time, open, high, low, close, volume), pipeline tính toán 39 features:

1. **Base Returns & Volume Changes**
   - `ret_1d_clipped`: Log return 1 ngày (đã winsorize)
   - `vol_chg_clipped`: Log volume change (đã winsorize)

2. **Lag Features**
   - `ret_lag1` đến `ret_lag10`: Returns với lag 1-10 ngày
   - `vol_lag1` đến `vol_lag5`: Volume changes với lag 1-5 ngày

3. **Volatility & Statistics**
   - `vol_5`, `vol_10`, `vol_20`: Rolling standard deviation
   - `ret_roll_min_20`, `ret_roll_max_20`: Min/max trong 20 ngày
   - `ret_z_20`: Z-score của return
   - `mean_ret_5`, `mean_ret_10`, `mean_ret_20`: Rolling mean returns

4. **Price Indicators**
   - `sma10`, `sma20`: Simple Moving Average
   - `price_trend_10`, `price_trend_20`: Price trend relative to SMA
   - `rsi_14`: Relative Strength Index (14 periods)
   - `bb_width_20`: Bollinger Bands width

5. **V6 Selective Features**
   - `roc_10`, `roc_20`: Rate of Change (10, 20 days)
   - `momentum_10`, `momentum_20`: Price momentum
   - `volume_ratio`: Current volume / average volume (20 days)

6. **Calendar Features**
   - `dow`: Day of week (0-6)
   - `month`: Month (1-12)

### 2. Model Prediction

1. **Scale Features**: Sử dụng StandardScaler đã được train
2. **Predict Return**: ElasticNet model dự đoán log return
3. **Calibration**: Áp dụng LinearRegression calibration
4. **Convert to Price**: `price = current_price * exp(predicted_return)`
5. **Ensemble** (optional): Blend với naive model (giữ nguyên giá)

### 3. Multi-step Forecasting

Để dự đoán N ngày:

1. Bắt đầu với historical data buffers
2. Với mỗi bước:
   - Tính features từ buffers hiện tại
   - Dự đoán return và price cho ngày tiếp theo
   - Cập nhật buffers với prediction
   - Chuyển sang ngày tiếp theo (business day)
3. Lặp lại cho đến khi đủ N ngày

## ⚙️ Luồng vận hành

1. **Clone & cài đặt**
   ```bash
   git clone <repo>
   cd project6.1
   pip install -r requirements.txt
   ```
2. **Chạy API**
   ```bash
   uvicorn app.main:app --reload
   ```
   - Nếu models đã có: tự động load và in `✅ Models loaded successfully`.
   - Nếu models thiếu: API hỏi `Train models now? (y/n)`:
     - **y** → Script `model_trainer` chạy ngay (cần `data/raw/FPT_train.csv`, mất vài phút). Sau khi train, models được load tự động.
     - **n** → API vẫn chạy nhưng các endpoint dự đoán báo lỗi cho tới khi bạn train (chạy `python export_models.py` hoặc trả lời `y` lần tới).
3. **Triển khai production**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```
4. **Sử dụng**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## ⚡ QUICKSTART

1. **Install & run**
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
2. **(Tuỳ chọn) Train thủ công**
   ```bash
   # Khi muốn chủ động train trước
   python export_models.py          # Train nếu thiếu models
   python export_models.py --force  # Bắt buộc retrain
   ```
3. **Kiểm tra API**
   ```bash
   python example_usage.py   # Health, model info, single/multi/full predict
   ```
   hoặc dùng Swagger UI để gửi request thử nghiệm.

## 📡 Sử dụng API

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/api/v1/model/info
```

## 🔌 API Endpoints

### 1. `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running",
  "models_loaded": true
}
```

### 2. `GET /api/v1/model/info`

Lấy thông tin về model đã load.

**Response:**
```json
{
  "status": "loaded",
  "model_type": "ElasticNet",
  "features_count": 39,
  "config": {
    "window_size": 252,
    "window_type": "sliding",
    "alpha": 0.0005,
    "l1_ratio": 0.8,
    "ensemble_weight": 0.0
  }
}
```

### 3. `POST /api/v1/predict/single`

Dự đoán giá cho **1 ngày tiếp theo**.

**Request:**
```json
{
  "historical_data": [
    {
      "time": "2025-03-01",
      "open": 120.0,
      "high": 122.0,
      "low": 119.0,
      "close": 121.0,
      "volume": 1000000
    },
    ...
  ]
}
```

**Response:**
```json
{
  "predicted_price": 121.5,
  "predicted_return": 0.004132,
  "forecast_date": "2025-03-11"
}
```

**Yêu cầu**: Tối thiểu 20 ngày dữ liệu lịch sử.

### 4. `POST /api/v1/predict/multi`

Dự đoán giá cho **N ngày** (1-100 ngày).

**Request:**
```json
{
  "historical_data": [...],
  "n_steps": 30
}
```

**Response:**
```json
{
  "predictions": [
    {
      "date": "2025-03-11",
      "price": 121.5,
      "return": 0.004132
    },
    ...
  ],
  "n_steps": 30
}
```

### 5. `POST /api/v1/predict/full`

Dự đoán giá cho **100 ngày** (như baseline).

**Request:**
```json
{
  "historical_data": [...]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id": 1,
      "date": "2025-03-11",
      "price": 121.5,
      "return": 0.004132
    },
    ...
  ]
}
```

## 💡 Ví dụ sử dụng

### Python

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Historical data (example)
historical_data = [
    {
        "time": "2025-03-01",
        "open": 120.0,
        "high": 122.0,
        "low": 119.0,
        "close": 121.0,
        "volume": 1000000
    },
    # ... thêm nhiều ngày hơn
]

# Single prediction
response = requests.post(
    f"{BASE_URL}/api/v1/predict/single",
    json={"historical_data": historical_data}
)
result = response.json()
print(f"Predicted price: {result['predicted_price']}")

# Multi-step prediction (30 days)
response = requests.post(
    f"{BASE_URL}/api/v1/predict/multi",
    json={
        "historical_data": historical_data,
        "n_steps": 30
    }
)
result = response.json()
print(f"Forecasted {result['n_steps']} days")

# Full 100-day prediction
response = requests.post(
    f"{BASE_URL}/api/v1/predict/full",
    json={"historical_data": historical_data}
)
result = response.json()
print(f"Forecasted {len(result['predictions'])} days")
```

### cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": [
      {
        "time": "2025-03-01",
        "open": 120.0,
        "high": 122.0,
        "low": 119.0,
        "close": 121.0,
        "volume": 1000000
      }
    ]
  }'

# Multi-step prediction
curl -X POST "http://localhost:8000/api/v1/predict/multi" \
  -H "Content-Type: application/json" \
  -d '{
    "historical_data": [...],
    "n_steps": 30
  }'
```

### JavaScript/TypeScript

```javascript
const BASE_URL = "http://localhost:8000";

const historicalData = [
  {
    time: "2025-03-01",
    open: 120.0,
    high: 122.0,
    low: 119.0,
    close: 121.0,
    volume: 1000000
  },
  // ... more data
];

// Single prediction
fetch(`${BASE_URL}/api/v1/predict/single`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ historical_data: historicalData })
})
  .then(res => res.json())
  .then(data => console.log("Predicted price:", data.predicted_price));
```

## ⚠️ Lưu ý quan trọng

1. **Dữ liệu đầu vào**: Cần tối thiểu 20 ngày dữ liệu lịch sử để tính đủ features
2. **Thứ tự dữ liệu**: Dữ liệu phải được sắp xếp theo thời gian (tăng dần)
3. **Model files**: Đảm bảo các file model đã được export và nằm trong `app/models/artifacts/`
4. **Business days**: Forecasting tự động bỏ qua weekends (chỉ tính business days)
5. **Feature consistency**: Features được tính toán giống hệt baseline để đảm bảo tính nhất quán

## 🐛 Troubleshooting

### Models không load được

- Kiểm tra các file trong `app/models/artifacts/` có tồn tại không
- Chạy script `export_models.py` để tạo lại models

### Lỗi "Not enough historical data"

- Cần tối thiểu 20 ngày dữ liệu
- Đảm bảo dữ liệu được sắp xếp theo thời gian

### Lỗi validation

- Kiểm tra format dữ liệu: `time` phải là YYYY-MM-DD
- `high >= low`, `close` phải nằm trong [low, high]
- Tất cả giá trị phải > 0

## ❓ FAQ

### Q: Tôi có cần chạy `export_models.py` trước khi chạy API không?

**A**: **KHÔNG CẦN!**

Chỉ cần chạy `uvicorn app.main:app` ngay. API sẽ tự động:
- Kiểm tra models khi khởi động
- Hỏi bạn có muốn train không nếu models chưa có
- Tự động train nếu bạn đồng ý

Xem chi tiết trong [SETUP_GUIDE.md](SETUP_GUIDE.md) hoặc [QUICKSTART.md](QUICKSTART.md).

## 📝 License

Project này được phát triển cho mục đích học tập và nghiên cứu.

## 👥 Tác giả

Dựa trên baseline V6 từ notebook `improved_v6_selective_features.ipynb`

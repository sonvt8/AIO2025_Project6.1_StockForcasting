# FPT Stock Prediction API

ElasticNet selective-features baseline (+100 day forecasting) được triển khai thành FastAPI backend + Streamlit UI. README này gộp nội dung quan trọng từ các tài liệu khác, giúp bạn hiểu kiến trúc, pipeline, cách khởi tạo và chạy lại toàn bộ project (kể cả luồng realtime data).

---

## 1. Kiến trúc & cấu trúc thư mục

```text
project6.1/
├── app/
│   ├── api/
│   │   ├── routes.py          # Định nghĩa tất cả API endpoints
│   │   └── schemas.py         # Pydantic schemas (request/response)
│   ├── services/
│   │   ├── feature_engineering.py  # Xây features V6 (39 features)
│   │   ├── forecast_service.py     # Logic multi-step forecasting
│   │   ├── model_service.py        # Quản lý model loader + forecast service
│   │   └── data_fetcher.py         # Fetch dữ liệu FPT realtime bằng vnstock
│   ├── models/
│   │   ├── model_loader.py    # Load ElasticNet + scaler + calibration
│   │   └── artifacts/         # Các file model: *.pkl, model_config.json
│   ├── utils/
│   │   ├── data_processing.py # Chuẩn bị dữ liệu, winsorize, buffers
│   │   ├── model_trainer.py   # Train & export model từ FPT_train.csv
│   │   └── helpers.py         # Hàm phụ: RSI, rolling stats, …
│   ├── config.py              # Cấu hình features, đường dẫn, model config
│   └── main.py                # FastAPI app (entrypoint)
├── data/
│   └── raw/
│       └── FPT_train.csv      # Dataset gốc (2020–2025)
├── frontend/
│   └── streamlit_app/
│       ├── app.py             # UI demo: chart + bảng forecast
│       └── assets/…           # CSS, JS, components
├── notebooks/
│   └── baseline.ipynb         # Notebook baseline V6 selective-features
├── test_data_fetcher.py       # Test fetch dữ liệu realtime + metadata
├── test_api.py                # Test end-to-end API
├── pyproject.toml / requirements.txt
└── README.md
```

**Tóm tắt kiến trúc:**
- **Model**: ElasticNet (V6 selective features), train từ `FPT_train.csv` rồi export vào `app/models/artifacts/`.
- **API**: FastAPI đọc artifacts, xử lý input, dự báo single/multi/full/realtime.
- **Realtime**: `data_fetcher.py` dùng vnstock để lấy phần dữ liệu mới, merge với dataset gốc, rồi tái sử dụng cùng pipeline.
- **UI**: Streamlit sử dụng API để hiển thị forecast và metadata.

---

## 2. Feature engineering & pipeline

Từ dữ liệu OHLCV (`time, open, high, low, close, volume`), pipeline xây **39 features**:

- **Base returns & volume changes**
  - `ret_1d_clipped`: log-return ngày (winsorized)
  - `vol_chg_clipped`: log-volume-change (winsorized)
- **Lag features**
  - `ret_lag1` → `ret_lag10`
  - `vol_lag1` → `vol_lag5`
- **Volatility & statistics**
  - `vol_5`, `vol_10`, `vol_20` (rolling std)
  - `ret_roll_min_20`, `ret_roll_max_20`
  - `ret_z_20` (z-score)
  - `mean_ret_5`, `mean_ret_10`, `mean_ret_20`
- **Price indicators**
  - `sma10`, `sma20`
  - `price_trend_10`, `price_trend_20`
  - `rsi_14`
  - `bb_width_20`
- **V6 selective features**
  - `roc_10`, `roc_20`
  - `momentum_10`, `momentum_20`
  - `volume_ratio`
- **Calendar**
  - `dow` (day-of-week), `month`

**Model pipeline:**
1. Chuẩn hóa features bằng `StandardScaler` (fit trên training window).
2. ElasticNet dự đoán log-return ngày tiếp theo.
3. Áp calibration bằng LinearRegression để hiệu chỉnh bias.
4. Chuyển sang giá: `price_next = price_today * exp(predicted_return)`.
5. Multi-step forecasting: lặp lại bước 1–4, cập nhật buffers, tăng ngày theo business-day.

> **Quan trọng:** Khi fetch dữ liệu mới → **KHÔNG retrain model**. Chỉ tính lại features và dự báo bằng model đã train.

---

## 3. Chuẩn bị môi trường

```bash
git clone <repo>
cd project6.1
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

- Model artifacts cần có trong `app/models/artifacts/`:
  - `elasticnet_model.pkl`
  - `scaler.pkl`
  - `calibration_model.pkl`
  - `model_config.json`
- Nếu chưa có, dùng script training (xem mục 4.2).

> Nếu dùng vnstock 0.x trên Windows và gặp lỗi liên quan emoji/encoding:
> ```powershell
> $env:PYTHONIOENCODING="utf-8"
> ```

---

## 4. Training vs Prediction

### 4.1. Training (chạy khi cần build/rebuild model)

Nguồn dữ liệu training: `data/raw/FPT_train.csv`.

```bash
# Ví dụ (tùy file script thực tế):
python -m app.utils.model_trainer        # Train và export artifacts
```

Kết quả: các file `.pkl` + `model_config.json` được ghi vào `app/models/artifacts/`. Sau đó, mọi request prediction sẽ dùng đúng model này.

### 4.2. Prediction (runtime)

- API load model qua `ModelLoader` khi start server (hoặc khi nhận request đầu tiên).
- Các endpoint `/predict/single`, `/multi`, `/full` nhận `historical_data` từ client.
- Endpoint `/predict/realtime` **tự động**:
  1. Đọc `FPT_train.csv` để biết last date đang có.
  2. Dùng vnstock để lấy giá FPT từ ngày sau đó đến ngày hiện tại.
  3. Merge, winsorize, build features và dự báo N ngày tới.

Model **không thay đổi** trừ khi bạn chạy lại training script.

---

## 5. Chạy API & Streamlit

```bash
uvicorn app.main:app --reload
```

- Swagger: `http://localhost:8000/docs`
- ReDoc:   `http://localhost:8000/redoc`

UI demo (tùy chọn):
```bash
streamlit run frontend/streamlit_app/app.py
```

Trong UI có toggle “Use Realtime Data from Internet” → khi bật, app sẽ gọi `/api/v1/predict/realtime` và hiển thị forecast + metadata fetch.

---

## 6. Luồng realtime data (tóm tắt)

1. User gọi `POST /api/v1/predict/realtime` với payload:
   ```json
   {
     "n_steps": 30,
     "historical_days": 120
   }
   ```
2. `data_fetcher`:
   - Đọc `FPT_train.csv` → lấy `last_date` hiện có.
   - Chỉ fetch từ `last_date + 1 BDay` đến ngày hôm nay (bằng vnstock).
   - Merge vào dataframe, loại trùng, sort theo thời gian.
3. `forecast_service`:
   - Tính lại returns, winsorize theo config baseline.
   - Build features V6, chuẩn hóa và dùng ElasticNet để dự báo N bước.
4. API format kết quả:
   ```json
   {
     "fetched_data_count": 1332,
     "latest_date": "2025-11-28",
     "fetched_new_data": true,
     "previous_last_date": "2025-03-10",
     "predictions": [...],
     "n_steps": 30
   }
   ```

---

## 7. Kiểm thử nhanh

```bash
# 1. Kiểm tra luồng fetch + metadata
python test_data_fetcher.py

# 2. (Sau khi chạy uvicorn) – kiểm tra các endpoint chính
python test_api.py
```

Các script sẽ in:
- `Last date in dataset`
- Số bản ghi tổng cộng sau khi merge
- Metadata `fetched_new_data`, `previous_last_date`, `latest_date`
- Tóm tắt predictions (giá min/max/avg, số ngày dự báo)

---

## 8. API endpoints chính

| Method | Endpoint                    | Mô tả ngắn gọn                                 |
|--------|----------------------------|-----------------------------------------------|
| GET    | `/health`                  | Kiểm tra trạng thái API + models_loaded       |
| GET    | `/api/v1/model/info`       | Thông tin model (type, số features, config)   |
| POST   | `/api/v1/predict/single`   | Dự báo 1 ngày tiếp theo                       |
| POST   | `/api/v1/predict/multi`    | Dự báo N ngày (1–100)                         |
| POST   | `/api/v1/predict/full`     | Dự báo 100 ngày (chuẩn baseline)              |
| POST   | `/api/v1/predict/realtime` | Tự fetch dữ liệu FPT mới nhất rồi dự báo      |

Payload mẫu (single/multi/full):
```json
{
  "historical_data": [
    {
      "time": "2025-02-20",
      "open": 92.0,
      "high": 93.5,
      "low": 91.0,
      "close": 92.8,
      "volume": 1500000
    }
    // ... ≥ 20 bản ghi, sắp xếp tăng dần theo time
  ],
  "n_steps": 30
}
```

---

## 9. Quy trình gợi ý để “reproduce” kết quả

1. **Cài đặt**: `pip install -r requirements.txt` trong venv.
2. **(Nếu cần) Train lại model**: `python -m app.utils.model_trainer`.
3. **Chạy API**: `uvicorn app.main:app --reload`.
4. **Xác thực**:
   - Dùng Swagger để gọi `/health`, `/api/v1/model/info`.
   - Gửi thử `/api/v1/predict/multi` với lịch sử lấy từ `FPT_train.csv`.
5. **Realtime**:
   - `python test_data_fetcher.py` để chắc chắn fetch hoạt động.
   - Gọi `/api/v1/predict/realtime` từ Swagger hoặc `test_api.py`.
6. **UI (tùy chọn)**: chạy Streamlit và bật chế độ “Use Realtime Data from Internet”.

---

## 10. Ghi chú & nguồn

- Dữ liệu đầu vào phải hợp lệ (giá > 0, `high ≥ low`, `close` trong [low, high]).
- Thời gian dự báo sử dụng business days (bỏ cuối tuần, holidays mặc định theo pandas).
- Project phục vụ mục đích học tập / kiểm tra, không dùng trực tiếp cho trading thật.
- Pipeline dựa trên notebook baseline `improved_v6_selective_features.ipynb` và đã được đóng gói lại thành các service/module trong thư mục `app/`.

Chúc bạn chạy lại kết quả nhanh chóng và dễ dàng mở rộng thêm tính năng! 🎯

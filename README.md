# FPT Stock Prediction API (v2 - PatchTST)

Phiên bản v2 thay thế hoàn toàn baseline ElasticNet v1 bằng mô hình PatchTST kết hợp hai kỹ thuật hậu huấn luyện:
- Post-processing Regression (Linear Regression)
- Smooth Bias Correction (Linear 20%)

Mục tiêu: giữ nguyên trải nghiệm API và UI, nhưng dự báo tốt hơn theo kết quả notebook baseline_patchtst_v2.ipynb. Toàn bộ artifacts và logic v1 đã được loại bỏ khỏi codebase.

---

## 1. Kiến trúc & cấu trúc thư mục

```text
project6.1/
├── app/
│   ├── api/
│   │   ├── routes.py              # Định nghĩa các API endpoints (giữ tên v1, logic v2)
│   │   └── schemas.py             # Pydantic schemas (request/response)
│   ├── services/
│   │   ├── forecast_service.py    # Forecast v2: PatchTST + post-processing + smooth
│   │   ├── model_service.py       # Kết nối loader + forecast service v2
│   │   └── data_fetcher.py        # Fetch dữ liệu FPT realtime bằng vnstock
│   ├── models/
│   │   ├── patchtst_loader.py     # Loader PatchTST + auto-download artifacts (Releases)
│   │   └── artifacts/             # Artifacts v2: patchtst.pt, *.json, post_model.pkl
│   ├── utils/
│   │   └── model_trainer.py       # Train PatchTST v2 và export artifacts
│   ├── config.py                  # Cấu hình API, PatchTST params, Release info
│   └── main.py                    # FastAPI app (entrypoint)
├── data/
│   └── raw/
│       └── FPT_train.csv          # Dataset gốc (2020–2025)
├── frontend/
│   └── streamlit_app/
│       ├── app.py                 # UI demo: chart + bảng forecast
│       └── assets/…               # CSS, JS, components
├── notebooks/
│   └── baseline_patchtst_v2.ipynb # Notebook tham chiếu phương pháp tốt nhất
├── test_data_fetcher.py           # Test fetch dữ liệu realtime + metadata
├── test_api.py                    # Test end-to-end API (health, model info, realtime)
├── pyproject.toml / requirements.txt
└── README.md
```

Tóm tắt kiến trúc v2:
- Model: PatchTST (hparams cố định từ notebook), huấn luyện trên close series và xuất state_dict.
- Hậu huấn luyện: LinearRegression post-processing + Smooth Bias Correction (Linear 20%).
- API: giữ nguyên endpoints cũ nhưng logic dự báo dùng v2; trả về date, price, return như trước để UI chạy không cần đổi.
- Realtime: dùng vnstock fetch phần dữ liệu mới, hợp nhất với dataset gốc để hiển thị chart và lấy mốc thời gian; mô hình không retrain khi runtime.

---

## 1.1. Dataset Detection Logic

Giữ nguyên hành vi như v1, nhưng chỉ dùng cho fetch hiển thị và mốc thời gian:
- Tự động tìm file CSV trong `data/raw/` có "train" trong tên (không phân biệt hoa/thường)
- Ví dụ: `FPT_train.csv`, `train_YYYYMMDD.csv`, `my_train_data.csv`
- Khi có dataset: dùng TẤT CẢ dữ liệu trong file + fetch mới (nếu có) để trả lịch sử hiển thị; model không retrain.
- Khi không có dataset: cho phép upload hoặc fetch từ internet dựa theo slider.

Định dạng yêu cầu xem thêm mục 11. Data Format Reference.

---

## 2. Mô hình & Pipeline v2

- Base model: PatchTST (NeuralForecast)
- Hparams (cố định):
  - input_size: 100
  - patch_len: 32
  - stride: 4
  - learning_rate: 0.001610814898983045
  - max_steps: 250
  - revin: True
  - horizon: 100
- Hậu huấn luyện:
  - Post-processing Regression: LinearRegression map y_pred → y_true (train theo TimeSeriesSplit)
  - Smooth Bias Correction: Linear 20% (đầu giữ baseline, cuối dùng post-processing)

Suy diễn (inference) không retrain:
- Từ chuỗi close lịch sử hiện tại, lấy cửa sổ `input_size` cuối → forward PatchTST → dự báo `horizon` bước → áp LinearRegression → áp Smooth Linear 20% → clamp non-negative.
- Tính log-return để giữ tương thích với UI.

---

## 3. Chuẩn bị môi trường

```bash
git clone <repo>
cd project6.1
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

- Model artifacts cần có trong `app/models/artifacts/` (xem mục 4 – Plan A):
  - `patchtst.pt`
  - `best_params.json`
  - `post_model.pkl`
  - `smooth_config.json`

---

## 4. Artifacts & Plan A (GitHub Releases)

V2 sử dụng Plan A: artifacts được phát hành trên GitHub Releases và loader tự động tải về nếu thiếu.

- Release Tag: `version-2.0-patchtst`
- Assets trên Release:
  - `patchtst.pt`
  - `best_params.json`
  - `post_model.pkl`
  - `smooth_config.json`

Cấu hình tải tự động nằm trong `app/config.py` (GITHUB_RELEASE). Khi API khởi chạy hoặc request đầu tiên, nếu thiếu files trong `app/models/artifacts/`, loader sẽ tải từ Release này về.

Lưu ý: Nếu bạn muốn tự tạo artifacts trước khi phát hành Release, xem mục 4.1.

### 4.1. Training & Export (tùy chọn – để tự tạo artifacts)

Nguồn dữ liệu training: `data/raw/FPT_train.csv` (hoặc bất kỳ CSV có "train" trong tên).

```bash
# Train v2 và export artifacts
python -m app.utils.model_trainer
```

Sau khi train, thư mục `app/models/artifacts/` sẽ có đủ 4 files. Hãy upload đúng tên files lên GitHub Release `version-2.0-patchtst`.

---

## 5. Chạy API & Streamlit

### 5.1. Chạy FastAPI Backend

```bash
uvicorn app.main:app --reload
```

- Swagger: `http://localhost:8000/docs`
- ReDoc:   `http://localhost:8000/redoc`

Khi khởi động, API sẽ cố gắng load artifacts. Nếu missing, loader sẽ tự tải từ Release.

### 5.2. Chạy Streamlit UI

```bash
streamlit run frontend/streamlit_app/app.py
```

UI giữ nguyên hành vi hiển thị và gọi API; không cần chỉnh sửa.

---

## 6. Luồng realtime data

- `data_fetcher.py` sẽ:
  1) Tìm dataset trong `data/raw/*train*.csv`
  2) Xác định last date và chỉ fetch phần mới bằng vnstock
  3) Merge vào dataframe đầy đủ cho hiển thị chart
- API dự báo dùng model đã train (không retrain), bắt đầu từ ngày business tiếp theo so với điểm dữ liệu cuối cùng.

---

## 7. Kiểm thử nhanh

```bash
# 1) Kiểm tra /health và /api/v1/model/info
python test_api.py

# 2) (Sau khi chạy uvicorn) – kiểm tra realtime endpoint và xem thống kê
python test_api.py

# 3) Kiểm tra fetch + metadata (không gọi API)
python test_data_fetcher.py
```

Các script sẽ in:
- Trạng thái health, model info
- Thống kê dự báo (min/max/avg, số ngày)
- Metadata fetch (fetched_new_data, previous_last_date, latest_date)

---

## 8. API endpoints

| Method | Endpoint                    | Mô tả ngắn gọn                                   |
|--------|-----------------------------|--------------------------------------------------|
| GET    | `/health`                   | Kiểm tra trạng thái API + models_loaded          |
| GET    | `/api/v1/model/info`        | Thông tin model (type, hparams)                  |
| POST   | `/api/v1/predict/single`    | Dự báo 1 ngày tiếp theo                          |
| POST   | `/api/v1/predict/multi`     | Dự báo N ngày (1–100)                            |
| POST   | `/api/v1/predict/full`      | Dự báo 100 ngày                                  |
| POST   | `/api/v1/predict/realtime`  | Tự fetch dữ liệu FPT mới nhất rồi dự báo         |

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

Response (multi/full/realtime – ví dụ predictions):
```json
{
  "predictions": [
    {"date": "2025-12-04", "price": 104.12, "return": -0.003452},
    ...
  ]
}
```

---

## 9. Gợi ý quy trình “reproduce” v2

1) Cài đặt môi trường: `pip install -r requirements.txt`
2) (Tùy chọn) Tạo artifacts: `python -m app.utils.model_trainer`
3) Phát hành Release `version-2.0-patchtst` và upload 4 files artifacts
4) Chạy API: `uvicorn app.main:app --reload`
5) Dùng Swagger kiểm tra `/health`, `/api/v1/model/info`
6) Dùng `test_api.py` để gọi `/api/v1/predict/realtime` và xem kết quả
7) (Tùy chọn) Chạy Streamlit để xem chart và bảng dự báo

---

## 10. Ghi chú

- Ứng dụng không retrain khi fetch dữ liệu realtime; dữ liệu mới chỉ để hiển thị và mốc thời gian dự báo.
- Sử dụng business days khi tạo ngày dự báo (bỏ cuối tuần theo pandas).
- Project dành cho mục đích học tập/đánh giá; không khuyến nghị dùng trực tiếp cho giao dịch thật.
- Notebook tham chiếu: `notebooks/baseline_patchtst_v2.ipynb`.

---

## 11. Data Format Reference

### 11.1. Các cột trong Dataset

- time: YYYY-MM-DD
- open/high/low/close: số thập phân, đơn vị nghìn VND
- volume: số nguyên (số lượng cổ phiếu)
- symbol: "FPT" (tuỳ chọn trong dữ liệu fetch, UI/logic không bắt buộc)

### 11.2. Đồng bộ đơn vị giá

- Dữ liệu train dùng nghìn VND (ví dụ 96.10 = 96,100 VND)
- Nếu fetch về theo đơn vị VND (giá > 1000), code sẽ tự chia 1000 để đồng bộ

### 11.3. Kiểm tra/clean dữ liệu fetch

- Loại bản ghi có OHLC không hợp lệ (high < low, close ngoài [low, high], ...)
- Lọc giá ngoài khoảng [1, 500] (nghìn VND) để tránh outliers

---

## 12. Lưu trữ artifacts

- Plan A (khuyến nghị): phát hành artifacts trên GitHub Releases với tag `version-2.0-patchtst` và assets:
  - patchtst.pt
  - best_params.json
  - post_model.pkl
  - smooth_config.json
- Loader sẽ tự động tải về vào `app/models/artifacts/` nếu thiếu.
- Không commit artifacts vào repo (đã ignore *.pt, *.ckpt, *.pkl, *.json trong artifacts).

---

Chúc bạn triển khai v2 thuận lợi và dự báo hiệu quả! 🎯

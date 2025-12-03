# FPT Stock Prediction API

ElasticNet selective-features baseline (+100 day forecasting) được triển khai thành FastAPI backend + Streamlit UI. README này được phát triển dựa trên luồng thực nghiệm và đánh giá metric thông qua cuộc thi [Kaggle AIO2025-StockForcasting](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge/) , giúp bạn hiểu kiến trúc, pipeline, cách khởi tạo và chạy lại toàn bộ project (kể cả luồng realtime data).

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
- **Model**: ElasticNet (V6 selective features), train từ `*.csv` trong `data/raw/` rồi export vào `app/models/artifacts/`.
- **API**: FastAPI đọc artifacts, xử lý input, dự báo single/multi/full/realtime.
- **Realtime**: `data_fetcher.py` dùng vnstock để lấy phần dữ liệu mới, merge với dataset gốc, rồi tái sử dụng cùng pipeline.
- **UI**: Streamlit sử dụng API để hiển thị forecast và metadata.

---

## 1.1. Dataset Detection Logic

Hệ thống tự động phát hiện dataset trong thư mục `data/raw/` để quyết định cách xử lý dữ liệu:

**Logic phát hiện:**
- Chỉ tìm kiếm trong thư mục `data/raw/` (từ thư mục gốc project)
- Tìm các file CSV có chứa từ khóa **"train"** trong tên file (không phân biệt hoa/thường)
- Ví dụ: `FPT_train.csv`, `train_20250115.csv`, `my_train_data.csv` đều được nhận diện
- Trả về file đầu tiên tìm thấy nếu có nhiều file khớp

**Hành vi khi có dataset:**
- Hệ thống sử dụng **TẤT CẢ** dữ liệu từ file dataset (ví dụ: từ 2020-08-03)
- Chỉ fetch phần dữ liệu mới từ ngày cuối cùng trong dataset đến ngày hiện tại
- Merge và trả về toàn bộ dữ liệu (dataset gốc + dữ liệu mới fetch)
- Tham số `historical_days` trong slider **KHÔNG ảnh hưởng** khi đã có dataset

**Hành vi khi không có dataset:**
- Hiển thị cảnh báo và message box cho người dùng chọn:
  1. **Upload CSV file**: Upload và lưu file làm dataset (tự động đặt tên với "train" + date)
  2. **Fetch from internet (use slider)**: Dùng slider để fetch dữ liệu từ internet theo số ngày chỉ định
- Chỉ thực hiện fetching sau khi người dùng đã chọn một trong hai phương án

**Lưu ý:**
- File dataset phải có định dạng CSV với các cột: `time`, `open`, `high`, `low`, `close`, `volume`
- Format dữ liệu phải đúng chuẩn (xem mục 11. Data Format Reference)

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

## 3.1. Chạy với Docker (Khuyến nghị)

### 3.1.1. Yêu cầu
- Docker Engine 20.10+
- Docker Compose 2.0+

### 3.1.2. Khởi động ứng dụng

```bash
# Build và khởi động tất cả services
docker-compose up --build

# Hoặc chạy ở chế độ background
docker-compose up -d --build
```

Sau khi khởi động:
- **FastAPI Backend**: http://localhost:8000
  - Swagger UI: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc
- **Streamlit Frontend**: http://localhost:8501

### 3.1.3. Dừng ứng dụng

```bash
# Dừng services
docker-compose down

# Dừng và xóa volumes (xóa dữ liệu)
docker-compose down -v
```

### 3.1.4. Xem logs

```bash
# Xem logs tất cả services
docker-compose logs -f

# Xem logs một service cụ thể
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 3.1.5. Training model trong Docker

```bash
# Chạy training script trong container backend
docker-compose exec backend python -m app.utils.model_trainer

# Hoặc chạy một lệnh tùy chỉnh
docker-compose exec backend python -c "from app.utils.model_trainer import train_and_export_models; train_and_export_models()"
```

### 3.1.6. Cấu trúc Docker

- **Dockerfile**: Base image Python 3.11-slim, cài đặt dependencies
- **docker-compose.yml**:
  - Service `backend`: FastAPI trên port 8000
  - Service `frontend`: Streamlit trên port 8501
  - Volumes: Mount `data/` và `app/models/` để persist dữ liệu
  - Network: Bridge network để các services giao tiếp

### 3.1.7. Lưu ý

- Dữ liệu trong `data/` và `app/models/` được persist qua volumes
- Code được mount vào container để development dễ dàng (có thể disable trong production)
- Streamlit tự động kết nối đến backend qua internal network (`http://backend:8000`)
- Nếu cần thay đổi port, sửa trong `docker-compose.yml`

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
  1. Đọc `*.csv` để biết last date đang có.
  2. Dùng vnstock để lấy giá FPT từ ngày sau đó đến ngày hiện tại.
  3. Merge, winsorize, build features và dự báo N ngày tới.

Model **không thay đổi** trừ khi bạn chạy lại training script.

---

## 5. Chạy API & Streamlit

### 5.1. Chạy với Docker (Khuyến nghị)

Xem mục **3.1. Chạy với Docker** ở trên.

### 5.2. Chạy thủ công (không dùng Docker)

#### 5.2.1. Chạy FastAPI Backend

```bash
uvicorn app.main:app --reload
```

- Swagger: `http://localhost:8000/docs`
- ReDoc:   `http://localhost:8000/redoc`

#### 5.2.2. Chạy Streamlit UI

```bash
streamlit run frontend/streamlit_app/app.py
```

**Lưu ý**: Khi chạy thủ công, đảm bảo FastAPI đang chạy trước khi mở Streamlit.

### 5.3. Hai Chế Độ Prediction trong UI

#### Chế độ "Realtime API" (Mặc định: thứ 2)

**Hành vi:**
1. **Tự động scan** thư mục `data/raw/` để tìm file CSV có chứa "train" trong tên
2. **Nếu tìm thấy dataset:**
   - Hiển thị: "✅ Dataset found: [tên file]"
   - Sử dụng TẤT CẢ dữ liệu từ dataset + fetch phần mới từ internet
   - Tham số slider "Historical days" không ảnh hưởng (chỉ để tương thích ngược)
3. **Nếu không tìm thấy dataset:**
   - Hiển thị cảnh báo và message box với 2 lựa chọn:
     - **Upload CSV file**: Upload và lưu file làm dataset (tự động đặt tên `train_YYYYMMDD.csv`)
     - **Fetch from internet (use slider)**: Dùng slider để fetch dữ liệu từ internet theo số ngày
   - Chỉ thực hiện fetching sau khi người dùng đã chọn một phương án
4. Gọi API `/api/v1/predict/realtime` và hiển thị forecast + metadata fetch

#### Chế độ "Upload CSV → API" (Mặc định: thứ 1)

**Hành vi:**
1. Người dùng upload file CSV trực tiếp
2. File được sử dụng **chỉ để dự đoán** tại thời điểm đó
3. **KHÔNG có tùy chọn lưu file** (file upload chỉ phục vụ mục đích prediction)
4. Gọi API `/api/v1/predict/multi` với dữ liệu từ file đã upload
5. Hiển thị kết quả forecast

**Lưu ý:**
- Chế độ này phù hợp khi bạn muốn test với dữ liệu tùy chỉnh mà không cần lưu vào dataset
- Nếu muốn lưu file làm dataset, hãy dùng chế độ "Realtime API" và chọn "Upload CSV file"

---

## 6. Luồng realtime data (chi tiết)

### 6.1. Dataset Detection (Bước đầu tiên)

Khi người dùng chọn chế độ "Realtime API" trong Streamlit UI:

1. **Hệ thống tự động scan** thư mục `data/raw/` để tìm file CSV có chứa "train" trong tên
2. **Nếu tìm thấy dataset:**
   - Hiển thị thông báo: "✅ Dataset found: [tên file]"
   - Hiển thị slider "Historical days" (tham số này không ảnh hưởng khi đã có dataset)
   - Tự động cho phép thực hiện fetching
3. **Nếu không tìm thấy dataset:**
   - Hiển thị cảnh báo: "⚠️ No dataset found in data/raw/ (file with 'train' in name)"
   - Hiển thị message box với 2 lựa chọn:
     - **Upload CSV file**: Upload và lưu file làm dataset (tự động đặt tên `train_YYYYMMDD.csv`)
     - **Fetch from internet (use slider)**: Dùng slider để fetch dữ liệu từ internet
   - Chỉ thực hiện fetching sau khi người dùng đã chọn một phương án

### 6.2. Data Fetching Process

Sau khi đã xác định dataset (hoặc quyết định fetch từ internet):

1. **User gọi `POST /api/v1/predict/realtime`** với payload:
   ```json
   {
     "n_steps": 30,
     "historical_days": 120
   }
   ```
   > **Lưu ý**: `historical_days` chỉ có tác dụng khi **KHÔNG có dataset**. Khi đã có dataset, hệ thống sử dụng TẤT CẢ dữ liệu từ dataset.

2. **`data_fetcher` xử lý:**
   - **Nếu có dataset**:
     - Đọc file dataset (ví dụ: `FPT_train.csv`) → lấy `last_date` hiện có
     - Chỉ fetch phần dữ liệu mới từ `last_date + 1 BDay` đến ngày hôm nay (bằng vnstock)
     - Merge vào dataframe, loại trùng, sort theo thời gian
     - Trả về **TẤT CẢ** dữ liệu (dataset gốc + dữ liệu mới)
   - **Nếu không có dataset**:
     - Fetch dữ liệu từ internet theo số ngày chỉ định trong `historical_days`
     - Trả về dữ liệu đã fetch

3. **`forecast_service` xử lý:**
   - Tính lại returns, winsorize theo config baseline
   - Build features V6, chuẩn hóa và dùng ElasticNet để dự báo N bước

4. **API format kết quả:**
   ```json
   {
     "fetched_data_count": 1332,
     "latest_date": "2025-11-28",
     "fetched_new_data": true,
     "previous_last_date": "2025-03-10",
     "predictions": [...],
     "n_steps": 30,
     "historical_data": [...]  // Tất cả dữ liệu lịch sử để hiển thị chart
   }
   ```

### 6.3. Upload và Lưu Dataset (Trong Realtime Mode)

Khi người dùng chọn "Upload CSV file" trong realtime mode:

1. Upload file CSV qua file uploader
2. Nhấn nút "💾 Save as Dataset and Proceed"
3. File được lưu vào `data/raw/` với tên format: `train_YYYYMMDD.csv` (hoặc `train_[tên_gốc]_YYYYMMDD.csv`)
4. Hệ thống tự động refresh và phát hiện dataset mới
5. Tiếp tục thực hiện fetching với dataset vừa lưu

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
- Pipeline dựa trên notebook baseline `baseline_elastic_v1.ipynb` và đã được đóng gói lại thành các service/module trong thư mục `app/`.

---

## 11. Data Format Reference

### 11.1. Format của các cột trong Dataset

Dataset phải là file CSV với các cột sau:

#### Cột `time` (Date/Time)
- **Format**: `YYYY-MM-DD` (ví dụ: `2020-08-03`)
- **Type**: String (được parse thành datetime trong code)
- **Ví dụ**: `2020-08-03`, `2025-03-10`
- **Đồng bộ với dữ liệu fetch**: ✅ Cùng format `YYYY-MM-DD`

#### Cột `open`, `high`, `low`, `close` (Price)
- **Format**: Số thập phân, đơn vị **nghìn VND**
- **Ví dụ**:
  - `19.07` = 19,070 VND
  - `121.92` = 121,920 VND
- **Phạm vi trong training data**: ~19 đến ~132 (nghìn VND)
- **Đồng bộ với dữ liệu fetch**:
  - ✅ Dữ liệu fetch từ API có thể ở đơn vị VND (ví dụ: 96100 VND)
  - ✅ Code tự động normalize: chia cho 1000 để chuyển thành nghìn VND (96.10)
  - ✅ Đảm bảo tất cả price columns (open, high, low, close) được normalize cùng lúc

#### Cột `volume` (Trading Volume)
- **Format**: Số nguyên, đơn vị **số lượng cổ phiếu**
- **Ví dụ**:
  - `1392200` = 1,392,200 cổ phiếu
  - `2966941` = 2,966,941 cổ phiếu
- **Phạm vi trong training data**: ~500,000 đến ~13,000,000
- **Đồng bộ với dữ liệu fetch**:
  - ✅ Volume không cần normalize (đơn vị là số lượng cổ phiếu, không phụ thuộc vào đơn vị tiền tệ)
  - ✅ Code không thay đổi volume khi normalize prices

#### Cột `symbol`
- **Format**: String, giá trị cố định `"FPT"`
- **Mục đích**: Identifier cho cổ phiếu
- **Đồng bộ với dữ liệu fetch**: ✅ Tự động thêm `"FPT"` vào dữ liệu fetch

### 11.2. Tóm tắt Format

| Cột | Format | Đơn vị | Có normalize? | Ghi chú |
|-----|--------|--------|----------------|---------|
| `time` | YYYY-MM-DD | Date | ❌ | Cùng format |
| `open` | Số thập phân | Nghìn VND | ✅ | Tự động chia 1000 nếu > 1000 |
| `high` | Số thập phân | Nghìn VND | ✅ | Tự động chia 1000 nếu > 1000 |
| `low` | Số thập phân | Nghìn VND | ✅ | Tự động chia 1000 nếu > 1000 |
| `close` | Số thập phân | Nghìn VND | ✅ | Tự động chia 1000 nếu > 1000 |
| `volume` | Số nguyên | Số lượng cổ phiếu | ❌ | Không cần normalize |
| `symbol` | String | "FPT" | ❌ | Tự động thêm |

### 11.3. Logic Normalization

Khi fetch dữ liệu mới từ API:

1. **Kiểm tra**: Nếu bất kỳ giá trị price nào > 1000 → có thể đang ở đơn vị VND
2. **Normalize**: Chia tất cả price columns (open, high, low, close) cho 1000
3. **Đảm bảo**: Tất cả price columns được normalize cùng lúc để giữ tính nhất quán
4. **Volume**: Không thay đổi (đơn vị độc lập với đơn vị tiền tệ)

### 11.4. Ví dụ Normalization

**Trước normalize:**
```python
{
    "time": "2025-11-28",
    "open": 96100.00,    # VND
    "high": 96500.00,    # VND
    "low": 95800.00,     # VND
    "close": 96100.00,   # VND
    "volume": 5000000,   # Số lượng cổ phiếu (không đổi)
    "symbol": "FPT"
}
```

**Sau normalize:**
```python
{
    "time": "2025-11-28",
    "open": 96.10,       # Nghìn VND (96100 / 1000)
    "high": 96.50,       # Nghìn VND (96500 / 1000)
    "low": 95.80,        # Nghìn VND (95800 / 1000)
    "close": 96.10,      # Nghìn VND (96100 / 1000)
    "volume": 5000000,   # Số lượng cổ phiếu (không đổi)
    "symbol": "FPT"
}
```

---

Chúc bạn chạy lại kết quả nhanh chóng và dễ dàng mở rộng thêm tính năng! 🎯

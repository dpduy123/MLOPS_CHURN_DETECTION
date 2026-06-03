# Churn Prediction MLOps Stack

## Kiến trúc tổng quan

```
                         ┌──────────────────────────────────────────┐
                         │              LGTM Stack                  │
  Client                 │                                          │
    │                    │  Promtail ──► Loki ──► Grafana ◄── Prometheus
    │  /predict          │                           │         ▲    ▲
    ▼  /predict_batch    │                    alert email      │    │
  BentoML ──────────────────────────────────────────┘    scrape    │
    │  /metrics  ────────────────────────────────────────────────  │
    │                                                               │
    │  POST /log ──► Evidently Monitor ─── /metrics ───────────────┘
    │                      │
    │                 HTML Reports
    ▼
  MLflow (model registry)
```

### Flow chính

| Bước | Mô tả |
|------|-------|
| **Build** | `scripts/build.sh` export model champion từ MLflow → `bentoml build` → `bentoml containerize` |
| **Serve** | BentoML serve `/predict` và `/predict_batch` (adaptive batching) |
| **Metrics** | BentoML expose Prometheus metrics qua `/metrics` |
| **Drift** | BentoML POST prediction logs sang Evidently Monitor mỗi request |
| **Alert** | Prometheus scrape → Grafana evaluate rule `churn_low_confidence_streak >= 200` → Grafana gửi email qua SMTP |

---

## Cấu trúc thư mục

```
deployments/
├── .env                          # Secrets (SMTP, passwords)
├── docker-compose.yml            # Full stack
├── bentoml/
│   ├── service.py                # BentoML service (predict + predict_batch + metrics)
│   ├── bentofile.yaml            # Build spec (include model_artifacts/)
│   └── requirements.txt
├── monitoring/
│   ├── evidently/
│   │   ├── monitor.py            # FastAPI + Evidently + Prometheus metrics
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── churn_mlops.json  # Pre-built dashboard
│   │   └── provisioning/
│   │       ├── datasources/      # Prometheus + Loki + Tempo
│   │       ├── dashboards/
│   │       └── alerting/
│   │           └── alerting.yaml # Alert rules + email contact points
│   ├── loki/loki-config.yaml
│   ├── promtail/promtail-config.yaml
│   └── prometheus/prometheus.yml
└── scripts/
    ├── export_model.py           # Export champion model từ MLflow
    └── build.sh                  # Full build pipeline
```

---

## Hướng dẫn sử dụng

### 1. Chuẩn bị

```bash
# Copy và điền thông tin thực tế
cp deployments/.env deployments/.env.local
# Chỉnh sửa: SMTP_USER, SMTP_PASSWORD, GRAFANA_ADMIN_PASSWORD
```

### 2. Thêm email nhận alert

Mở `monitoring/grafana/provisioning/alerting/alerting.yaml`, tìm `addresses` và sửa:

```yaml
addresses: >-
  mlops-lead@company.com;
  data-scientist@company.com;
  platform-oncall@company.com
```

### 3. Build image (export model → dockerize)

```bash
# Đảm bảo MLflow server đang chạy và có model @champion
cd deployments

MLFLOW_TRACKING_URI=http://localhost:5100 \
MODEL_VERSION=1.0.0 \
bash scripts/build.sh
```

Script sẽ:
1. Chạy `scripts/export_model.py` → tải model + code về `model_artifacts/`
2. `bentoml build` → bundle model vào Bento
3. `bentoml containerize` → tạo Docker image `churn-prediction:latest`

> **Lưu ý:** model được embed vào image. Container KHÔNG cần kết nối MLflow lúc serving.

### 4. Đặt reference data cho Evidently

```bash
# reference.parquet = dữ liệu training (features + prediction column)
cp /path/to/training_data.parquet deployments/data/reference.parquet
```

### 5. Khởi động toàn bộ stack

```bash
cd deployments
docker compose --env-file .env up -d
```

### 6. Kiểm tra

| URL | Mô tả |
|-----|-------|
| `http://localhost:3000/healthz` | BentoML health |
| `http://localhost:3000/metrics` | Prometheus metrics |
| `http://localhost:3000/docs` | Swagger UI |
| `http://localhost:5100` | MLflow UI |
| `http://localhost:8001/health` | Evidently Monitor |
| `http://localhost:8001/report/latest` | Latest Evidently HTML report |
| `http://localhost:9090` | Prometheus |
| `http://localhost:3001` | Grafana (admin / mlops_secret) |

---

## API Usage

### Single predict

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["tenure", "monthly_charges", "total_charges"],
      "data": [[12, 65.5, 786.0]]
    }
  }'
```

Response:
```json
{
  "predictions": [1],
  "labels": ["Rời bỏ"],
  "confidences": [0.82],
  "row_count": 1,
  "low_conf_streak": 0
}
```

### Batch predict

```bash
curl -X POST http://localhost:3000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["tenure", "monthly_charges", "total_charges"],
      "data": [
        [12, 65.5, 786.0],
        [36, 45.0, 1620.0],
        [2,  90.0, 180.0]
      ]
    }
  }'
```

BentoML tự động gom các request đến `/predict_batch` thành batch
(tối đa 512 rows, max wait 500ms) trước khi gọi model 1 lần duy nhất.

---

## Alert Logic

```
BentoML                    Prometheus                  Grafana
  │                            │                          │
  │── churn_low_confidence ──►  │── scrape every 15s ──►  │
  │   _streak (gauge)           │                          │
  │   (tăng mỗi request        │   evaluate rule          │
  │    all rows conf < 0.60,   │   mỗi 1 phút             │
  │    reset nếu có ≥ 0.60)    │                          │
  │                            │   streak >= 200?         │
  │                            │                 YES ──►  │── SMTP email ──► team
```

**Email được gửi bởi Grafana** (không phải BentoML).  
Cấu hình SMTP trong `.env` → Grafana đọc qua `GF_SMTP_*`.  
Contact points và recipients định nghĩa trong `alerting/alerting.yaml`.

---

## Tag semver cho model trong MLflow

Để `export_model.py` validate version 1.0.0:

```python
import mlflow
client = mlflow.MlflowClient("http://localhost:5100")

# Sau khi register model version (ví dụ version số 3)
client.set_model_version_tag(
    name="Churn_Predict",
    version="3",
    key="semver",
    value="1.0.0"
)

# Set alias champion → version 3
client.set_registered_model_alias(
    name="Churn_Predict",
    alias="champion",
    version="3"
)
```

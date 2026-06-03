"""
service.py  –  BentoML Churn Prediction Service  (v2)
═══════════════════════════════════════════════════════
Features
--------
  • Single-row  /predict       – JSON in / JSON out
  • Batch       /predict_batch – gửi nhiều row cùng lúc
  • Health      /healthz
  • Metrics     /metrics        – Prometheus exposition (scraped by Prometheus)

Confidence alert logic
-----------------------
  Service ghi mỗi prediction (score + timestamp) vào một sliding window.
  Khi 200 request liên tiếp đều có max-confidence < LOW_CONF_THRESHOLD (0.60),
  service expose metric  churn_low_confidence_streak (gauge) = số request trong
  streak hiện tại.  Grafana alert rule đọc metric này và gửi email qua SMTP
  (cấu hình trong grafana/provisioning/alerting/).
  BentoML KHÔNG gửi email trực tiếp.
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, List

import bentoml
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from prometheus_client import (Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,)
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent.resolve()

# ─── Compute Config ───────────────────────────────────────────────────────────────
CPU = os.getenv("CPU", "0.5")
MEMORY = os.getenv("MEMORY", "1Gi")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_ARTIFACTS_DIR     = str(BASE_DIR / "mlflow_artifacts")
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI",  "http://mlflow:5100")
MODEL_URI               = os.getenv("MLFLOW_MODEL_URI", "models:/Churn_Predict@champion") # Fallback load model từ MLflow server nếu không có local artifacts.  Cũng trả về trong /healthz để tham khảo.
LOW_CONF_THRESHOLD      = float(os.getenv("LOW_CONF_THRESHOLD", "0.60"))
LOW_CONF_STREAK_LIMIT   = int(os.getenv("LOW_CONF_STREAK_LIMIT", "200"))

# ─── Prometheus metrics ───────────────────────────────────────────────────────
REQUEST_COUNTER         = Counter("churn_prediction_requests_total", "Tổng số lượng request dự đoán đã nhận", ["endpoint"])
PREDICTION_COUNTER      = Counter("churn_prediction_labels_total", "Tổng số kết quả dự đoán phân loại theo nhãn", ["label"])
CONFIDENCE_HIST         = Histogram("churn_prediction_confidence", "Phân bổ điểm độ tự tin trên mỗi dòng dữ liệu", buckets=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
LATENCY_HIST            = Histogram("churn_prediction_latency_seconds", "Độ trễ xử lý dự đoán (tính theo giây)", ["endpoint"])
BATCH_SIZE_HIST         = Histogram("churn_prediction_batch_size", "Số lượng dòng dữ liệu trong mỗi request batch", buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500])
LOW_CONF_STREAK         = Gauge("churn_low_confidence_streak", "Chuỗi request liên tiếp có độ tự tin thấp hơn ngưỡng quy định")

# ─── Schemas ──────────────────────────────────────────────────────────────────
class DataframeSplit(BaseModel):
    columns         : List[str]
    data            : List[List[Any]]
    index           : List[Any] | None = None

class PredictRequest(BaseModel):
    dataframe_split : DataframeSplit

class RowPrediction(BaseModel):
    prediction      : int
    label           : str
    confidence      : float          # max(proba) — 0..1

class PredictionResult(BaseModel):
    predictions     : List[int]
    labels          : List[str]
    confidences     : List[float]
    row_count       : int
    low_conf_streak : int            # informational — authoritative value is in Prometheus

# ─── Streak tracker (thread-safe) ────────────────────────────────────────────
class StreakTracker:
    """
    Theo dõi một chuỗi các request gần nhất để giám sát độ tin cậy của mô hình.
    
    Cơ chế hoạt động:
    - 'Request độ tin cậy thấp' = Tất cả các dòng dữ liệu trong một request đều có 
      xác suất dự đoán lớn nhất (max-proba) < ngưỡng (threshold).
    
    - 'Streak' (Chuỗi): Đếm số lượng request liên tiếp đạt trạng thái 'độ tin cậy thấp'.
    
    - Nguyên tắc reset:
      Chuỗi này sẽ bị reset về 0 ngay lập tức nếu xuất hiện MỘT request bất kỳ 
      chứa ít nhất một dòng dữ liệu có độ tin cậy ≥ threshold.
    """
    def __init__(self, limit: int, threshold: float):
        self._limit     = limit
        self._threshold = threshold
        self._streak    = 0
        self._lock      = threading.Lock()

    def record(self, confidences: list[float]) -> int:
        """
        confidences: list of per-row max-proba values for this request.
        Returns current streak after recording.
        """
        is_low = all(c < self._threshold for c in confidences)
        with self._lock:
            if is_low:
                self._streak += 1
            else:
                self._streak = 0
            LOW_CONF_STREAK.set(self._streak)
            return self._streak

    @property
    def current(self) -> int:
        with self._lock:
            return self._streak


_streak_tracker = StreakTracker(LOW_CONF_STREAK_LIMIT, LOW_CONF_THRESHOLD)

# ─── Model loader ─────────────────────────────────────────────────────────────
def _load_model():
    """
    Try local artifacts first (offline / pre-exported), fall back to MLflow server.
    """
    local_model_dir = Path(MODEL_ARTIFACTS_DIR) 
    if local_model_dir.exists():
        print(f"[BentoML] Loading model from LOCAL artifacts: {local_model_dir}")
        model = mlflow.pyfunc.load_model(str(local_model_dir))
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"[BentoML] Loading model from MLflow: {MODEL_URI}")
        model = mlflow.pyfunc.load_model(MODEL_URI)

    # read meta.json if available
    meta_path = Path(MODEL_ARTIFACTS_DIR) / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"[BentoML] Model meta: {meta}")
    return model, meta


def _predict_df(model, df: pd.DataFrame):
    raw = model.predict(df) # -> Expected là một DataFrame có cột "churn_probability".

    # Kiểm tra lại xem raw là DataFrame hay cái gì khác, rồi xử lí dần
    if raw is None:
        raise ValueError("Model prediction returned None")
    elif isinstance(raw, pd.DataFrame):         confidences = raw["churn_probability"].tolist() # DataFrame có cột "churn_probability" thì lấy ra trải thành list()
    elif hasattr(raw, "ndim") and raw.ndim > 1: confidences = raw[:, 1].tolist()                # # Xử lý numpy array (ví dụ predict_proba trả về mảng 2D)
    else:                                       confidences = raw.tolist()                      # Fallback cho list hoặc 1D array
    
    confidences = [float(p) for p in confidences]
    preds       = [1 if p >= 0.5 else 0 for p in confidences]
    return preds, confidences

# ─── BentoML Service ──────────────────────────────────────────────────────────
@bentoml.service(
    name="churn_prediction_service",
    resources={"cpu": CPU, "memory": MEMORY},
    traffic={"timeout": 60},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
        }
    },
)
class ChurnPredictionService:

    def __init__(self):
        self.model, self.meta = _load_model()
        print("[BentoML] Model loaded successfully.")

    # ── /predict ──────────────────────────────────────────────────────────────
    @bentoml.api()
    def predict(self, input: PredictRequest) -> PredictionResult:
        t0    = time.perf_counter()
        split = input.dataframe_split
        df    = pd.DataFrame(split.data, columns=split.columns, index=split.index)

        preds, confidences = _predict_df(self.model, df)
        labels             = ["Rời bỏ" if p == 1 else "Không rời" for p in preds]
        streak             = _streak_tracker.record(confidences)

        # Prometheus: Lưu các biến vào RAM chờ Prometheus scrape, không có I/O nào khác trong code này.
        REQUEST_COUNTER.labels(endpoint="predict").inc()
        LATENCY_HIST.labels(endpoint="predict").observe(time.perf_counter() - t0)
        BATCH_SIZE_HIST.observe(len(preds))
        for lbl, conf in zip(labels, confidences):
            key = "churn" if lbl == "Rời bỏ" else "no_churn"
            PREDICTION_COUNTER.labels(label=key).inc()
            CONFIDENCE_HIST.observe(conf)
        
        return PredictionResult(
            predictions=preds,
            labels=labels,
            confidences=confidences,
            row_count=len(preds),
            low_conf_streak=streak,
        )

    # ── /predict_batch ────────────────────────────────────────────────────────
    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=512, max_latency_ms=500)
    def predict_batch(self, input: PredictRequest) -> PredictionResult:
        """
        BentoML adaptive-batching endpoint.
        Incoming requests are queued and dispatched together (up to 512 rows,
        max wait 500 ms) before a single model.predict() call.
        """
        t0                  = time.perf_counter()
        split               = input.dataframe_split
        df                  = pd.DataFrame(split.data, columns=split.columns, index=split.index)

        preds, confidences  = _predict_df(self.model, df)
        labels              = ["Rời bỏ" if p == 1 else "Không rời" for p in preds]
        streak              = _streak_tracker.record(confidences)

        REQUEST_COUNTER.labels(endpoint="predict_batch").inc()
        LATENCY_HIST.labels(endpoint="predict_batch").observe(time.perf_counter() - t0)
        BATCH_SIZE_HIST.observe(len(preds))
        for lbl, conf in zip(labels, confidences):
            key = "churn" if lbl == "Rời bỏ" else "no_churn"
            PREDICTION_COUNTER.labels(label=key).inc()
            CONFIDENCE_HIST.observe(conf)


        return PredictionResult(
            predictions=preds,
            labels=labels,
            confidences=confidences,
            row_count=len(preds),
            low_conf_streak=streak,
        )

    # ── /metrics  (Prometheus scrape endpoint) ────────────────────────────────
    @bentoml.api(route="/metrics", input_spec=None, output_spec=None)
    def metrics(self) -> bytes:
        from starlette.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ── /healthz ──────────────────────────────────────────────────────────────
    @bentoml.api()
    def healthz(self) -> dict:
        return {
            "status":          "ok",
            "model_uri":       MODEL_URI,
            "tracking_uri":    MLFLOW_TRACKING_URI,
            "model_meta":      self.meta,
            "low_conf_streak": _streak_tracker.current,
            "streak_limit":    LOW_CONF_STREAK_LIMIT,
            "conf_threshold":  LOW_CONF_THRESHOLD,
        }

"""
monitoring/evidently/monitor.py
────────────────────────────────
FastAPI micro-service chuyên về data/model monitoring với Evidently AI.

Luồng hoạt động
───────────────
  1.  BentoML service POST /log    → gửi prediction records tới đây
  2.  Scheduler (APScheduler) chạy mỗi 5 phút → tính Evidently report
  3.  Kết quả được expose qua Prometheus metrics → Grafana đọc & alert
  4.  Report HTML được lưu vào /reports/ (mount ra host để xem offline)

Endpoints
─────────
  POST /log              – nhận prediction log từ BentoML
  GET  /metrics          – Prometheus scrape endpoint
  GET  /report/latest    – redirect tới HTML report mới nhất
  GET  /health           – health check
"""

from __future__ import annotations

import io
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from evidently import ColumnMapping
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
    ClassificationConfusionMatrix,
)
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, Response
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
REFERENCE_DATA_PATH   = os.getenv("REFERENCE_DATA_PATH",  "/data/reference.parquet")
REPORTS_DIR           = Path(os.getenv("REPORTS_DIR",     "/reports"))
WINDOW_SIZE           = int(os.getenv("WINDOW_SIZE",      "500"))   # rows kept in memory
REPORT_INTERVAL_SEC   = int(os.getenv("REPORT_INTERVAL",  "300"))   # 5 min

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Prometheus ───────────────────────────────────────────────────────────────
LOGS_RECEIVED = Counter("evidently_logs_received_total",  "Prediction logs received")
DRIFT_SCORE   = Gauge("evidently_dataset_drift_score",   "Evidently dataset drift share")
DRIFT_DETECT  = Gauge("evidently_dataset_drift_detected","1 if drift detected, 0 otherwise")
MISSING_SHARE = Gauge("evidently_missing_values_share",  "Share of missing values in current window")
REPORT_ERRORS = Counter("evidently_report_errors_total", "Errors during report generation")
CONF_MEAN     = Gauge("evidently_confidence_mean",       "Mean confidence in current window")
CONF_P10      = Gauge("evidently_confidence_p10",        "10th percentile confidence")

# ─── In-memory prediction log ─────────────────────────────────────────────────
_log_lock    = Lock()
_log_window: deque[dict] = deque(maxlen=WINDOW_SIZE)

# ─── Reference data ───────────────────────────────────────────────────────────
_reference_df: Optional[pd.DataFrame] = None

def load_reference():
    global _reference_df
    ref_path = Path(REFERENCE_DATA_PATH)
    if ref_path.exists():
        _reference_df = pd.read_parquet(ref_path)
        print(f"[Evidently] Reference data loaded: {len(_reference_df)} rows")
    else:
        print(f"[Evidently] WARN: reference data not found at {ref_path}. Drift metrics will be skipped.")

# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictionLog(BaseModel):
    timestamp   : float
    features    : Dict[str, Any]
    prediction  : int
    confidence  : float
    label       : str
    endpoint    : str = "predict"   # predict | predict_batch

class BatchPredictionLog(BaseModel):
    records: List[PredictionLog]

# ─── Report generation ────────────────────────────────────────────────────────
def _build_current_df() -> pd.DataFrame:
    with _log_lock:
        rows = list(_log_window)
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        row = dict(r["features"])
        row["prediction"] = r["prediction"]
        row["confidence"]  = r["confidence"]
        records.append(row)
    return pd.DataFrame(records)


def run_evidently_report():
    """Called by scheduler every REPORT_INTERVAL_SEC seconds."""
    current_df = _build_current_df()
    if current_df.empty:
        print("[Evidently] No data in window, skipping report.")
        return

    # Update simple metrics always
    CONF_MEAN.set(float(current_df["confidence"].mean()))
    CONF_P10.set(float(np.percentile(current_df["confidence"], 10)))

    if _reference_df is None:
        print("[Evidently] No reference data, skipping drift report.")
        return

    try:
        col_mapping = ColumnMapping(
            target="prediction",
            prediction="prediction",
            numerical_features=[c for c in current_df.columns
                                 if c not in ("prediction", "confidence", "label")],
        )

        metrics = [
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ]
        # Add per-column drift for feature columns shared with reference
        shared_cols = [c for c in current_df.columns if c in _reference_df.columns
                       and c not in ("prediction", "confidence")]
        for col in shared_cols[:10]:   # cap at 10 to keep report light
            metrics.append(ColumnDriftMetric(column_name=col))

        report = Report(metrics=metrics)
        report.run(
            reference_data=_reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        # Extract top-level drift numbers
        result = report.as_dict()
        drift_result = result["metrics"][0]["result"]
        DRIFT_SCORE.set(drift_result.get("share_of_drifted_columns", 0))
        DRIFT_DETECT.set(1 if drift_result.get("dataset_drift", False) else 0)

        missing_result = result["metrics"][1]["result"]
        MISSING_SHARE.set(missing_result.get("current", {}).get("share_of_missing_values", 0))

        # Save HTML report
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = REPORTS_DIR / f"report_{ts}.html"
        report.save_html(str(out_path))

        # Keep symlink to latest
        latest = REPORTS_DIR / "latest.html"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_path.name)

        print(f"[Evidently] Report saved: {out_path}")

    except Exception as exc:
        REPORT_ERRORS.inc()
        print(f"[Evidently] ERROR generating report: {exc}")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Evidently Monitor", version="1.0.0")

@app.on_event("startup")
def startup():
    load_reference()
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_evidently_report, "interval", seconds=REPORT_INTERVAL_SEC)
    scheduler.start()
    print(f"[Evidently] Scheduler started (interval={REPORT_INTERVAL_SEC}s)")


@app.post("/log")
def log_prediction(record: PredictionLog):
    with _log_lock:
        _log_window.append(record.model_dump())
    LOGS_RECEIVED.inc()
    return {"ok": True, "window_size": len(_log_window)}


@app.post("/log/batch")
def log_batch(batch: BatchPredictionLog):
    with _log_lock:
        for r in batch.records:
            _log_window.append(r.model_dump())
    LOGS_RECEIVED.inc(len(batch.records))
    return {"ok": True, "logged": len(batch.records), "window_size": len(_log_window)}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/report/latest")
def latest_report():
    latest = REPORTS_DIR / "latest.html"
    if not latest.exists():
        return {"error": "No report generated yet. Wait for first scheduled run."}
    return FileResponse(str(latest), media_type="text/html")


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "window_size": len(_log_window),
        "has_reference": _reference_df is not None,
    }

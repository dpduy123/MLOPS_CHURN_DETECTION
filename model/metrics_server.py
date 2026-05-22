"""
metrics_server.py
-----------------
HTTP server expose Prometheus metrics sau mỗi lần train.
Chạy liên tục trong container metrics-server, lắng nghe trên port 8000.
train.py sẽ ghi metrics vào file JSON, server này đọc và expose.
"""

import json
import time
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

METRICS_FILE = Path("/app/save_model/metrics.json")
PORT = 8000


def read_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def build_prometheus_text(data: dict) -> str:
    lines = []

    # Pipeline-level metrics
    pipeline = data.get("pipeline", {})

    lines.append("# HELP ml_pipeline_status 1=success, 0=failed, 2=running")
    lines.append("# TYPE ml_pipeline_status gauge")
    lines.append(f"ml_pipeline_status {pipeline.get('status', 0)}")

    lines.append("# HELP ml_pipeline_runs_total Total pipeline runs")
    lines.append("# TYPE ml_pipeline_runs_total counter")
    lines.append(f"ml_pipeline_runs_total {pipeline.get('runs_total', 0)}")

    lines.append("# HELP ml_last_training_timestamp_seconds Unix timestamp of last training")
    lines.append("# TYPE ml_last_training_timestamp_seconds gauge")
    lines.append(f"ml_last_training_timestamp_seconds {pipeline.get('last_run_ts', 0)}")

    lines.append("# HELP ml_training_samples_total Number of training samples")
    lines.append("# TYPE ml_training_samples_total gauge")
    lines.append(f"ml_training_samples_total {pipeline.get('train_samples', 0)}")

    lines.append("# HELP ml_test_samples_total Number of test samples")
    lines.append("# TYPE ml_test_samples_total gauge")
    lines.append(f"ml_test_samples_total {pipeline.get('test_samples', 0)}")

    lines.append("# HELP ml_feature_count Number of features used")
    lines.append("# TYPE ml_feature_count gauge")
    lines.append(f"ml_feature_count {pipeline.get('feature_count', 0)}")

    # Per-model metrics
    models = data.get("models", {})

    lines.append("# HELP ml_model_accuracy Model accuracy on test set")
    lines.append("# TYPE ml_model_accuracy gauge")
    for model_name, m in models.items():
        lines.append(f'ml_model_accuracy{{model_name="{model_name}"}} {m.get("accuracy", 0)}')

    lines.append("# HELP ml_model_f1_score Model F1-score on test set")
    lines.append("# TYPE ml_model_f1_score gauge")
    for model_name, m in models.items():
        lines.append(f'ml_model_f1_score{{model_name="{model_name}"}} {m.get("f1_score", 0)}')

    lines.append("# HELP ml_model_training_duration_seconds Time taken to train model")
    lines.append("# TYPE ml_model_training_duration_seconds gauge")
    for model_name, m in models.items():
        lines.append(f'ml_model_training_duration_seconds{{model_name="{model_name}"}} {m.get("duration_seconds", 0)}')

    lines.append("# HELP ml_model_precision Model precision on test set")
    lines.append("# TYPE ml_model_precision gauge")
    for model_name, m in models.items():
        lines.append(f'ml_model_precision{{model_name="{model_name}"}} {m.get("precision", 0)}')

    lines.append("# HELP ml_model_recall Model recall on test set")
    lines.append("# TYPE ml_model_recall gauge")
    for model_name, m in models.items():
        lines.append(f'ml_model_recall{{model_name="{model_name}"}} {m.get("recall", 0)}')

    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            data = read_metrics()
            body = build_prometheus_text(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Bỏ log thừa
        pass


if __name__ == "__main__":
    print(f"📊 Metrics server running on :{PORT}/metrics")
    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    server.serve_forever()

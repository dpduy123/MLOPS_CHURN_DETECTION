import time
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from arch import LogisticModel, PolyModel, RandomForestModel, KNNModel, LGBMModel
from dataset import ChurnDataset

DATA_ROOT = Path("Data")
SAVE_ROOT = Path("save_model")
METRICS_FILE = SAVE_ROOT / "metrics.json"


def load_periods(start: int, end: int) -> pd.DataFrame:
    frames = []
    for i in range(start, end + 1):
        frames.append(pd.read_csv(DATA_ROOT / f"period_{i}.csv"))
    return pd.concat(frames, ignore_index=True)


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)

    print(f"  acc: {acc:.4f}")
    print(f"  F1: {f1:.4f}")
    print("  classification_report:")
    print(classification_report(y_test, pred, digits=4))
    print("  confusion_matrix:")
    print(confusion_matrix(y_test, pred))

    return {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}


def save_metrics(pipeline_info: dict, model_metrics: dict):
    """Ghi metrics ra file JSON để metrics_server đọc và expose."""
    payload = {
        "pipeline": pipeline_info,
        "models": model_metrics,
    }
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n📊 Metrics đã lưu vào {METRICS_FILE}")


def load_run_count() -> int:
    if METRICS_FILE.exists():
        try:
            with open(METRICS_FILE) as f:
                data = json.load(f)
            return data.get("pipeline", {}).get("runs_total", 0)
        except Exception:
            pass
    return 0


def main():
    pipeline_start = time.time()

    df = load_periods(1, 10)
    dataset = ChurnDataset()
    X, y = dataset.get_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Train label distribution:", pd.Series(y_train).value_counts().to_dict())
    print("Test label distribution:", pd.Series(y_test).value_counts().to_dict())

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    models = [
        ("logistic", LogisticModel()),
        ("poly", PolyModel()),
        ("random_forest", RandomForestModel()),
        ("knn", KNNModel()),
        ("lgbm", LGBMModel()),
    ]

    model_metrics = {}

    for name, model in models:
        print("=" * 60)
        print(name)
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0

        metrics = evaluate(model, X_test, y_test)
        metrics["duration_seconds"] = round(duration, 3)
        model_metrics[name] = metrics

        model.save(str(SAVE_ROOT / f"{name}.pkl"))

    runs_total = load_run_count() + 1
    pipeline_info = {
        "status": 1,
        "runs_total": runs_total,
        "last_run_ts": time.time(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X_train.shape[1] if hasattr(X_train, "shape") else 0,
        "total_duration_seconds": round(time.time() - pipeline_start, 3),
    }

    save_metrics(pipeline_info, model_metrics)


if __name__ == "__main__":
    main()

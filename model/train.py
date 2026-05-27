import os
os.environ["OMP_NUM_THREADS"] = "1"  # Giới hạn OpenMP chỉ sử dụng tối đa 1 luồng để tránh treo/crash máy

import argparse
from pathlib import Path
import pandas as pd
import optuna
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from arch import LogisticModel, PolyModel, RandomForestModel, KNNModel, LGBMModel
from dataset import ChurnDataset

DATA_ROOT = Path("Data")
SAVE_ROOT = Path("save_model")

# Tắt hiển thị log của Optuna để màn hình console sạch sẽ hơn
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_periods(start: int, end: int) -> pd.DataFrame:
    frames = []
    for i in range(start, end + 1):
        frames.append(pd.read_csv(DATA_ROOT / f"period_{i}.csv"))
    return pd.concat(frames, ignore_index=True)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    return {"accuracy": acc, "f1_score": f1}


# Hàm định nghĩa không gian tìm kiếm tham số cho Optuna ứng với từng mô hình
def get_model_and_params(model_name, trial):
    if model_name == "logistic":
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"])
        }
        return LogisticModel(**params), params

    elif model_name == "poly":
        # Giới hạn bậc đa thức tối đa là 2 để tránh làm chậm hệ thống
        degree = trial.suggest_int("degree", 2, 2)
        params = {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True)
        }
        return PolyModel(degree=degree, **params), {"degree": degree, **params}

    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
        return RandomForestModel(**params), params

    elif model_name == "knn":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15)
        }
        return KNNModel(**params), params

    elif model_name == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        }
        return LGBMModel(**params), params

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_and_optimize(data_version: str, model_name: str, model_class, X_train, X_test, y_train, y_test, n_trials: int):
    parent_run_name = f"{model_name}_{data_version}"
    
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        # Ghi nhận tags để sau này lọc tìm kiếm trên giao diện MLflow dễ dàng
        mlflow.set_tags({
            "data_version": data_version,
            "model_type": model_name
        })

        def objective(trial):
            model, params = get_model_and_params(model_name, trial)
            
            # Mỗi lượt chạy thử nghiệm (trial) của Optuna sẽ là một Run con (Nested Run)
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                
                model.fit(X_train, y_train)
                metrics = evaluate(model, X_test, y_test)
                
                mlflow.log_metrics(metrics)
                return metrics["f1_score"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log kết quả tốt nhất tìm được lên Parent Run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_f1_score", study.best_value)

        best_model, _ = get_model_and_params(model_name, study.best_trial)
        best_model.fit(X_train, y_train)
        
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        model_path = SAVE_ROOT / f"best_{model_name}_{data_version}.pkl"
        best_model.save(str(model_path))
        
        mlflow.log_artifact(str(model_path))
        print(f"-> [Xong] {parent_run_name} | Best F1: {study.best_value:.4f}")


def main():
    # Nhận tham số phiên bản dữ liệu từ dòng lệnh terminal
    parser = argparse.ArgumentParser(description="Churn Prediction Training with DVC and MLflow")
    parser.add_argument(
        "--version", 
        type=str, 
        default="v1", 
        help="Tên phiên bản dữ liệu đang checkout từ DvC (ví dụ: v1, v2, v3)"
    )
    args = parser.parse_args()
    data_version = args.version

    print(f"Đang huấn luyện với phiên bản dữ liệu DVC: {data_version}")

    mlflow.set_experiment("Churn_Prediction_Multi_Version")

    df = load_periods(1, 10)
    dataset = ChurnDataset()
    X, y = dataset.get_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models_to_run = [
        ("logistic", LogisticModel, 10),
        
        ("poly", PolyModel, 3),              
        
        ("random_forest", RandomForestModel, 10),
        ("knn", KNNModel, 5),   
        
        ("lgbm", LGBMModel, 20),
    ]

    for name, model_class, n_trials in models_to_run:
        print(f"\n>>> Đang tối ưu hóa mô hình: {name} với {n_trials} trials...")
        train_and_optimize(
            data_version=data_version, 
            model_name=name, 
            model_class=model_class, 
            X_train=X_train, 
            X_test=X_test, 
            y_train=y_train, 
            y_test=y_test,
            n_trials=n_trials
        )


if __name__ == "__main__":
    main()
import os
import argparse, mlflow, optuna, pandas as pd, shutil, tempfile
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.models import WRAPPER_MAP, LogisticModel, PolyModel, RandomForestModel, KNNModel, LGBMModel
from src.train.dataset import ChurnDataset
from sklearn.metrics import accuracy_score, f1_score
from mlflow.models.signature import infer_signature

os.environ["OMP_NUM_THREADS"] = "1"


PROJECT_ROOT        = Path(__file__).resolve().parent.parent.parent
ML_DATA_ROOT        = PROJECT_ROOT / "data" / "processed"
OUTPUTS_ROOT        = PROJECT_ROOT / "outputs" 
DATA_SCHEMA_PATH    = PROJECT_ROOT / "src" / "config" / "schema.yaml"



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
    
def evaluate(model, X_test, y_test):
    pred    = model.predict(X_test)
    acc     = accuracy_score(y_test, pred)
    f1      = f1_score(y_test, pred, zero_division=0)
    return {"accuracy": acc, "f1_score": f1}

def train_and_optimize(
    model_name, 
    data_version, 
    raw_data,
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    n_trials
):
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_run_name = f"{model_name}_{data_version}_{timestamp}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        mlflow.set_tags({"data_version": data_version, "model_type": model_name})

        def objective(trial):
            model, params = get_model_and_params(model_name, trial)

            with mlflow.start_run(run_name=f"{model_name}_trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                
                model.fit(X_train, y_train)
                metrics = evaluate(model, X_test, y_test)
                
                mlflow.log_metrics(metrics)
                return metrics["f1_score"]
            

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_f1_score", study.best_value)

        best_model, _   = get_model_and_params(model_name, study.best_trial)
        best_model.fit(X_train, y_train)

        OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
        model_path      = OUTPUTS_ROOT / "model_weights" / f"best_{model_name}_{data_version}.pkl"
        best_model.save(str(model_path))

        model_wrapper   = WRAPPER_MAP[model_name]()
        sample_input    = raw_data.drop(columns=["Churn", "CustomerID"]).head(1)    # raw, chưa encode
        sample_output   = best_model.predict(X_test.head(1))                        # predict vẫn dùng processed
        signature       = infer_signature(sample_input, sample_output)

        print("sample_input columns:", sample_input.columns.tolist()) 
        print(sample_input)

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                print(f"--- Đang kiểm tra (validate) model tại {tmp_dir} ---")
                
                # Lưu thử vào thư mục tạm
                mlflow.pyfunc.save_model(
                    path=tmp_dir,
                    python_model=model_wrapper,
                    artifacts={
                        "model_path": str(model_path),
                        "preprocess_path": str(PROJECT_ROOT / "outputs" / "transform_weights" / f"full_pipeline_1_10.pkl"),
                        "data_schema": str(DATA_SCHEMA_PATH)
                    }
                )   
            
                loaded_model = mlflow.pyfunc.load_model(tmp_dir)

                sample_input = raw_data.drop(columns=["Churn", "CustomerID"]).head(1)
                sample_prediction = loaded_model.predict(sample_input)
                print("✅ Prediction test thành công! Kết quả: {}".format(sample_prediction))
                print("✅ Mlflow load_model validation thành công ! Bắt đầu log lên MLFlow.")

                artifacts = {
                    "model_path":      str(model_path).replace("\\", "/"),
                    "preprocess_path": str(PROJECT_ROOT / "outputs" / "transform_weights" / f"full_pipeline_1_10.pkl").replace("\\", "/"),
                    "data_schema":     str(DATA_SCHEMA_PATH).replace("\\", "/")
                }

                mlflow.pyfunc.log_model(
                    artifact_path       = "model",
                    code_paths          = [str(PROJECT_ROOT / "src")],            
                    python_model        = model_wrapper,
                    input_example       = sample_input,
                    signature           = signature,
                    artifacts           = artifacts,
                    pip_requirements    = ["scikit-learn", "pandas", "pyyaml", "lightgbm"]
                )

                print(f"-> Đã log model {model_name} thành công.")

            except Exception as e:
                print(f"❌ CẢNH BÁO: Model validation thất bại, không log lên MLFlow!")
                print(f"Chi tiết lỗi: {e}")
                raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", type=str, default="v2.0")
    args = parser.parse_args()


    mlflow.set_tracking_uri("http://localhost:5100")
    experiment_name = f"Churn_Prediction_data{args.data_version}"
    mlflow.set_experiment(experiment_name)
    
    # Sử dụng ChurnDataset đã refactor với schema
    df = pd.read_csv(ML_DATA_ROOT / "period_1_10_processed.csv") # Đảm bảo file này tồn tại sau bước xử lý
    dataset = ChurnDataset()
    X, y = dataset.get_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    raw_data = pd.read_csv(ML_DATA_ROOT / "period_1_10_raw.csv") 
    raw_data = raw_data.head(1) 

    print(raw_data.columns.tolist())
    print(raw_data.head(1))

    models_to_run = [
        ("knn", KNNModel, 2),      
        #("logistic", LogisticModel, 2),
        #("poly", PolyModel, 2),        
        #("random_forest", RandomForestModel, 2),
        #("lgbm", LGBMModel, 2),
    ]


    for name, cls, trials in models_to_run:
        train_and_optimize(
            model_name=name, 
            data_version=args.data_version, 
            raw_data=raw_data,
            X_train=X_train, 
            X_test=X_test, 
            y_train=y_train, 
            y_test=y_test, 
            n_trials=trials
        )

if __name__ == "__main__":
    main()
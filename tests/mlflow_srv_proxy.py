import os
from dotenv import load_dotenv
from pathlib import Path
import mlflow

prj_dir = Path(__file__).resolve().parent.parent
mlflow_srv_code_dir = prj_dir / "mlflow_srv"
tracking_uri = "http://localhost:5100"
experiment_name = "test-v0"

print(f"Project directory: {prj_dir.as_posix()}.") 
print(f"MLflow Server code directory: {mlflow_srv_code_dir.as_posix()}.")
print(f"MLflow Tracking URI: {tracking_uri}.")
print(f"Experiment Name: {experiment_name}.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# 1. Tìm tên run duy nhất
i = 0
while True:
    run_name = f"test_run_{i}"    
    if mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.mlflow.runName = '{run_name}'").empty: 
        break # Tên này chưa có, sử dụng nó
    i += 1
print(f"Đã tìm được tên run duy nhất: {run_name}.")


# 2. Start run với tên đã tìm được
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_param("test_key", "test_value")
    
    with open("test_artifact.txt", "w") as f: f.write("Hello MinIO! I am an artifact.")
    mlflow.log_artifact("test_artifact.txt")
    
    print(f"Log thành công! Run name: {run_name}")
    print(f"Run ID: {run.info.run_id}")

# Dọn dẹp
if os.path.exists("test_artifact.txt"): os.remove("test_artifact.txt")

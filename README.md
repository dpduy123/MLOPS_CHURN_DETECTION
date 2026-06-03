# MLOps Churn Detection

Dự án học máy dự đoán khách hàng rời bỏ dịch vụ (Customer Churn Detection) được thiết kế theo quy trình MLOps.
Dự án sử dụng **DVC** để quản lý phiên bản dữ liệu và **Docker** để đóng gói môi trường thực thi một cách đồng nhất.

## 🛠 Yêu cầu hệ thống (Prerequisites)
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Git](https://git-scm.com/)
- [DVC](https://dvc.org/) (Sẽ được cài sẵn nếu bạn dùng \`pip install -r requirements.txt\`)

## 🗺 Project structure

```text
MLOPS_CHURN_DETECTION/
├── .dvc/
├── 📂 data/
|   ├── org/
|   ├── raw/
|   ├── processed/
|   ├── raw.dvc
|   ├── processed.dvc
├── src/
│   ├── config/
│   ├── data_process/
│   ├── models/
│   ├── train/
│   ├── utils/
├── deployments/
    ├── docker-compose.yml
    ├── bentoml/
    │   ├── service.py
    │   └── bentofile.yaml
    │   └── requirements.txt
    ├── monitoring/
        ├── grafana/          # Chứa dashboard (JSON files)
        ├── prometheus/       # Chứa prometheus.yml
        └── loki/             # Chứa loki-config.yaml


```
---

## 🚀 Hướng dẫn chạy (Local)

### 1. Lấy mã nguồn (Code)
```bash
git clone https://github.com/dpduy123/MLOPS_CHURN_DETECTION.git
cd MLOPS_CHURN_DETECTION
```

### 2. Lấy dữ liệu (Data)

- **Cách 1**: Dữ liệu lớn của dự án không nằm trên Github mà được quản lý tách biệt bằng DVC. Chạy lệnh sau để kéo dữ liệu đã qua xử lý chuẩn bị về thư mục `data/`:

    ```bash
    dvc pull
    ```

    Hoặc

    ```bash
    # Xem tất cả các lần commit có gắn tag
    git tag --list --format="%(objectname:short) %(creatordate:short) %(refname:strip=2) %(contents:subject)" 
    
    # Lấy tag của commit chứa phiên bản data b muốn pull về (ví dụ v2.0-data)
    dvc get . data/raw          --rev {commit-tag (ví dụ v2.0-data)} --out data/raw
    dvc get . data/processed    --rev {commit-tag (ví dụ v2.0-data)} --out data/processed
    ```

- **Cách 2**: Nếu bạn không có mật khẩu SSH, đầu tiên hãy thực hiện tải dữ liệu về theo hướng dẫn tại [đây](/data/org/how_to_download_data.txt). Sau đó thực hiện lần lượt các lệnh sau:

    ```python
    python data/org/split_data.py
    ```

### 3. Khởi tạo MLFlow Server Local

```bash
bash scripts/restart_mlflow_srv_dev.sh  # Sau khi chạy lệnh này, bạn có thể truy cập MLFlow tại http://localhost:5100
```

### 4. Khởi tạo môi trường venv, chuẩn bị dữ liệu và train mô hình
```python
py -3.10 -m venv .venv          # Hãy đảm bảo đã tải Python 3.10 về máy
python -m src.train.dataset     # Tạo file .pkl của pipeline data preprocessing và dataset đã preprocessing phục vụ train
python -m src.train.train --data-version v2.0
```

### 5. Gán Alias cho MVP model

```python
# Truy cập http://localhost:5100, vào Model training (bên dưới phải logo MLFlow, bên cạnh chữ GenAI)
# Vào Experiments -> một Parent Run bất kì -> Models -> Register Model -> Tạo hoặc chọn Model "Churn_Predict" -> Register 
# Truy cập Model Registry -> Chọn "Churn_Predict" -> Đăng kí alias @champion cho model tốt nhất
# Sau đó chạy lệnh sau

python tests/mlflow_loadmodel.py
```

### 6. Tạo BentoML API

```bash
Remove-Item -Recurse -Force $HOME\bentoml\bentos    # Xóa hết các BentoML Folder trước đó cho đỡ nặng máy

python deployments/scripts/export_model.py `
    --tracking-uri "http://localhost:5100" `
    --model-name "Churn_Predict" `
    --alias "champion" `
    --version "1.0.0" `
    --out-dir "deployments/bentoml/mlflow_artifacts"

Copy-Item -Path "requirements.txt" -Destination "deployments/bentoml/requirements.txt" -Force
(Get-Content "requirements.txt") | Where-Object { $_ -notmatch "pywin32" -and $_ -notmatch "pypiwin32" } | Set-Content "deployments/bentoml/requirements.txt" -Force

bentoml delete churn-prediction:1.0.0               # Xóa bento:1.0.0 trước đó nếu có
bentoml build deployments/bentoml --version 1.0.0
bentoml serve churn-prediction:1.0.0

# Mở 1 terminal khác. Lúc này Bento ML đang được serve ở localhost:3000, hãy test request với file tests/send_req_bentoml.py.
python tests/send_req_bentoml.py
```


### 7. Đóng gói BentoML thành Docker
```bash
bentoml containerize churn-prediction:1.0.0 --verbose --opt progress=plain 
    # --opt progress=plain: Cho phép in log ra màn hình
    # --verbose: In mọi log ra màn hình từ lúc build tới lúc kết thúc
docker run --rm -p 3000:3000 churn-prediction:1.0.0
    # Nếu gặp lỗi không load được file vì path chứa \\, hãy chạy lệnh sau:
        #Get-ChildItem -Recurse -Filter "MLmodel" | ForEach-Object {
        #    (Get-Content $_.FullName) -replace '\\', '/' | Set-Content $_.FullName
        #}

```
---
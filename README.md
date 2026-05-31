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
├── .dvc
├── 📂 data/
|    ├── raw/
|    ├── processed/
|    ├── raw.dvc
|    ├── processed.dvc
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
python -m src.train.train
```

### 5. Gán Alias cho MVP model

```python
# Truy cập http://localhost:5100, vào Model training (bên dưới phải logo MLFlow, bên cạnh chữ GenAI)
# Vào Experiments -> một Parent Run bất kì -> Models -> Register Model -> Tạo hoặc chọn Model "Churn_Predict" -> Register 
# Truy cập Model Registry -> Chọn "Churn_Predict" -> Đăng kí alias @champion cho model tốt nhất
# Sau đó chạy lệnh sau

python tests/mlflow_loadmodel.py
```

---
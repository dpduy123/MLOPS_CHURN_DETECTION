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

## 🚀 Cài đặt (Setup)

**1. Lấy mã nguồn (Code)**
Cloning dự án từ Github:
```bash
git clone https://github.com/dpduy123/MLOPS_CHURN_DETECTION.git
cd MLOPS_CHURN_DETECTION
```

**2. Lấy dữ liệu (Data)**
Dữ liệu lớn của dự án không nằm trên Github mà được quản lý tách biệt bằng DVC. Chạy lệnh sau để kéo dữ liệu đã qua xử lý chuẩn bị về thư mục \`Data/\`:
```bash
dvc pull
```

**3. Khởi tạo môi trường (Environment)**
Tạo Docker image chứa Python và các thư viện cần thiết:
```bash
docker compose build
```
*(💡 Ghi chú: Nếu hệ thống của bạn xài bản cũ, hãy thêm dấu gạch ngang thành `docker-compose build`)*

---

## 🏃 Hướng dẫn chạy (How to run)

### Cách 1: Chạy Tự động (Khuyên dùng)
Dự án có sẵn một script tự động kéo data, build môi trường và chạy huấn luyện. Bạn chỉ cần cấp quyền và thực thi:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Cách 2: Chạy Thủ công
Vì dữ liệu lấy từ DVC đã là phiên bản hoàn thiện nhất (đã qua làm sạch và Feature Engineering), bạn chỉ cần chạy bước Huấn luyện mô hình (Training):
```bash
docker compose run --rm mlops-pipeline python model/train.py
```

---

## 📦 Thu hoạch Mô hình (Model Artifacts)
Nhờ cơ chế Volumes của Docker, sau khi quá trình huấn luyện hoàn tất, các tệp tin mô hình (`.pkl` như Logistic, RandomForest, LightGBM...) sẽ được lưu trực tiếp vào thư mục \`save_model/\` trên máy của bạn.

# MLOps Churn Detection

Dự án học máy dự đoán khách hàng rời bỏ dịch vụ (Customer Churn Detection) được thiết kế theo quy trình MLOps.
Dự án sử dụng **DVC** để quản lý phiên bản dữ liệu và **Docker** để đóng gói môi trường thực thi một cách đồng nhất.

## 🛠 Yêu cầu hệ thống (Prerequisites)
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Git](https://git-scm.com/)
- [DVC](https://dvc.org/) (Sẽ được cài sẵn nếu bạn dùng \`pip install -r requirements.txt\`)

---

## 🚀 Cài đặt (Setup)

**1. Lấy mã nguồn (Code)**
Cloning dự án từ Github:
```bash
git clone https://github.com/dpduy123/MLOPS_CHURN_DETECTION.git
cd MLOPS_CHURN_DETECTION
```

---

## 🏃 Hướng dẫn chạy (How to run)

### 1. Kéo data
```bash
dvc pull
```

### 2. Khởi động monitoring stack
```bash
docker compose up -d prometheus grafana metrics-server
```

### 3. Chạy training
```bash
docker compose run --rm mlops-pipeline python model/train.py
```

### 4. Xem dashboard

Mở browser:

```text
http://localhost:3000
```

Đăng nhập:

```text
admin / admin123
```

---

## 📦 Thu hoạch Mô hình (Model Artifacts)
Nhờ cơ chế Volumes của Docker, sau khi quá trình huấn luyện hoàn tất, các tệp tin mô hình (`.pkl` như Logistic, RandomForest, LightGBM...) sẽ được lưu trực tiếp vào thư mục \`save_model/\` trên máy của bạn.

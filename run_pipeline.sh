#!/bin/bash

# Dừng toàn bộ script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi
set -e

echo "🚀 Bắt đầu quy trình MLOps Churn Detection..."

echo "------------------------------------------------------"
echo "📥 1. Kéo dữ liệu mới nhất từ DVC..."
dvc pull

echo "------------------------------------------------------"
echo "🐳 2. Đảm bảo Docker Image đã được build..."
docker compose build

echo "------------------------------------------------------"
echo "🧹 3. (Bỏ qua) Dữ liệu từ DVC đã được làm sạch sẵn!"
# docker compose run --rm mlops-pipeline python Process_data/clean_data.py

echo "------------------------------------------------------"
echo "⚙️ 4. (Bỏ qua) Dữ liệu từ DVC đã có sẵn các đặc trưng Feature Engineering!"
# docker compose run --rm mlops-pipeline python Process_data/Feture_Engineering.py

echo "------------------------------------------------------"
echo "🧠 5. Đang chạy huấn luyện mô hình (Model Training)..."
docker compose run --rm mlops-pipeline python model/train.py

echo "------------------------------------------------------"
echo "✅ Hoàn tất toàn bộ quy trình! Các file mô hình (.pkl) đã được lưu vào thư mục save_model/"
ls -lh save_model/

#!/bin/bash
set -e

echo "🚀 Bắt đầu quy trình MLOps Churn Detection..."

echo "------------------------------------------------------"
echo "📥 1. Kéo dữ liệu mới nhất từ DVC..."
dvc pull

echo "------------------------------------------------------"
echo "🐳 2. Build Docker Image..."
docker compose build

echo "------------------------------------------------------"
echo "📡 3. Khởi động monitoring stack (Prometheus + Grafana + Metrics Server)..."
docker compose up -d prometheus grafana metrics-server
echo "   ✅ Prometheus : http://localhost:9090"
echo "   ✅ Grafana     : http://localhost:3000  (admin / admin123)"
echo "   ✅ Metrics     : http://localhost:8000/metrics"

echo "------------------------------------------------------"
echo "⏳ Chờ metrics-server sẵn sàng..."
until docker compose exec -T metrics-server python -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; do
  sleep 2
done
echo "   ✅ Metrics server đã sẵn sàng!"

echo "------------------------------------------------------"
echo "🧠 4. Chạy huấn luyện mô hình (Model Training)..."
docker compose run --rm mlops-pipeline python model/train.py

echo "------------------------------------------------------"
echo "✅ Hoàn tất! Các file mô hình (.pkl) đã lưu vào save_model/"
ls -lh save_model/

echo ""
echo "📊 Xem dashboard tại: http://localhost:3000"
echo "   Tài khoản: admin / admin123"
echo "   Dashboard: MLOps → MLOps Churn Detection - Model Metrics"

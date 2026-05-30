#!/bin/bash

# Kiểm tra file .env.dev
if [ ! -f .env.dev ]; then
    echo "Lỗi: Không tìm thấy file .env.dev"
    exit 1
fi

echo "🚀 Đang khởi động hạ tầng MLflow..."

# Down các container cũ nếu có
docker compose --env-file .env.dev -f docker-compose.dev.yml down -v

# Chạy Docker Compose
docker compose --env-file .env.dev -f docker-compose.dev.yml up -d --build

echo "------------------------------------------------"
echo "✅ MLflow Server: http://localhost:5100"
echo "✅ MinIO Console: http://localhost:5000"
echo "------------------------------------------------"
#!/bin/bash

set -e # Dừng script nếu có lệnh nào bị lỗi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MLFLOW_SRV_DIR="$PROJECT_ROOT/mlflow_srv"

echo "🚀 Đang chuyển hướng tới thư mục: $MLFLOW_SRV_DIR"
cd "$MLFLOW_SRV_DIR"

if [ ! -f ".env.dev" ]; then
    echo "Lỗi: Không tìm thấy file .env.dev tại $(pwd)"
    exit 1
fi

echo "🚀 Đang xóa hạ tầng MLflow..."

docker compose --env-file .env.dev -f docker-compose.dev.yml down -v

echo "Xóa thành công! Bạn có thể khởi động lại hạ tầng bằng cách chạy scripts/restart_mlflow_srv_dev.sh"
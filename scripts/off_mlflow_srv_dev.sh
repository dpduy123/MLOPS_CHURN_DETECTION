#!/bin/bash

# Lấy đường dẫn tuyệt đối của thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR

# Trỏ tới file start_dev.sh trong mlflow_srv
START_SCRIPT="$SCRIPT_DIR/../mlflow_srv/start_dev.sh"
echo "Đường dẫn tới script khởi động: $START_SCRIPT"

# Kiểm tra xem file có tồn tại không trước khi chạy
if [ -f "$START_SCRIPT" ]; then
    echo "🚀 Đang chuyển hướng tới script khởi động..."     # Chuyển thư mục làm việc về mlflow_srv để start_dev.sh chạy đúng context
    cd "$(dirname "$START_SCRIPT")"
    docker compose --env-file .env.dev -f docker-compose.dev.yml down -v
else
    echo "❌ Lỗi: Không tìm thấy file start_dev.sh tại $START_SCRIPT"
    exit 1
fi
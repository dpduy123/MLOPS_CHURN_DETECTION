#!/bin/bash

# Lấy đường dẫn tuyệt đối của thư mục chứa script này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR

# Trỏ tới file start_dev.sh trong mlflow_srv
START_SCRIPT="$SCRIPT_DIR/../mlflow_srv/start_dev.sh"
echo "Đường dẫn tới script khởi động: $START_SCRIPT"

if [ -f "$START_SCRIPT" ]; then             # Kiểm tra xem file có tồn tại không trước khi chạy
    echo "🚀 Đang chuyển hướng tới script khởi động..."
    cd "$(dirname "$START_SCRIPT")"         # Chuyển thư mục làm việc về mlflow_srv để start_dev.sh chạy đúng context
    
    bash start_dev.sh                       # Chạy script start_dev.sh
else
    echo "❌ Lỗi: Không tìm thấy file start_dev.sh tại $START_SCRIPT"
    exit 1
fi
# Sử dụng Python 3.10 slim để môi trường nhẹ và ổn định
FROM python:3.10-slim

# Cài đặt thư viện hệ thống cần thiết (nếu thư viện C++ yêu cầu, như LightGBM đôi khi cần libgomp1)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc mặc định trong container
WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn vào container
COPY ./Process_data ./Process_data
COPY ./model ./model

# Thư mục chứa model và data sẽ được mount qua volume, 
# nhưng ta vẫn có thể tạo sẵn thư mục trống để tránh lỗi path
RUN mkdir -p /app/save_model /app/Data

# Lệnh mặc định, có thể ghi đè khi chạy docker-compose run
CMD ["python", "model/train.py"]

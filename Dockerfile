# Sử dụng Python làm base image
FROM python:3.9-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /.

# Sao chép tệp dự án vào container
COPY . /.

# Cài đặt các thư viện phụ thuộc
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

# Chỉ định lệnh chạy khi container khởi động
CMD ["python", "main.py"]

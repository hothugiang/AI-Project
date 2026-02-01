FROM python:3.10-slim

WORKDIR /app

# Cài các package hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code + model
COPY . .

# Streamlit dùng port 10000 (Render yêu cầu)
EXPOSE 10000

# Lệnh chạy app
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]

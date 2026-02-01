FROM python:3.9-slim

WORKDIR /app

# Cài thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code + model
COPY . .

# Expose port cho HF
EXPOSE 7860

# Chạy Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

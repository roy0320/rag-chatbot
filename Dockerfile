FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 建立 chroma_db 目錄
RUN mkdir -p /app/chroma_db

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
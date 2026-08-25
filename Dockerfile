# Python 3.11 tabanlı hafif işletim sistemi imajı
FROM python:3.11-slim

# Çalışma dizinini belirle
WORKDIR /app

# Gerekli dosyaları kopyala ve kütüphaneleri kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını konteynere aktar
COPY . .

# API'yi arka planda başlat ve Streamlit Arayüzünü Render'ın dış dünyasına aç
#CMD uvicorn api_server:app --host 127.0.0.1 --port 8050 & streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0
CMD uvicorn api_server:app --host 0.0.0.0 --port 8050 & streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
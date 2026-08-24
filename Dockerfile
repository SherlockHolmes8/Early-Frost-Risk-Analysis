FROM ubuntu:latest
LABEL authors="imshe"

ENTRYPOINT ["top", "-b"]
# Python 3.10 tabanlı hafif işletim sistemi imajı
FROM python:3.11-slim

# Çalışma dizinini belirle
WORKDIR /app

# Gerekli dosyaları kopyala ve kütüphaneleri kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını konteynere aktar
COPY . .
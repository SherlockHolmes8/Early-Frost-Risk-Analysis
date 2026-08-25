# 🌾 Zirai Don Erken Uyarı ve Risk Yönetim Sistemi (AI-Powered)

Meteorolojik verileri analiz ederek tarımsal don riskini **24 saat önceden** tahmin eden, uçtan uca (end-to-end) tasarlanmış makine öğrenmesi ve derin öğrenme tabanlı erken uyarı sistemi. 

Bu proje, açık kaynaklı hava durumu verilerinden başlayıp, model çıkarımlarına, RESTful API sunumuna, etkileşimli kullanıcı arayüzüne ve tam otonom CI/CD süreçlerine kadar eksiksiz bir MLOps mimarisi barındırır.

---

## 🚀 Öne Çıkan Özellikler

- **Gelişmiş Yapay Zeka Modelleri:** LSTM (Deep Learning) ve XGBoost/Random Forest (Ensemble) modelleriyle yüksek performans (XGBoost: 0.975 AUC, LSTM: 0.969 AUC).
- **Asimetrik Risk Yönetimi & XAI:** Tarımsal koruma önceliklerine göre karar eşiği (Threshold) %35'e çekilerek False Negative oranı minimize edilmiştir. Karar mekanizmaları XAI yaklaşımlarıyla görselleştirilmiştir.
- **RESTful API (FastAPI):** SlowAPI ile Rate-Limiting (Maks: 5 istek/dakika) ve API-Key kimlik doğrulaması ile korunan asenkron backend.
- **Etkileşimli Dashboard (Streamlit):** Çiftçilerin ve yöneticilerin teknik bilgi gerektirmeden kullanabileceği, Plotly entegreli ve oturum (login) korumalı görsel arayüz.
- **Otonom Bildirimler:** Risk eşiği aşıldığında kurumsal HTML formatında SMTP e-posta uyarıları.
- **Güvenlik & DevOps:** Hassas veri sanitizasyonu, Docker konteyner mimarisi, GitHub Actions (.github/workflows) ile CI/CD otonom test boru hattı.

---

## 🛠️ Teknoloji Yığını (Tech Stack)

* **Veri Bilimi & ML:** `Pandas`, `Numpy`, `Scikit-Learn`, `XGBoost`, `TensorFlow/Keras` (LSTM)
* **Backend:** `FastAPI`, `Uvicorn`, `SQLAlchemy`, `SQLite`
* **Frontend:** `Streamlit`, `Plotly`
* **DevOps & Güvenlik:** `Docker`, `GitHub Actions`, `SlowAPI`

---

## 📂 Proje Mimarisi (Dosya Yapısı)

```text
├── .github/workflows/ci.yml       # GitHub Actions CI/CD boru hattı
├── api_server.py                  # FastAPI RESTful API motoru
├── dashboard.py                   # Streamlit kullanıcı arayüzü ve görselleştirme
├── core_system.py                 # OOP standartlarında sistem çekirdeği (FrostWarningSystem)
├── data_fetcher.py                # Open-Meteo API'den veri çekme otomasyonu
├── data_preprocessing.py          # Veri temizleme, ölçeklendirme ve time-shifting
├── database.py                    # SQLite veritabanı ORM yapılandırması
├── model_lstm.py / model_xgb.py   # Model mimarileri ve eğitim betikleri
├── alert_system.py                # SMTP e-posta uyarı modülü
├── mlops_pipline.py               # Uçtan uca MLOps otomasyon betiği
├── Dockerfile                     # Konteynerizasyon ayarları
├── requirements.txt               # Bağımlılık (Dependency) listesi
└── *.png / *.pkl / *.db           # Model ağırlıkları, veritabanı ve XAI çıktıları
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import subprocess
from sklearn.metrics import accuracy_score
import time


def run_drift_analysis_and_retrain():
    print("\n" + "=" * 50)
    print("🔍 [MLOps] Veri Kayması (Data Drift) Analizi Başlatılıyor...")
    print("=" * 50)

    # 1. Mevcut Modeli ve Ölçeklendiriciyi Yükle
    try:
        model = load_model('frost_warning_lstm.h5')
        scaler = joblib.load('scaler.pkl')
        print("✅ [MLOps] Mevcut model (frost_warning_lstm.h5) belleğe alındı.")
    except Exception as e:
        print(f"⚠️ [MLOps] Model bulunamadı! Hata: {e}")
        print("⚙️ [MLOps] Sıfırdan eğitim başlatılıyor...")
        subprocess.run(["python", "model_lstm.py"])
        return

    # 2. Son Verileri Çekme (Simülasyon: Veri setinin en güncel son 500 saatini alıyoruz)
    print("📥 [MLOps] Son döneme ait meteorolojik gerçekleşmeler veritabanından çekiliyor...")
    df = pd.read_csv("islenmis_veri_model_icin.csv")
    df['target_24h_ahead'] = df['frost_event'].shift(-24)
    df.dropna(inplace=True)

    # Zaman serisinin son 500 satırını "yeni test verisi" olarak ayırıyoruz
    recent_data = df.tail(500)
    X_raw = recent_data.drop(["time", "target_24h_ahead"], axis=1).values
    y_raw = recent_data["target_24h_ahead"].values

    # 3. Veri Ön İşleme (Scaler Transform)
    X_scaled = scaler.transform(X_raw)

    def create_sequences(X, y, time_steps=6):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y_raw, time_steps=6)

    # 4. Sınav Vakti (Modelin Güncel Performansını Ölçme)
    print("🧠 [MLOps] Model son veriler üzerinde sınava sokuluyor...")
    y_pred_prob = model.predict(X_seq, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    current_accuracy = accuracy_score(y_seq, y_pred)
    print(f"📊 [MLOps] Modelin Güncel Verideki Başarı Skoru: %{current_accuracy * 100:.2f}")

    # 5. Eşik Kontrolü ve Otonom Retraining (Yeniden Eğitim)
    THRESHOLD = 0.85  # %85 Başarı Barajı

    if current_accuracy < THRESHOLD:
        print(f"🚨 [MLOps] KRİTİK: Model başarı oranı %{THRESHOLD * 100} barajının altına düştü!")
        print("⚙️ [MLOps] Otonom Yeniden Eğitim (Automated Retraining) protokolü başlatılıyor...")
        time.sleep(2)  # Sistemin nefes alması için kısa bir bekleme

        # İşletim sistemi seviyesinde model_lstm.py'yi çalıştır
        subprocess.run(["python", "model_lstm.py"])
        print("\n✅ [MLOps] Yeniden eğitim tamamlandı! Güncel ağırlıklar sisteme başarıyla entegre edildi.")
    else:
        print(f"✅ [MLOps] Başarı skoru barajın (%{THRESHOLD * 100}) üzerinde. Model performansı stabil.")
        print("✅ [MLOps] Yeniden eğitime gerek yok, analiz sonlandırıldı.")


if __name__ == "__main__":
    run_drift_analysis_and_retrain()
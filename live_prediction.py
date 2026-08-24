import numpy as np
from tensorflow.keras.models import load_model
import joblib

print("Erken Uyarı Sistemi Başlatılıyor...")

# 1. Kaydedilen Modeli ve Ölçeklendiriciyi Yükle
model = load_model('frost_warning_lstm.h5')
scaler = joblib.load('scaler.pkl')

# 2. API'den Gelen Son 6 Saatin Canlı Verisi (Simülasyon)
# Sırasıyla: Sıcaklık, Çiy Noktası, Bağıl Nem, Rüzgar Hızı
# Örnek: Sıcaklık 5 dereceden 2 dereceye doğru düşme trendinde
# Sırasıyla: Sıcaklık, Bağıl Nem, Çiy Noktası, Rüzgar Hızı, Şu Anki Don Durumu
live_data = np.array([
    [5.1, 75, 1.2, 12, 0],
    [4.8, 76, 1.0, 11, 0],
    [4.2, 78, 0.8, 10, 0],
    [3.5, 80, 0.5, 8, 0],
    [2.8, 82, 0.1, 6, 1], # Sıcaklık <= 3 olduğu için don olayı 1 oldu
    [2.1, 85, -0.2, 5, 1] # Sıcaklık <= 3 olduğu için don olayı 1 oldu
])

# 3. Veriyi Modele Hazırla (Ölçekleme ve Boyutlandırma)
scaled_data = scaler.transform(live_data)
model_input = np.expand_dims(scaled_data, axis=0)

# 4. Tahmin Yap (Inference)
prediction_prob = model.predict(model_input)[0][0]

# 5. Sonucu Kullanıcıya İlet
if prediction_prob > 0.5:
    print(f"\n[KRİTİK UYARI] %{prediction_prob * 100:.1f} ihtimalle 24 saat sonra DON RİSKİ bekleniyor!")
    print("Lütfen seralarda gerekli önlemleri alınız.")
else:
    print(f"\n[BİLGİ] Önümüzdeki 24 saat için don riski bulunmamaktadır. (Risk: %{prediction_prob * 100:.1f})")
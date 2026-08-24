import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
print("Derin dbrename (LSTM) Modeli Hazırlanıyor...\n")

# 1. İşlenmiş veriyi yükle ve Zaman Kaydırmayı yap
df = pd.read_csv("islenmis_veri_model_icin.csv")
df['target_24h_ahead'] = df['frost_event'].shift(-24)
df.dropna(inplace=True)

# Girdileri (X) ve Hedefi (y) belirle
X_raw = df.drop(["time", "target_24h_ahead"], axis=1).values
y_raw = df["target_24h_ahead"].values

# 2. VERİ ÖLÇEKLEME (SCALING) - Sinir ağları 0-1 arası veriyi sever
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. ZAMAN PENCERESİ OLUŞTURMA (LOOKBACK WINDOW)
# Modele "Son 6 saatin (time_steps) değişim trendine bak" diyoruz.
def create_sequences(X, y, time_steps=6):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 6
X_seq, y_seq = create_sequences(X_scaled, y_raw, time_steps)

# 4. Eğitim ve Test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 5. LSTM SİNİR AĞI MİMARİSİ
model = Sequential([
    LSTM(64, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2), # Modelin ezberlemesini (overfitting) önlemek için nöronların %20'sini rastgele kapatıyoruz
    Dense(1, activation='sigmoid') # Çıktı katmanı (0 ile 1 arası olasılık verecek)
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim sürecini 'history' değişkenine atıyoruz
print("Eğitim Başlıyor! (Epochs)...\n")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Eğitim sonrası Loss Grafiğini Çizdir
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı (Train Loss)', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı (Validation Loss)', color='orange', linewidth=2)
plt.title('LSTM Model Öğrenme Eğrisi (Loss Değerleri)', fontsize=14)
plt.xlabel('İterasyon (Epoch)', fontsize=12)
plt.ylabel('Kayıp (Loss)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('lstm_loss_grafiki.png', dpi=300)
print("\nLoss grafiği 'lstm_loss_grafiki.png' olarak kaydedildi!")
plt.show()

# 7. TAHMİN VE DEĞERLENDİRME
y_pred_prob = model.predict(X_test)
# Çıkan olasılıkları 0.5 eşiğine göre 0 veya 1'e yuvarlıyoruz
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nLSTM Doğruluğu (Accuracy): % {accuracy * 100:.2f}")
print("\n--- LSTM 24 Saat Sonrası İçin Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))

# Modeli ve Ölçeklendiriciyi Kaydet
model.save('frost_warning_lstm.h5')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel ve Scaler başarıyla diske kaydedildi!")
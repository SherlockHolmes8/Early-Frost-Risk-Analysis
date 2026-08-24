import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Erken Uyarı (24 Saat Sonrası) Model Eğitimi Başlıyor...\n")

# 1. İşlenmiş veriyi yükle
df = pd.read_csv("islenmis_veri_model_icin.csv")

# 2. ZAMAN KAYDIRMA (TIME SHIFTING) - 24 Saat Sonrasını Hedefleme
# Yeni bir sütun oluşturup, frost_event'i 24 satır yukarı çekiyoruz.
df['target_24h_ahead'] = df['frost_event'].shift(-24)

# Son 24 saatin hedefi boş (NaN) kalacağı için o satırları siliyoruz
df.dropna(inplace=True)

# 3. Özellikler (X) ve Hedef (y) Değişkenini Ayır
# time: Tarih olduğu için siliyoruz.
# target_24h_ahead: Hedefimiz (y) olduğu için X'ten siliyoruz.
# frost_event (şu anki don durumu) ve temperature_2m (şu anki sıcaklık) artık kopya olmadığı için X'te KALIYOR!
X = df.drop(["time", "target_24h_ahead"], axis=1)
y = df["target_24h_ahead"]

# 4. Veriyi Eğitim (%80) ve Test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeli Oluştur ve Eğit
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Test verisi üzerinde tahmin yap
y_pred = rf_model.predict(X_test)

# 7. Sonuçları Değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Erken Uyarı Model Doğruluğu (Accuracy): % {accuracy * 100:.2f}")

print("\n--- 24 Saat Sonrası İçin Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))
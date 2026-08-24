import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("Model eğitimi başlatılıyor (Random Forest)...")

# 1. İşlenmiş veriyi yükle
df = pd.read_csv("islenmis_veri_model_icin.csv")

# 2. Özellikler (X) ve Hedef (y) Değişkenini Ayır
# Kopya çekmesini engellemek için temperature_2m'yi de siliyoruz!
X = df.drop(["time", "frost_event", "temperature_2m"], axis=1)
y = df["frost_event"]

# 3. Veriyi Eğitim (%80) ve Test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Eğitim verisi: {len(X_train)} satır")
print(f"Test verisi (Modelin görmeyeceği veri): {len(X_test)} satır\n")

# 4. Modeli Oluştur ve Eğit (100 karar ağacı kullanıyoruz)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Test verisi üzerinde tahmin yap
y_pred = rf_model.predict(X_test)

# 6. Sonuçları Değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu (Accuracy): % {accuracy * 100:.2f}")

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))
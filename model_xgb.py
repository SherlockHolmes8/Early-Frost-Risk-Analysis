import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Erken Uyarı Modeli Eğitiliyor (XGBoost)...\n")

# 1. İşlenmiş veriyi yükle ve Zaman Kaydırmayı (Time Shifting) yap
df = pd.read_csv("islenmis_veri_model_icin.csv")
df['target_24h_ahead'] = df['frost_event'].shift(-24)
df.dropna(inplace=True)

# 2. Özellikler (X) ve Hedef (y) Değişkenini Ayır
X = df.drop(["time", "target_24h_ahead"], axis=1)
y = df["target_24h_ahead"]

# 3. Veriyi Eğitim (%80) ve Test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. XGBoost Modelini Oluştur ve Eğit
# n_estimators: Ağaç sayısı, learning_rate: Öğrenme hızı (0.1 ideal bir başlangıçtır)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# 5. Tahmin ve Sonuçları Değerlendir
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"XGBoost Doğruluğu (Accuracy): % {accuracy * 100:.2f}")
print("\n--- XGBoost 24 Saat Sonrası İçin Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))


# 6. ÖZELLİK ÖNEMİ (FEATURE IMPORTANCE) GÖRSELLEŞTİRMESİ
# Modelin değişkenlere verdiği önem skorlarını alıyoruz
feature_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)

# Skorları büyükten küçüğe sıralıyoruz
feature_importances = feature_importances.sort_values(ascending=False)

# Grafik ayarları
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, hue=feature_importances.index, palette="viridis", legend=False)

# Etiketler ve Başlık
plt.xlabel('Özellik Önemi (Feature Importance) Skoru', fontsize=12)
plt.ylabel('Meteorolojik Değişkenler', fontsize=12)
plt.title('XGBoost Erken Uyarı Sistemi: Hangi Veriler Daha Önemli?', fontsize=14)

# Grafiği ekranda göster ve bilgisayara kaydet
plt.tight_layout()
plt.savefig('xgboost_ozellik_onemi.png', dpi=300)
print("\nGrafik 'xgboost_ozellik_onemi.png' olarak kaydedildi!")
plt.show()
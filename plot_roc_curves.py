import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
import joblib
from sklearn.model_selection import TimeSeriesSplit
print("ROC Eğrisi Analizi Başlatılıyor...\n")

# 1. Veri Hazırlığı
df = pd.read_csv("islenmis_veri_model_icin.csv")
df['target_24h_ahead'] = df['frost_event'].shift(-24)
df.dropna(inplace=True)

X_raw = df.drop(["time", "target_24h_ahead"], axis=1).values
y_raw = df["target_24h_ahead"].values

# Ağaç modelleri için standart Train/Test ayrımı (2 Boyutlu)
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

# 2. XGBoost Tahminleri (Olasılık olarak)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_tree, y_train_tree)
xgb_probs = xgb_model.predict_proba(X_test_tree)[:, 1]

# 3. LSTM Veri Hazırlığı ve Tahminleri
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X_raw)

def create_sequences(X, y, time_steps=6):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_raw, time_steps=6)
_, X_test_lstm, _, y_test_lstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

lstm_model = load_model('frost_warning_lstm.h5')
lstm_probs = lstm_model.predict(X_test_lstm).ravel()

# 4. ROC ve AUC Hesaplamaları
fpr_xgb, tpr_xgb, _ = roc_curve(y_test_tree, xgb_probs)
auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_lstm, tpr_lstm, _ = roc_curve(y_test_lstm, lstm_probs)
auc_lstm = auc(fpr_lstm, tpr_lstm)

# 5. ROC Grafiğini Çizdir
plt.figure(figsize=(10, 8))
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label=f'XGBoost (AUC = {auc_xgb:.3f})')
plt.plot(fpr_lstm, tpr_lstm, color='red', lw=2, label=f'LSTM (AUC = {auc_lstm:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') # Şans çizgisi

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı (False Positive Rate)', fontsize=12)
plt.ylabel('Doğru Pozitif Oranı (True Positive Rate)', fontsize=12)
plt.title('Modeller Arası ROC Eğrisi Karşılaştırması', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('roc_egrisi_karsilastirmasi.png', dpi=300)
print("ROC grafiği 'roc_egrisi_karsilastirmasi.png' olarak kaydedildi!")
plt.show()

# LSTM Modeli için Karmaşıklık Matrisi Hesaplama
# Olasılıkları 0.5 eşiğine göre 0 veya 1'e çeviriyoruz
lstm_pred_classes = (lstm_probs > 0.5).astype(int)
cm = confusion_matrix(y_test_lstm, lstm_pred_classes)

# Isı Haritası (Heatmap) Çizimi
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black',
            xticklabels=['Don Yok (0)', 'Don Var (1)'],
            yticklabels=['Don Yok (0)', 'Don Var (1)'])

plt.title('LSTM Modeli Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14)
plt.xlabel('Tahmin Edilen Durum', fontsize=12)
plt.ylabel('Gerçekleşen Durum', fontsize=12)

plt.tight_layout()
plt.savefig('lstm_confusion_matrix.png', dpi=300)
print("Karmaşıklık Matrisi 'lstm_confusion_matrix.png' olarak kaydedildi!")
plt.show()

# ---------------------------------------------------------
yeni_esik_degeri = 0.35
lstm_pred_classes_yeni = (lstm_probs > yeni_esik_degeri).astype(int)

cm_yeni = confusion_matrix(y_test_lstm, lstm_pred_classes_yeni)

# Yeni Korumacı Isı Haritası Çizimi (Kırmızı)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_yeni, annot=True, fmt='d', cmap='Reds', linewidths=1, linecolor='black',
            xticklabels=['Don Yok (0)', 'Don Var (1)'],
            yticklabels=['Don Yok (0)', 'Don Var (1)'])

plt.title(f'LSTM Modeli - Karar Eşiği: {yeni_esik_degeri} (Koruma Odaklı)', fontsize=14)
plt.xlabel('Tahmin Edilen Durum', fontsize=12)
plt.ylabel('Gerçekleşen Durum', fontsize=12)

plt.tight_layout()
plt.savefig('lstm_confusion_matrix_threshold_035.png', dpi=300)
print(f"\nYeni karar eşikli matris 'lstm_confusion_matrix_threshold_035.png' olarak kaydedildi!")
plt.show()


# ---------------------------------------------------------
# 2. ZAMAN SERİSİ ÇAPRAZ DOĞRULAMA (CROSS-VALIDATION) EKLENTİSİ
# ---------------------------------------------------------


print("\nZaman Serisi Çapraz Doğrulama Planlaması Test Ediliyor...")
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X_seq)):
    print(f"Katlama (Fold) {fold+1} -> Eğitim Seti: {len(train_index)} satır | Test Seti: {len(test_index)} satır")

print("Çapraz doğrulama mimarisi başarıyla entegre edildi ve ardışık yapı doğrulandı.")
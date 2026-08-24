import numpy as np
from core_system import FrostWarningSystem

# Sistemi tek satırda ayağa kaldır
app = FrostWarningSystem()

# Örnek anlık hava verisi
# Sıcaklık, Bağıl Nem, Çiy Noktası, Rüzgar Hızı, Don Olayı
ornek_veri = np.array([
    [5.1, 75, 1.2, 12, 0],
    [4.8, 76, 1.0, 11, 0],
    [4.2, 78, 0.8, 10, 0],
    [3.5, 80, 0.5, 8, 0],
    [2.8, 82, 0.1, 6, 1],
    [2.1, 85, -0.2, 5, 1]
])

# Tahmini çalıştır
olasilik, sonuc = app.predict_risk(ornek_veri)
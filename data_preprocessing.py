import pandas as pd
import numpy as np

print("Veri ön işleme ve etiketleme başlatılıyor...")

df = pd.read_csv("tarimsal_hava_verisi.csv")
df["time"] = pd.to_datetime(df["time"])

if "soil_temperature_0cm" in df.columns:
    df.drop("soil_temperature_0cm", axis=1, inplace=True)
    print("Boş dönen toprak sıcaklığı sütunu kaldırıldı.")

df.ffill(inplace=True)

# DON OLAYI ETİKETLEME - ERKEN UYARI EŞİĞİ GÜNCELLEMESİ
# Erken uyarı için hava sıcaklığı 3°C ve altına düştüğünde don riski (1) kabul ediliyor.
df["frost_event"] = np.where(df["temperature_2m"] <= 3, 1, 0)

print("\n--- İşlenmiş Verinin İlk 5 Satırı ---")
print(df.head())

print(f"\nToplam Veri Sayısı: {len(df)}")
print(f"Toplam Tespit Edilen Don Riski (Saat): {df['frost_event'].sum()}")

df.to_csv("islenmis_veri_model_icin.csv", index=False)
print("\nVeri, model eğitimi için 'islenmis_veri_model_icin.csv' olarak kaydedildi.")
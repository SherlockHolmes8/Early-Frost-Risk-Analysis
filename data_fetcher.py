import requests
import pandas as pd


def fetch_weather_data(lat, lon, start_date, end_date):
    print("Konya verileri Open-Meteo'dan çekiliyor...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "soil_temperature_0cm",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)
        df["time"] = pd.to_datetime(df["time"])

        print(f"Başarılı! Toplam {len(df)} satır veri çekildi.")
        return df
    else:
        print(f"Hata oluştu: {response.status_code}")
        return None


# Konya için son 5 yılın verisi
df = fetch_weather_data(37.8746, 32.4931, "2019-01-01", "2023-12-31")

if df is not None:
    df.to_csv("tarimsal_hava_verisi.csv", index=False)
    print("Yeni Konya verisi 'tarimsal_hava_verisi.csv' olarak kaydedildi.")
from fastapi.testclient import TestClient
from api_server import app

# API'mizi test etmek için sanal bir istemci oluşturuyoruz
client = TestClient(app)


def test_api_ayakta_mi():
    """Ana dizine istek atarak sunucunun uyanık olup olmadığını test eder."""
    response = client.get("/")
    assert response.status_code == 200
    assert "mesaj" in response.json()


def test_guvenlik_kalkani():
    """API Key (Şifre) olmadan sisteme sızmaya çalışıp 403 Forbidden yediğimizi test eder."""
    payload = {
        "temperature": -2, "humidity": 80, "dew_point": -3,
        "wind_speed": 10, "current_frost_status": 1
    }
    # Şifresiz istek atıyoruz (headers yok)
    response = client.post("/predict", json=payload)

    # 403 Yasaklı (Forbidden) kodu dönmeli, dönerse testimiz BAŞARILI demektir.
    assert response.status_code == 403
from fastapi import FastAPI, Depends, Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn
import numpy as np
from sqlalchemy.orm import Session
import asyncio
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Kendi dosyalarımız
from core_system import FrostWarningSystem
import database
from database import PredictionRecord, engine
from alert_system import send_frost_alert
import mlops_pipeline

# ================= GÜVENLİK VE RATE LIMITING =================
# Dakikada maksimum 5 isteğe izin veriyoruz (Sistemi yormamak için)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Zirai Don Erken Uyarı API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = "ZIRAI_DON_GIZLI_ANAHTAR_2026"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yetkisiz Erişim: Geçersiz veya eksik API Anahtarı!"
        )
    return api_key
# =============================================================

database.Base.metadata.create_all(bind=engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ARKA PLAN GÖREVLEYİCİSİ (SCHEDULER)
async def auto_system_check():
    while True:
        print("SİSTEM BİLGİSİ: Arka plan servisleri devrede. Rutin denetimler yapılıyor...")
        await asyncio.sleep(3600)
        try:
             mlops_pipeline.run_drift_analysis_and_retrain()
        except Exception as e:
             print(f"SİSTEM HATASI: MLOps modülü çalıştırılamadı: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_system_check())

@app.get("/")
def read_root():
    return {"mesaj": "Zirai Don Erken Uyarı API'sine Hoş Geldiniz!"}

print("Yapay Zeka Motoru Başlatılıyor...")
ai_system = FrostWarningSystem()

class WeatherData(BaseModel):
    temperature: float
    humidity: float
    dew_point: float
    wind_speed: float
    current_frost_status: int

# ANALİZ UÇ NOKTASI (Güvenlik Katmanlarıyla Korunuyor)
@app.post("/predict")
@limiter.limit("5/minute")
def predict_frost_risk(request: Request, data: WeatherData, db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    live_data = np.array([[
        data.temperature, data.humidity, data.dew_point,
        data.wind_speed, data.current_frost_status
    ]])

    probability, risk_status = ai_system.predict_risk(live_data)

    risk_prob_rounded = round(float(probability), 3)
    is_frost = bool(risk_status)

    db_record = PredictionRecord(
        temperature=data.temperature, humidity=data.humidity,
        dew_point=data.dew_point, wind_speed=data.wind_speed,
        current_frost_status=data.current_frost_status,
        risk_probability=risk_prob_rounded, is_frost_expected=is_frost
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    if is_frost:
        send_frost_alert(data.temperature, risk_prob_rounded)

    return {
        "record_id": db_record.id,
        "risk_probability": risk_prob_rounded,
        "is_frost_expected": is_frost,
        "warning_message": "KRİTİK UYARI: Don riski tespit edildi!" if is_frost else "Sistem stabil, don riski bulunmuyor."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8050)
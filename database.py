from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# SQLite veritabanı dosyasının oluşturulacağı yol
SQLALCHEMY_DATABASE_URL = "sqlite:///./frost_predictions.db"

# Veritabanı motorunu (engine) başlat
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Veritabanındaki tablomuzun şeması
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    temperature = Column(Float)
    humidity = Column(Float)
    dew_point = Column(Float)
    wind_speed = Column(Float)
    current_frost_status = Column(Integer)

    # Yapay zekanın ürettiği çıktılar
    risk_probability = Column(Float)
    is_frost_expected = Column(Boolean)
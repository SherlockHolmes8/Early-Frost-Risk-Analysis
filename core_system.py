import numpy as np
from tensorflow.keras.models import load_model
import joblib
from logger_config import logger


class FrostWarningSystem:
    def __init__(self, model_path='frost_warning_lstm.h5', scaler_path='scaler.pkl'):
        logger.info("Sistem başlatılıyor, yapay zeka modelleri diske yükleniyor.")
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("LSTM modeli ve MinMaxScaler başarıyla belleğe alındı.")
        except Exception as e:
            logger.error(f"Model yükleme aşamasında kritik hata: {e}")
            raise

    def predict_risk(self, live_data, threshold=0.35):
        logger.info("Anlık hava durumu verisi işleniyor.")
        try:
            # Gelen veriyi ölçeklendir ve tensör yapısına dönüştür
            scaled_data = self.scaler.transform(live_data)
            model_input = np.expand_dims(scaled_data, axis=0)

            # Modeli çalıştır ve sonucu al
            prediction_prob = self.model.predict(model_input, verbose=0)[0][0]

            # Eşik değerine göre kararı ver
            risk_status = 1 if prediction_prob > threshold else 0

            if risk_status == 1:
                logger.warning(f"KRİTİK UYARI: Don riski tespit edildi! İhtimal: %{prediction_prob * 100:.1f}")
            else:
                logger.info(f"Sistem stabil, don riski bulunmuyor. İhtimal: %{prediction_prob * 100:.1f}")

            return prediction_prob, risk_status

        except Exception as e:
            logger.error(f"Tahminleme algoritmasında boyut veya veri hatası: {e}")
            return None, None
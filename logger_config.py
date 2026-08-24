import logging


def setup_logger():
    # Ana loglayıcıyı oluştur
    logger = logging.getLogger("FrostWarningLogger")
    logger.setLevel(logging.DEBUG)

    # Log mesajlarının formatını belirle
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Dosyaya yazdırma işlemi için ayarlar
    file_handler = logging.FileHandler('system_logs.log', encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Ekrana yazdırma işlemi için ayarlar
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Ayarları sisteme entegre et
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Diğer dosyalardan doğrudan çağrılabilmesi için değişkeni dışa aktar
logger = setup_logger()
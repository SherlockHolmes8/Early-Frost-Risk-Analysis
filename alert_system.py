import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime


def send_frost_alert(temperature, risk_probability):
    SENDER_EMAIL = "senin_mailin@gmail.com"
    SENDER_PASSWORD = "gmail_uygulama_sifren"
    RECEIVER_EMAIL = "hedef_kisi@gmail.com"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🚨 ZİRAİ DON UYARISI - Risk: %{risk_probability * 100:.1f}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    # Kurumsal HTML Şablonu
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333;">
        <div style="border: 2px solid #d9534f; padding: 20px; border-radius: 10px; max-width: 600px; margin: auto;">
            <h2 style="color: #d9534f; text-align: center;">⚠️ KRİTİK DON RİSKİ TESPİT EDİLDİ</h2>
            <hr>
            <p><strong>Tarih/Saat:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Sistem, sahadan alınan anlık meteorolojik verileri analiz ederek yüksek don riski tespit etmiştir.</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background-color: #f9f9f9;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Sıcaklık:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: #d9534f;"><strong>{temperature} °C</strong></td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Risk İhtimali:</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd; color: #d9534f;"><strong>%{risk_probability * 100:.1f}</strong></td>
                </tr>
            </table>
            <p style="margin-top: 20px; text-align: center; font-size: 14px; color: #777;">
                Lütfen ilgili lokasyonlardaki don önleyici sistemleri aktif hale getiriniz.
            </p>
        </div>
      </body>
    </html>
    """

    msg.attach(MIMEText(html_content, 'html'))

    try:
        # server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.starttls()
        # server.login(SENDER_EMAIL, SENDER_PASSWORD)
        # server.send_message(msg)
        # server.quit()
        print(f"BİLGİ: Kurumsal HTML E-posta modülü tetiklendi. (Hedef: {RECEIVER_EMAIL})")
    except Exception as e:
        print(f"Hata: E-posta gönderilemedi. Detay: {e}")
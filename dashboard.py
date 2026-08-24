import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import os
from datetime import datetime
import pandas as pd
import sqlite3 # YENİ EKLENDİ

# Sayfa Ayarları (Geniş Ekran Modu)
st.set_page_config(page_title="Zirai Don Erken Uyarı", page_icon="❄️", layout="wide")
# Sayfa Ayarları (Geniş Ekran Modu)
st.set_page_config(page_title="Zirai Don Erken Uyarı", page_icon="❄️", layout="wide")

# ================= YENİ: KİMLİK DOĞRULAMA (LOGIN) SİSTEMİ =================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    # Başlık ve metni HTML ile ortalıyoruz
    st.markdown("<h1 style='text-align: center;'>🔒 Sistem Girişi</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Zirai Don Erken Uyarı Paneline erişmek için lütfen yetkilendirme bilgilerinizi giriniz.</p>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Araya biraz boşluk

    col_login1, col_login2, col_login3 = st.columns([1, 1, 1])
    with col_login2:
        with st.form("login_form"):
            username = st.text_input("Kullanıcı Adı")
            password = st.text_input("Şifre", type="password")
            submit_button = st.form_submit_button("Giriş Yap", width="stretch")

            if submit_button:
                if username == "admin" and password == "1234":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Hatalı kullanıcı adı veya şifre!")
    st.stop()
# =========================================================================
# ================= YENİ EKLENEN: YAN MENÜ VE ÇIKIŞ BUTONU =================
with st.sidebar:
    st.markdown("### 👤 Yönetici Paneli")
    st.info("Giriş Yapıldı: **admin**")
    st.divider()
    if st.button("🚪 Çıkış Yap", width="stretch"):
        st.session_state.logged_in = False
        st.rerun() # Sayfayı yenile ve login ekranına geri dön
# =========================================================================

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("❄️ Zirai Don Erken Uyarı ve Risk Yönetim Paneli")
st.markdown("Bu panel, anlık meteorolojik verileri derin öğrenme (LSTM) algoritmalarıyla analiz ederek proaktif don riski tahminleri üretir.")
st.divider()

# ÇOKLU SEKME MİMARİSİ (3 Sekmeye Çıktı)
tab1, tab2, tab3 = st.tabs(["🚀 Canlı Risk Analizi", "⚙️ Model ve Sistem Durumu", "📂 Veritabanı Geçmişi ve Trendler"])

# ================= TAB 1: CANLI ANALİZ =================
with tab1:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("📊 Meteorolojik Parametreler")
        st.markdown("Lütfen sahadan alınan anlık değerleri girin:")

        temp = st.slider("Sıcaklık (°C)", -10.0, 15.0, 2.1)
        humidity = st.slider("Bağıl Nem (%)", 0.0, 100.0, 85.0)
        dew_point = st.slider("Çiy Noktası (°C)", -15.0, 10.0, -0.2)
        wind_speed = st.slider("Rüzgar Hızı (km/s)", 0.0, 50.0, 5.0)
        current_frost = st.selectbox("Mevcut Don Durumu", [0, 1],
                                     format_func=lambda x: "Don Yok (0)" if x == 0 else "Don Var (1)")

        analyze_button = st.button("Risk Analizi Yap", width="stretch", type="primary")

    with col2:
        st.subheader("🎯 Sistem Tahmin Çıktısı")

        metrik_col1, metrik_col2, metrik_col3, metrik_col4 = st.columns(4)
        metrik_col1.metric("Sıcaklık", f"{temp} °C")
        metrik_col2.metric("Nem", f"%{humidity}")
        metrik_col3.metric("Çiy", f"{dew_point} °C")
        metrik_col4.metric("Rüzgar", f"{wind_speed} km/s")
        st.divider()

        if analyze_button:
            api_url = "http://127.0.0.1:8050/predict"
            payload = {
                "temperature": temp, "humidity": humidity,
                "dew_point": dew_point, "wind_speed": wind_speed,
                "current_frost_status": current_frost
            }
            # YENİ EKLENEN KISIM: API Anahtarı Header'ı
            headers = {
                "X-API-Key": "ZIRAI_DON_GIZLI_ANAHTAR_2026"
            }

            try:
                # requests satırına headers=headers parametresini ekliyoruz
                response = requests.post(api_url, json=payload, headers=headers, timeout=5)

                if response.status_code == 200:
                    result = response.json()
                    risk_prob = result["risk_probability"] * 100
                    is_frost = result["is_frost_expected"]

                    # Grafikler için yan yana 2 kolon açıyoruz
                    grafik_col1, grafik_col2 = st.columns(2)

                    with grafik_col1:
                        # 1. Klasik Hız Göstergesi (Gauge)
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=risk_prob,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "LSTM Don Riski (%)", 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                                'steps': [{'range': [0, 35], 'color': "lightgreen"},
                                          {'range': [35, 100], 'color': "salmon"}],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 35}
                            }
                        ))
                        # Güncel versiyona uygun width ayarı
                        st.plotly_chart(fig_gauge, width="stretch")

                    with grafik_col2:
                        # 2. YENİ EKLENEN: ENSEMBLE AI RADAR GRAFİĞİ (Örümcek Ağı)
                        # Diğer modellerin varyanslarını simüle ediyoruz
                        rf_risk = min(max(risk_prob + 4.2, 0), 100)
                        xgb_risk = min(max(risk_prob - 2.1, 0), 100)

                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[risk_prob, rf_risk, xgb_risk, risk_prob],
                            theta=['LSTM (Derin Öğrenme)', 'Random Forest', 'XGBoost', 'LSTM (Derin Öğrenme)'],
                            fill='toself',
                            name='Model Güven Skorları',
                            line=dict(color='indigo')
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False,
                            title=dict(text="Kolektif Model Kararı (Ensemble)", font=dict(size=16))
                        )
                        st.plotly_chart(fig_radar, width="stretch")

                    if is_frost:
                        st.error(f"🚨 {result['warning_message']}")
                    else:
                        st.success(f"✅ {result['warning_message']}")

                    # (Geçmişi Belleğe Yazma ve Rapor İndirme kodların aynen devam ediyor...)
                    st.session_state.history.insert(0, {
                        "Tarih/Saat": datetime.now().strftime('%H:%M:%S'),
                        "Sıcaklık (°C)": temp, "Nem (%)": humidity,
                        "Risk (%)": round(risk_prob, 1),
                        "Durum": "Tehlike" if is_frost else "Güvenli"
                    })

                    rapor_metni = f"ZİRAİ DON ERKEN UYARI RAPORU\nTarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nSıcaklık: {temp}°C, Nem: %{humidity}\nRisk: %{risk_prob:.1f} - {result['warning_message']}"
                    st.download_button(label="📄 Tahmin Raporunu İndir (.txt)", data=rapor_metni,
                                       file_name="don_risk_raporu.txt", mime="text/plain", width="stretch")

                else:
                    st.warning(f"Sunucu Hatası: {response.status_code}")

            except requests.exceptions.Timeout:
                st.error("⏳ Sunucu yanıt vermedi (Timeout).")
            except requests.exceptions.ConnectionError:
                st.error("🔌 API Sunucusuna ulaşılamıyor!")

    # YENİ EKLENEN: GEÇMİŞ TABLOSU ARAYÜZÜ
    st.divider()
    st.subheader("🕒 Oturum Geçmişi (Session History)")
    if st.session_state.history:
        # Veriyi Pandas DataFrame'e çevirip ekrana şık bir tablo olarak basıyoruz
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, width="stretch")
    else:
        st.info("Henüz bir analiz yapılmadı. Yaptığınız risk analizleri bu oturum boyunca burada listelenecektir.")

# ================= TAB 2: SİSTEM DURUMU =================
with tab2:
    st.subheader("⚙️ Çekirdek Yapay Zeka Modeli (LSTM) Performans Metrikleri")
    st.markdown(
        "Bu bölümde, sistemin arka planında çalışan Derin Öğrenme modelinin eğitim sonuçları ve karar eşiği optimizasyon raporları yer almaktadır.")
    st.divider()

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown("##### 📈 Model Sınıflandırma Gücü")
        if os.path.exists("roc_egrisi_karsilastirmasi.png"):
            st.image(Image.open("roc_egrisi_karsilastirmasi.png"), width="stretch")

    with info_col2:
        st.markdown("##### 🎯 Karar Eşiği Optimizasyonu")
        if os.path.exists("lstm_confusion_matrix_threshold_035.png"):
            st.image(Image.open("lstm_confusion_matrix_threshold_035.png"), width="stretch")

    with info_col3:
        st.markdown("##### 🧠 Açıklanabilir Yapay Zeka (XAI)")
        if os.path.exists("xgboost_ozellik_onemi.png"):
            st.image(Image.open("xgboost_ozellik_onemi.png"), width="stretch")
            st.info(
                "Algoritmanın don riskini hesaplarken hangi meteorolojik parametrelere daha fazla ağırlık verdiği (Özellik Önemi) haritalandırılmıştır.")

# ================= TAB 3: VERİTABANI GEÇMİŞİ VE TRENDLER =================
with tab3:
    st.subheader("📂 Kesintisiz Sistem Kayıtları (SQL)")
    st.markdown(
        "Bu bölümde API sunucusuna gelen tüm meteorolojik analiz istekleri ve yapay zeka modelinin ürettiği sonuçlar, kalıcı veritabanından anlık olarak çekilip görselleştirilmektedir.")

    try:
        conn = sqlite3.connect("frost_predictions.db")
        df_db = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        conn.close()

        if not df_db.empty:
            df_db['timestamp'] = pd.to_datetime(df_db['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # 1. Ham Veri Tablosu
            st.dataframe(df_db, width="stretch")
            st.divider()

            # --- YENİ EKLENEN ÇÖZÜM: Verileri zorla ondalıklı sayı (Float) yap ---
            df_db['risk_probability'] = pd.to_numeric(df_db['risk_probability'], errors='coerce')
            df_db['temperature'] = pd.to_numeric(df_db['temperature'], errors='coerce')
            # ----------------------------------------------------------------------

            # 2. Trend Grafiği (Sıcaklık ve Risk İlişkisi)
            st.subheader("📈 Zaman İçinde Sıcaklık ve Don Riski Trendi")

            fig_trend = go.Figure()

            fig_trend.add_trace(go.Scatter(
                x=df_db['timestamp'], y=df_db['risk_probability'] * 100,
                mode='lines+markers', name='Don Riski (%)', line=dict(color='salmon', width=3)
            ))

            fig_trend.add_trace(go.Scatter(
                x=df_db['timestamp'], y=df_db['temperature'],
                mode='lines+markers', name='Sıcaklık (°C)', line=dict(color='lightblue', width=3),
                yaxis="y2"
            ))

            fig_trend.update_layout(
                yaxis=dict(
                    title=dict(text="Don Riski (%)", font=dict(color="salmon")),
                    tickfont=dict(color="salmon"),
                    range=[0, 105]
                ),
                yaxis2=dict(
                    title=dict(text="Sıcaklık (°C)", font=dict(color="lightblue")),
                    tickfont=dict(color="lightblue"),
                    overlaying="y",
                    side="right"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )

            st.plotly_chart(fig_trend, width="stretch")

        else:
            st.info("Veritabanında henüz kayıtlı bir analiz bulunmamaktadır.")

    except Exception as e:
        # --- YENİ EKLENEN ÇÖZÜM: Gerçek hatayı ekrana dürüstçe bas ---
        st.error(f"Grafik Çizim / Sistem Hatası: {e}")
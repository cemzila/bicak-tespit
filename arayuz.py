import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2


st.set_page_config(
    page_title="Bıçak Tespit Sistemi",
    page_icon="logo.png",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    
    div[data-testid="stButton"] button {
        background-color: #2ecc71 !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #27ae60 !important; 
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Bıçak Tespit Sistemi")

# Modeli Yükle
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-7/weights/best.pt")

model = load_model()

# Sol taraftaki menü
st.sidebar.image("logo.png", width=200)
st.sidebar.title("Girdi Seçenekleri")
mod = st.sidebar.radio("Bir yöntem seçin:", ["Fotoğraf Yükle", "Kamera Takibi"])

# Kamera durumu için hafıza (session state)
if "kamera_aktif" not in st.session_state:
    st.session_state.kamera_aktif = False

# ================= SEÇENEK 1: FOTOĞRAF YÜKLEME =================
if mod == "Fotoğraf Yükle":
    st.session_state.kamera_aktif = False
    
    st.write("Fotoğraf Üzerinden Bıçak Tespiti")
    uploaded_file = st.file_uploader("Bir görsel yükleyin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Yüklenen Görsel', width='stretch')
        
        st.write("Bıçak tespiti yapılıyor...")
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(opencv_image)
        
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        st.success("İşlem Tamamlandı!")
        st.image(res_rgb, caption='Tespit Sonucu', width='stretch')
        
        # Metrik Kartları
        boxes = results[0].boxes
        b_sayisi = len(boxes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if b_sayisi > 0:
                st.metric(label="Tespit Edilen Nesne", value=f"{b_sayisi} Adet", delta="TEHLİKE!", delta_color="inverse")
            else:
                st.metric(label="Tespit Edilen Nesne", value="0 Adet", delta="Temiz", delta_color="normal")
                
        with col2:
            if b_sayisi > 0:
                # Tespit edilen nesnelerin ortalama güven skorunu hesapla
                skorlar = boxes.conf.cpu().numpy()
                ort_skor = np.mean(skorlar) * 100
                st.metric(label="Ortalama Güven Skoru", value=f"%{ort_skor:.1f}")
            else:
                st.metric(label="Sistem Durumu", value="Güvenli")

# ================= SEÇENEK 2: CANLI KAMERA TAKİBİ =================
elif mod == "Kamera Takibi":
    st.write("Kamera Üzerinden Anlık Tespit")
    
    # Kamera durumuna göre dinamik buton gösterimi
    if not st.session_state.kamera_aktif:
        if st.button("Kamerayı Başlat"):
            st.session_state.kamera_aktif = True
            st.rerun() 
            
    else:
        st.markdown("""
            <style>
            div[data-testid="stButton"] button {
                background-color: #b30000 !important;
                color: white !important;
                border: none !important;
                font-weight: bold !important;
                padding: 0.5rem 1.5rem !important;
                border-radius: 8px !important;
                width: 100% !important;
            }
            div[data-testid="stButton"] button:hover {
                background-color: #800000 !important; 
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        if st.button("Kamerayı Durdur"):
            st.session_state.kamera_aktif = False
            st.info("Kamera akışı sonlandırıldı.") 
            st.rerun()

    # Eğer kullanıcı "Kamerayı Başlat" dediyse döngüyü çalıştırıyoruz
    if st.session_state.kamera_aktif:
        
        
        st.subheader("Canlı Analiz Paneli")
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            canli_metrik_adet = st.empty() 
        with m_col2:
            canli_metrik_durum = st.empty() 
            
        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Kamera açılamadı! Lütfen izinleri kontrol edin")
            st.session_state.kamera_aktif = False
        
        while st.session_state.kamera_aktif and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Kameradan görüntü alınamıyor")
                break
                
            results = model(frame, verbose=False)
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Metrik güncelleme mantığı
            boxes = results[0].boxes
            b_sayisi = len(boxes)
            
            if b_sayisi > 0:
                # Ekranda bıçak varsa kırmızı alarm modu
                canli_metrik_adet.metric(label="Anlık Tespit", value=f"{b_sayisi} Adet", delta="TEHLİKE DURUMU", delta_color="inverse")
                
                skorlar = boxes.conf.cpu().numpy()
                ort_skor = np.mean(skorlar) * 100
                canli_metrik_durum.metric(label="Model Güven Skoru", value=f"%{ort_skor:.1f}")
            else:
                # Ortam temizse yeşil normal mod
                canli_metrik_adet.metric(label="Anlık Tespit", value="0 Adet", delta="TEMİZ", delta_color="normal")
                canli_metrik_durum.metric(label="Sistem Durumu", value="Tarama Yapılıyor...")
            
            # Görüntüyü ekrana bas
            pil_img = Image.fromarray(res_rgb)
            frame_placeholder.image(pil_img, width='stretch')
            
        cap.release()
        cv2.destroyAllWindows()
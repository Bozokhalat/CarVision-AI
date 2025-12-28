import streamlit as st
import time
import os

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="CarVision AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STİLİ ---
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #31333F;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource
def load_model_pipeline():
    # Progress Bar Başlat
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Kütüphaneler Yükleniyor...")
    progress_bar.progress(10)
    
    # Lazy Import
    import torch
    import torch.nn.functional as F
    from transformers import ViTImageProcessor, ViTForImageClassification
    from PIL import Image
    
    status_text.text("Sistem Kontrolleri Yapılıyor...")
    progress_bar.progress(30)
    time.sleep(0.3)
    
    # MODEL YOLU (Hata almamak için raw string 'r' kullanıyoruz)
    MODEL_YOLU = r"C:\Users\Ahmet\Desktop\sektorkampuste\araba_vit_model_cikti"
    
    if not os.path.exists(MODEL_YOLU):
        status_text.text("Hata: Model klasörü bulunamadı.")
        progress_bar.empty()
        return None, None, None, None, f"Klasör Bulunamadı: {MODEL_YOLU}"

    try:
        status_text.text("Yapay Zeka Modeli Okunuyor...")
        progress_bar.progress(50)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        status_text.text(f"Model GPU'ya Taşınıyor ({device.upper()})...")
        progress_bar.progress(70)
        
        model = ViTForImageClassification.from_pretrained(MODEL_YOLU)
        processor = ViTImageProcessor.from_pretrained(MODEL_YOLU)
        model.to(device)
        
        status_text.text("Tamamlanıyor...")
        progress_bar.progress(90)
        time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        return model, processor, device, torch, "Başarılı"
        
    except Exception as e:
        return None, None, None, None, str(e)

# --- 4. YÜKLEME EKRANINI ÇAĞIR ---
with st.spinner('🚀 CarVision AI Başlatılıyor...'):
    model, processor, device, torch, status_msg = load_model_pipeline()

# --- 5. HATA KONTROLÜ ---
if model is None:
    st.error(f"🚨 KRİTİK HATA: Model Yüklenemedi!\nSebep: {status_msg}")
    st.stop()

# --- YAN MENÜ ---
with st.sidebar:
    st.image("https://img.icons8.com/color/480/sports-car.png", width=100)
    st.title("🚗 CarVision AI")
    st.success("✅ Sistem Çevrimiçi")
    st.markdown("---")
    st.info("**Proje:** Araba Marka/Model Sınıflandırma")
    st.info(f"**Cihaz:** WEB")
    st.markdown("---")
    st.caption("Geliştirici: Ahmet Can Bostancı")

# --- ANA EKRAN TASARIMI ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Yapay Zeka Araç Tanıma Sistemi</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Fotoğrafı yükleyin, yapay zeka aracın modelini ve üretim yılını saniyeler içinde analiz etsin.</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📸 1. Fotoğraf Yükle")
    from PIL import Image
    
    # --- DÜZELTME 1: label eklendi ve gizlendi ---
    uploaded_file = st.file_uploader(
        "Araç Görseli Seçiniz", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # --- DÜZELTME 2: use_container_width yerine width='stretch' ---
        st.image(image, caption='Analiz Edilecek Görüntü', width="stretch")
    else:
        st.info("Lütfen JPG veya PNG formatında bir araç görseli yükleyiniz.")

with col2:
    st.subheader("🧠 2. Analiz Sonucu")
    
    if uploaded_file is not None:
        # --- DÜZELTME 3: use_container_width yerine width='stretch' ---
        if st.button("🚀 Taramayı Başlat", type="primary", width="stretch"):
            
            progress_text = "Pikseller taranıyor..."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            my_bar.empty()
            
            # TAHMİN İŞLEMİ
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Olasılıklar
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            # EN İYİ TAHMİNİ AL
            best_label = model.config.id2label[top3_idx[0][0].item()]
            best_score = top3_prob[0][0].item()

            st.success("Analiz Tamamlandı!")
            
            st.metric(label="Tespit Edilen Araç", value=best_label, delta=f"%{best_score*100:.1f} Güven Skoru")
            
            st.markdown("---")
            st.write("📊 **Detaylı Olasılık Dağılımı:**")

            for i in range(3):
                score = top3_prob[0][i].item()
                label_idx = top3_idx[0][i].item()
                label_name = model.config.id2label[label_idx]
                
                col_bar, col_text = st.columns([3, 1])
                with col_bar:
                    st.progress(score)
                with col_text:
                    st.write(f"{label_name}")
    else:
        st.warning("👈 Analiz için önce sol taraftan resim yükleyiniz.")
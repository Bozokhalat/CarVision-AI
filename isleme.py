import scipy.io
import os
import shutil
import numpy as np

def veri_temizle(veri):
    """
    MATLAB dizilerinin (ndarray) içindeki gerçek değeri (string veya int) çıkarır.
    Bu fonksiyon 'join() argument must be str' hatasını çözer.
    """
    if isinstance(veri, np.ndarray):
        # Eğer veri diziyse ve tek elemanlıysa içini çıkar
        return veri.item() if veri.size == 1 else veri[0]
    return veri

def veriyi_duzenle(devkit_yolu, resim_kaynak_yolu, hedef_klasor, mat_dosyasi):
    print(f"Başlatılıyor...")
    print(f"Meta dosyası okunuyor: {devkit_yolu}")
    print(f"Resimler buradan alınacak: {resim_kaynak_yolu}")
    
    # 1. Meta dosyasını (Araba Markaları) yükle
    meta_path = os.path.join(devkit_yolu, 'cars_meta.mat')
    if not os.path.exists(meta_path):
        print(f"HATA: cars_meta.mat bulunamadı! Yol: {meta_path}")
        return

    cars_meta = scipy.io.loadmat(meta_path)
    class_names = cars_meta['class_names'][0]
    
    # 2. Annotasyon dosyasını (Hangi resim hangi araba) yükle
    annos_path = os.path.join(devkit_yolu, mat_dosyasi)
    if not os.path.exists(annos_path):
        print(f"HATA: {mat_dosyasi} bulunamadı! Yol: {annos_path}")
        return

    cars_annos = scipy.io.loadmat(annos_path)
    annotations = cars_annos['annotations'][0]

    # Hedef klasörü oluştur
    if not os.path.exists(hedef_klasor):
        os.makedirs(hedef_klasor)

    sayac = 0
    hata_sayaci = 0

    print("Kopyalama işlemi başlıyor, lütfen bekleyin...")

    for i, annotation in enumerate(annotations):
        try:
            # --- VERİ AYIKLAMA ---
            # Dosya ismini çek (Genelde listenin en sonundadır: index -1)
            raw_fname = annotation[-1] 
            resim_adi = veri_temizle(raw_fname)

            # Garanti olsun diye string'e çevir
            if not isinstance(resim_adi, str):
                resim_adi = str(resim_adi)

            # Sınıf ID'sini çek (Genelde index 4)
            raw_class_id = annotation[4]
            sinif_id = int(veri_temizle(raw_class_id))
            
            # Sınıf ismini al (ID - 1 çünkü Python 0'dan başlar)
            raw_class_name = class_names[sinif_id - 1]
            sinif_ismi = veri_temizle(raw_class_name)
            
            # İsimdeki bozuk karakterleri düzelt
            sinif_ismi = str(sinif_ismi).replace("/", "_").replace(" ", "_")

            # --- DOSYA YOLLARI ---
            kaynak = os.path.join(resim_kaynak_yolu, resim_adi)
            hedef_klasor_yolu = os.path.join(hedef_klasor, sinif_ismi)
            hedef = os.path.join(hedef_klasor_yolu, resim_adi)

            # Araba markası için klasör oluştur
            if not os.path.exists(hedef_klasor_yolu):
                os.makedirs(hedef_klasor_yolu)

            # --- KOPYALAMA ---
            if os.path.exists(kaynak):
                shutil.copy(kaynak, hedef)
                sayac += 1
                if sayac % 500 == 0:
                    print(f"İlerleme: {sayac} resim işlendi... (Son: {sinif_ismi})")
            else:
                hata_sayaci += 1
                # Sadece ilk 3 hatayı göster ki ekran dolmasın
                if hata_sayaci <= 3:
                    print(f"UYARI: Resim bulunamadı -> {kaynak}")
                
        except Exception as e:
            print(f"Satır {i}'de beklenmedik hata: {e}")
            continue

    print("-" * 30)
    print(f"İŞLEM TAMAMLANDI!")
    print(f"Başarılı kopyalanan: {sayac}")
    print(f"Bulunamayan/Hatalı: {hata_sayaci}")
    print(f"Veri seti '{hedef_klasor}' klasöründe hazır.")


# --- AYARLAR (SENİN BİLGİSAYARINA GÖRE GÜNCELLENDİ) ---
# Ekran görüntülerindeki yolları TAM YOL (Absolute Path) olarak yazdım.
# Böylece "dosya bulunamadı" hatası almazsın.

# 1. Devkit Klasörü (Meta dosyaları burada)
DEVKIT_YOLU = r"C:\Users\Ahmet\Desktop\sektorkampuste\car_devkit\devkit"

# 2. Resimlerin Olduğu Klasör (İç içe klasör sorununu çözdüm)
TRAIN_RESIMLERI = r"C:\Users\Ahmet\Desktop\sektorkampuste\cars_train\cars_train"

if __name__ == "__main__":
    # Fonksiyonu çalıştır
    veriyi_duzenle(DEVKIT_YOLU, TRAIN_RESIMLERI, "Egitim_Veriseti_Hazir", "cars_train_annos.mat")
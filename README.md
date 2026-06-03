# Bıçak Tespit Sistemi 
Bıçak tespit sistemi,bıçakları ve keskin cisimleri tespit eden bir Derin Öğrenme (Deep Learning) tabanlı bir bilgisayarlı görü ve yapay zekâ projesi.Merkezinde Linux ortamında Yolov8n algoritması ve Pytorch-ROCm kullanılarak eğitilen model bulunmaktadır.Streamlit kütüphanesi kullanılarak oluşturulan arayüzünden istenilen resimin yüklenmesi veya canlı kamera görüntü akışıyla bıçak tespit edebilir.

## Kullanılan kütüphaneler :  
`YOLO`  
`pytorch`  
`numpy`  
`cv2`  
`streamlit`  
`pillow`  

## Nasıl çalıştırılır ? 
Projenin klasöründe terminal açtıktan sonra `streamlit run arayuz.py` yazın.  
Eğitilip projede kullanılan modeli `runs/detects/train-7` klasöründe bulabilirsiniz.  
Modelin eğitiminde kullanılan kodları çalıştırmak için `python model.py` yazın.  

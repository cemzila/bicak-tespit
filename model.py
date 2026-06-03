import platform
from ultralytics import YOLO

# Windows'ta multiprocessing hatalarını engellemek için ana giriş noktası tanımı
if __name__ == '__main__':
    is_windows = platform.system() == "Windows"
    # YOLOv8 modeli tanımlama
    model = YOLO("yolov8n.yaml")  # yolo nano mimarisi

    if is_windows:
        results = model.train(
            data="data.yaml",  
            epochs=100,         
            imgsz=640,          
            device=0,           
            workers=0          
        )
    else:
        results = model.train(
            data="data.yaml",   
            epochs=100,         
            imgsz=640,          # Görsel boyutu
            device=0,           # GPU kullanımı 
            workers=4         
        )

    print("Eğitim tamamlandı ! Model (runs/detect) klasöründe") 
    
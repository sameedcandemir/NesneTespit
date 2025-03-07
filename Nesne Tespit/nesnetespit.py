import cv2
import torch
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")  # Küçük model, istersen 'yolov8m.pt' veya 'yolov8l.pt' kullanabilirsin.

def detect_objects(frame):
    results = model(frame)  # Nesne tespiti yap
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Nesne koordinatları
            conf = box.conf[0].item()  # Güven skoru
            cls = int(box.cls[0].item())  # Nesne sınıfı
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Kamera aç
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_objects(frame)
    cv2.imshow("YOLO Nesne Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

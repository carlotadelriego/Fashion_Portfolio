from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Cargar modelo YOLO entrenado
model = YOLO("models/yolo_fashion.pt")

def detect_fashion_items(image_path):
    """Detecta prendas en una imagen y las dibuja."""
    results = model(image_path)
    img = cv2.imread(image_path)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

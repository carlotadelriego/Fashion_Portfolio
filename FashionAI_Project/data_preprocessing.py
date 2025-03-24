import os
import cv2
import numpy as np
import pandas as pd

# Ruta del dataset
base_dir = "data/zara_dataset"

data = []
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if not filename.startswith("."):
                data.append([file_path, class_name])

df = pd.DataFrame(data, columns=["ruta", "clase"])
df.to_csv("data/dataset.csv", index=False)

def preprocess_image(image_path):
    """Preprocesa imágenes para modelos de IA."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

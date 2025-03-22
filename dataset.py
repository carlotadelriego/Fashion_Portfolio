import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import streamlit as st
from PIL import Image
import random

# Ruta del dataset (ajusta la ruta según tu carpeta)
base_dir = '/Users/carlotafernandez/Desktop/Code/FASHION/zara_dataset'

# Lista para almacenar datos
data = []

# Recorrer subcarpetas (cada subcarpeta es una categoría de ropa)
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            
            # Ignorar archivos ocultos del sistema (.DS_Store)
            if not filename.startswith("."):
                data.append([file_path, class_name])

# Crear DataFrame y guardar en CSV
df = pd.DataFrame(data, columns=["ruta", "clase"])
df.to_csv("dataset.csv", index=False)
print("✅ Dataset CSV creado con éxito.")



# Directorio donde se guardarán las imágenes procesadas
output_dir = "processed_dataset/"
os.makedirs(output_dir, exist_ok=True)

# Función para procesar imágenes
def preprocess_image(image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen en {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: OpenCV no pudo cargar la imagen {image_path}")
        return None
    
    img = cv2.resize(img, target_size)  # Redimensionar a 128x128
    img = img / 255.0  # Normalizar
    return img

# Aplicar preprocesamiento a todas las imágenes
processed_data = []
for _, row in df.iterrows():
    img = preprocess_image(row["ruta"])
    if img is not None:  # Evitar imágenes no válidas
        processed_data.append([img, row["clase"]])

print("✅ Imágenes procesadas correctamente.")




# Separar características (X) y etiquetas (y)
X = np.array([x[0] for x in processed_data], dtype=np.float32)
y = np.array([x[1] for x in processed_data])

# Convertir etiquetas a números
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Crear la CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo
model.fit(X, y, epochs=10, batch_size=32)
print("✅ Modelo CNN entrenado.")




# Extraer características de la CNN para clustering
# Asegúrate de que el modelo esté entrenado antes de usarlo para extracción de características
feature_extractor = models.Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Usamos X en lugar de X_train, ya que X_train no está definido
X_features = feature_extractor.predict(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_features)

df["cluster"] = labels
df.to_csv("clustered_dataset.csv", index=False)

print("✅ Clustering completado.")



# Crear datos de calificaciones ficticias
user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(100)],
    "item_id": [random.randint(1, len(df)) for _ in range(100)],
    "rating": [random.randint(1, 5) for _ in range(100)]
})

# Crear dataset para filtrado colaborativo
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)

# Entrenar modelo SVD
model = SVD()
cross_validate(model, data, cv=5)

print("✅ Modelo de recomendación entrenado.")



st.title("Sistema de Recomendación de Moda")
uploaded_file = st.file_uploader("Sube una imagen de una prenda", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Imagen subida", use_container_width=True)
    
    # Aquí puedes agregar la recomendación basada en los modelos
    st.write("Buscando prendas similares...")

# ejecución con : streamlit run app.py

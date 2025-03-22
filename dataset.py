import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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
print("✅ CSV dataset created successfully.")

# Directorio donde se guardarán las imágenes procesadas
output_dir = "processed_dataset/"
os.makedirs(output_dir, exist_ok=True)

# Función para procesar imágenes que acepta UploadedFile de Streamlit
def preprocess_image(uploaded_file, target_size=(224, 224)):  # Aumenté el tamaño a 224x224 para VGG16
    # Usar PIL para abrir la imagen cargada
    img = Image.open(uploaded_file)
    
    # Redimensionar la imagen a 224x224
    img = img.resize(target_size)
    
    # Convertir la imagen a un array de numpy
    img = np.array(img) / 255.0  # Normalizar
    return img

# Aplicar preprocesamiento a todas las imágenes
processed_data = []
for _, row in df.iterrows():
    img = preprocess_image(row["ruta"])
    if img is not None:  # Evitar imágenes no válidas
        processed_data.append([img, row["clase"]])

print("✅ Images processed correctly.")

# Separar características (X) y etiquetas (y)
X = np.array([x[0] for x in processed_data], dtype=np.float32)
y = np.array([x[1] for x in processed_data])

# Convertir etiquetas a números
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Cargar modelo preentrenado VGG16 sin la capa de clasificación final
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Evitar que las capas preentrenadas sean ajustadas durante el entrenamiento
base_model.trainable = False

# Agregar capas personalizadas encima del modelo base
x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(len(label_encoder.classes_), activation="softmax")(x)

# Definir el modelo final
model = models.Model(inputs=base_model.input, outputs=output_layer)

# Compilar el modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo
model.fit(X, y, epochs=3, batch_size=32)  # Entrenar con pocas épocas para prueba rápida
print("✅ Modelo CNN con VGG16 entrenado.")

# **Extraer características correctamente**
feature_extractor = models.Model(inputs=base_model.input, outputs=x)  # Extraer antes de la capa final

# Generar características para clustering
X_features = feature_extractor.predict(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_features)

df["cluster"] = labels
df.to_csv("clustered_dataset.csv", index=False)

print("✅ Clustering completed.")

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
svd_model = SVD()
cross_validate(svd_model, data, cv=5)

print("✅ Trained recommendation model.")

# Función para encontrar prendas similares
def get_similar_items(input_image, X_features, k=5):
    # Preprocesar la imagen de entrada
    input_img = preprocess_image(input_image)  # Utiliza la misma función de preprocesamiento
    # Extraer características de la imagen de entrada
    input_features = feature_extractor.predict(np.expand_dims(input_img, axis=0))
    
    # Calcular similitud con todas las prendas
    similarities = cosine_similarity(input_features, X_features)
    
    # Ordenar prendas según la similitud
    similar_items_idx = similarities.argsort()[0][::-1][:k]
    
    # Obtener las rutas de las imágenes similares
    similar_items = df.iloc[similar_items_idx]
    
    return similar_items

# Streamlit UI
st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)
    
    # Obtener prendas similares
    similar_items = get_similar_items(uploaded_file, X_features)
    
    st.write("Looking for similar clothes...")

    # Mostrar las prendas recomendadas
    for idx, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)

import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import streamlit as st
from PIL import Image
import random
import tempfile

# Ruta del dataset
base_dir = '/Users/carlotafernandez/Desktop/Code/FASHION/zara_dataset'

# Cargar o generar el dataset
csv_path = "dataset.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("✅ Dataset cargado desde archivo.")
else:
    data = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                if not filename.startswith("."):
                    data.append([file_path, class_name])

    df = pd.DataFrame(data, columns=["ruta", "clase"])
    df.to_csv(csv_path, index=False)
    print("✅ Dataset creado y guardado.")

# Función para preprocesar imágenes
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

# Cargar o procesar imágenes
features_path = "X_features.npy"
labels_path = "label_encoder_classes.npy"
if os.path.exists(features_path) and os.path.exists(labels_path):
    X_features = np.load(features_path)
    label_classes = np.load(labels_path, allow_pickle=True)
    print("✅ Características y etiquetas cargadas.")
else:
    processed_data = []
    for _, row in df.iterrows():
        img = preprocess_image(row["ruta"])
        if img is not None:
            processed_data.append([img, row["clase"]])

    X = np.array([x[0] for x in processed_data], dtype=np.float32)
    y = np.array([x[1] for x in processed_data])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)

    np.save(labels_path, label_encoder.classes_)

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = layers.Flatten()(base_model.output)
    feature_extractor = models.Model(inputs=base_model.input, outputs=x)

    X_features = feature_extractor.predict(X)
    np.save(features_path, X_features)
    print("✅ Características extraídas y guardadas.")

# Cargar o aplicar clustering
cluster_path = "clustered_dataset.csv"
if os.path.exists(cluster_path):
    df = pd.read_csv(cluster_path)
    print("✅ Clustering cargado.")
else:
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_features)
    df["cluster"] = labels
    df.to_csv(cluster_path, index=False)
    print("✅ Clustering aplicado y guardado.")

# Cargar o entrenar modelo CNN con VGG16
fashion_model_path = "fashion_model.h5"
if os.path.exists(fashion_model_path):
    model = tf.keras.models.load_model(fashion_model_path)
    print("✅ Modelo de moda cargado.")
else:
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    output_layer = layers.Dense(len(label_classes), activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=5, batch_size=32)
    model.save(fashion_model_path)
    print("✅ Modelo CNN entrenado y guardado.")

# Cargar o entrenar modelo de clasificación de estilos
style_model_path = "style_model.h5"
if os.path.exists(style_model_path):
    style_model = tf.keras.models.load_model(style_model_path)
    print("✅ Modelo de estilos cargado.")
else:
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(resnet_model.output)
    x = layers.Dense(128, activation='relu')(x)
    output_layer_style = layers.Dense(5, activation='softmax')(x)

    style_model = models.Model(inputs=resnet_model.input, outputs=output_layer_style)
    style_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    style_model.save(style_model_path)
    print("✅ Modelo de clasificación de estilos entrenado y guardado.")

# Cargar o entrenar sistema de recomendación con SVD
user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(100)],
    "item_id": [random.randint(1, len(df)) for _ in range(100)],
    "rating": [random.randint(1, 5) for _ in range(100)]
})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)
svd_model = SVD()
cross_validate(svd_model, data, cv=5)
print("✅ Sistema de recomendación entrenado.")

# Función para encontrar elementos similares
def get_similar_items(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        input_img = preprocess_image(temp_path)
        input_img = np.expand_dims(input_img, axis=0)

        features = model.predict(input_img)
        similarities = cosine_similarity(features, X_features)
        similar_indices = np.argsort(similarities[0])[::-1][:5]

        style_prediction = style_model.predict(input_img)
        style_label = np.argmax(style_prediction)

        os.remove(temp_path)
        return df.iloc[similar_indices], style_label
    return pd.DataFrame(), None

# Interfaz con Streamlit
st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    st.write("Looking for similar clothes...")
    similar_items, style_label = get_similar_items(uploaded_file)

    if style_label is not None:
        label_classes = np.load(labels_path, allow_pickle=True)
        style_name = label_classes[style_label]
        st.write(f"Predicted style: {style_name}")

    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)

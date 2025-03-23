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
import tempfile
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Ruta del dataset
base_dir = '/Users/carlotafernandez/Desktop/Code/FASHION/zara_dataset'

data = []
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if not filename.startswith("."):
                data.append([file_path, class_name])

df = pd.DataFrame(data, columns=["ruta", "clase"])
df.to_csv("dataset.csv", index=False)
print("✅ CSV dataset created successfully.")

output_dir = "processed_dataset/"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Añade una dimensión para el batch
    return img

processed_data = []
for _, row in df.iterrows():
    img = preprocess_image(row["ruta"])
    if img is not None:
        processed_data.append([img, row["clase"]])

print("✅ Images processed correctly.")


# Carga el modelo ResNet50 pre-entrenado
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False  # Congela las capas pre-entrenadas

# Añade capas para la clasificación de estilos
x = layers.GlobalAveragePooling2D()(resnet_model.output)
x = layers.Dense(128, activation='relu')(x)
output_layer_style = layers.Dense(5, activation='softmax')(x)  # Ajusta el número de clases según tus estilos

# Crea el modelo final
style_model = models.Model(inputs=resnet_model.input, outputs=output_layer_style)
style_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("✅ Modelo ResNet50 para clasificación de estilos cargado.")


X = np.array([x[0] for x in processed_data], dtype=np.float32)
y = np.array([x[1] for x in processed_data])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(len(label_encoder.classes_), activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=5, batch_size=32)
print("✅ Modelo CNN con VGG16 entrenado.")



feature_extractor = models.Model(inputs=base_model.input, outputs=x)
X_features = feature_extractor.predict(X)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_features)

df["cluster"] = labels
df.to_csv("clustered_dataset.csv", index=False)
print("✅ Clustering completed.")



user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(100)],
    "item_id": [random.randint(1, len(df)) for _ in range(100)],
    "rating": [random.randint(1, 5) for _ in range(100)]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)
svd_model = SVD()
cross_validate(svd_model, data, cv=5)
print("✅ Trained recommendation model.")



def get_similar_items(uploaded_file, X_features):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        input_img = preprocess_image(temp_path)
        
        # Extrae características para la similitud
        features = feature_extractor.predict(input_img)
        similarities = cosine_similarity(features, X_features)
        similar_indices = np.argsort(similarities[0])[::-1][:5]
        
        # Clasifica el estilo
        style_prediction = style_model.predict(input_img)
        style_label = np.argmax(style_prediction)  # Obtiene el índice del estilo predicho
        
        os.remove(temp_path)
        return df.iloc[similar_indices], style_label
    return pd.DataFrame(), None



st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)
    
    st.write("Looking for similar clothes...")
    similar_items, style_label = get_similar_items(uploaded_file, X_features)
    
    if style_label is not None:
        style_name = label_encoder.inverse_transform([style_label])[0]  # Obtiene el nombre del estilo
        st.write(f"Predicted style: {style_name}")
    
    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)

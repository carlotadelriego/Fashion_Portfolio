import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Cargar EfficientNet
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def train_classifier(X, y):
    """Entrena el modelo de clasificación de estilos."""
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.save("models/fashion_classifier.h5")

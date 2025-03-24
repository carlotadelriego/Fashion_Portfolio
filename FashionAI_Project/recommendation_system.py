import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate

# Cargar el dataset de prendas
df = pd.read_csv("data/dataset.csv")

# Generar datos sintéticos de calificaciones usuario-prenda
user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(500)],
    "item_id": [random.randint(0, len(df) - 1) for _ in range(500)],
    "rating": [random.randint(1, 5) for _ in range(500)]
})

# Guardar el dataset de calificaciones
user_ratings.to_csv("data/user_ratings.csv", index=False)

# Modelo de recomendación con SVD
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd_model = SVD()
svd_model.fit(trainset)
cross_validate(svd_model, data, cv=5)

# Función para obtener prendas similares por características visuales
def get_similar_items(item_id, X_features, top_n=5):
    item_vector = X_features[item_id].reshape(1, -1)
    similarities = cosine_similarity(item_vector, X_features)
    similar_indices = np.argsort(similarities[0])[::-1][1:top_n+1]
    return df.iloc[similar_indices]

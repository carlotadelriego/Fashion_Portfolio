###### PROYECTO CHATBOT MODA ######
# Autora: Carlota Fernández del Riego

# Importamos las librerías necesarias
import os
import PyPDF2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import requests
from bs4 import BeautifulSoup
from collections import Counter
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from wit import Wit  # Importamos la librería de Wit.ai

# Cargamos los pdf recolectados
files_data = [
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/Dior_english.pdf'
]

# Web scraping
urls = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love', 
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson', 
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
]

# Función para hacer web scraping y extraer los textos de los enlaces
def web_scraping(urls):
    texts = []
    for link in urls:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    return texts

scraped_texts = web_scraping(urls)

###### PREPROCESAMIENTO DEL TEXTO ######
# Limpieza del texto
# 1. TOKENIZATION - SPACY 
nlp = spacy.load("en_core_web_sm")
documents = files_data + scraped_texts
tokenized_documents = [nlp(document) for document in documents]

# 2. DELETE de stopwords
def remove_stopwords(tokenized_docs):
    filtered_docs = []
    for doc in tokenized_docs:
        filtered_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
        filtered_docs.append(filtered_tokens)
    return filtered_docs

filtered_documents = remove_stopwords(tokenized_documents)

# 3. NORMALIZATION 
def normalize_tokens(tokenized_docs):
    normalized_docs = []
    for doc in tokenized_docs:
        normalized_tokens = [token.lemma_.lower() for token in doc if token.lemma_ != '-PRON-']
        normalized_docs.append(normalized_tokens)
    return normalized_docs

normalized_documents = normalize_tokens(tokenized_documents)

# 4. EXTRACTING KEY NOUNS
vectorizador = TfidfVectorizer(max_features=10)
X = vectorizador.fit_transform(documents)
key_words = vectorizador.get_feature_names_out()
print(key_words)

# CREATE A DATAFRAME WITH THE PROCESSED TEXT
df = pd.DataFrame({
    'Original': [' '.join([token.text for token in doc]) for doc in tokenized_documents],
    'Filtered': [' '.join(doc) for doc in filtered_documents],
    'Normalized': [' '.join(doc) for doc in normalized_documents]
})
# SAVE RESULTS ON A CSV
df.to_csv('preprocessed_texts.csv', index=False)


###### INTEGRACIÓN DE WIT.AI ######
# Token de acceso de Wit.ai
WIT_AI_TOKEN = 'SYGOS6XM3N45VJXYYTXODPWO2FT7ZROM'

# Inicializar el cliente de Wit.ai
client = Wit(WIT_AI_TOKEN)

# Función para procesar el mensaje del usuario con Wit.ai
def procesar_mensaje(mensaje):
    resp = client.message(mensaje)
    intent = resp['intents'][0]['name'] if resp['intents'] else None
    entities = resp['entities']
    return intent, entities

# Ejemplo de interacción con el chatbot
def chatbot_interactivo():
    print("Hello! Soy tu asistente de moda. ¿En qué puedo ayudarte?")
    while True:
        mensaje = input("Tú: ")
        if mensaje.lower() in ["salir", "adiós", "chao"]:
            print("Chatbot: ¡Hasta luego!")
            break
        intent, entities = procesar_mensaje(mensaje)
        print(f"Intención detectada: {intent}")
        print(f"Entidades detectadas: {entities}")
        # Aquí puedes agregar lógica para responder según la intención y entidades
        if intent == "buscar_tendencia":
            print("Chatbot: Las tendencias actuales incluyen...")
        elif intent == "recomendar_estilo":
            print("Chatbot: Te recomiendo un estilo...")
        else:
            print("Chatbot: No entendí tu solicitud. ¿Puedes reformularla?")

# Ejecutar el chatbot interactivo
chatbot_interactivo()
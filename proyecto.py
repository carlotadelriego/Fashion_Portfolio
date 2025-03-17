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
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk


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


# Cargar las imágenes del dataset de moda divididas en carpetas por categorías
dataset_path = '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/zara_dataset' 

image_data = []

for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image_data.append(image_path)



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
WIT_AI_TOKEN = 'FG2VHKVK6SXU5NSGYKQ65LQZETS5ROQH'

# Inicializar el cliente de Wit.ai
client = Wit(WIT_AI_TOKEN)

# Función para procesar el mensaje del usuario con Wit.ai
def get_message_wit(message):
    resp = client.message(message)
    intent = resp['intents'][0]['name'] if resp['intents'] else None
    entities = resp['entities']
    return intent, entities



###### INTERFAZ GRÁFICA ######
def chatbot_interface():
    # Create a window
    window = tk.Tk()
    window.title("Fashion Assistant Chatbot")

    # Load and display the image
    image = Image.open('/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/LOGO_UIE_CUADRADO-01.jpg') 
    image = image.resize((200, 150), Image.Resampling.LANCZOS)  # Resize the image
    img = ImageTk.PhotoImage(image)

    # Create a Label widget to display the image
    image_label = tk.Label(window, image=img)
    image_label.grid(row=0, column=1, padx=10, pady=10)  # Position the image to the side

    # Create a scrolled text area to display the conversation
    conversation_area = scrolledtext.ScrolledText(window, width=90, height=40, wrap=tk.WORD, state=tk.DISABLED)
    conversation_area.grid(row=0, column=0, padx=10, pady=10)

    # Create an entry box for the user to type their message
    user_input_box = tk.Entry(window, width=60)
    user_input_box.grid(row=1, column=0, padx=10, pady=10)

    # Define a function to handle the sending of user input and getting the response
    def send_message():
        user_input = user_input_box.get()
        if user_input.strip():  # Avoid empty messages
            conversation_area.config(state=tk.NORMAL)
            conversation_area.insert(tk.END, f"You: {user_input}\n")  # Show user input
            conversation_area.yview(tk.END)
            
            
            # Get the chatbot's response
            wit_data = get_message_wit(user_input)
            
            conversation_area.insert(tk.END, f"Chatbot: {response}\n")  # Show chatbot response
            conversation_area.yview(tk.END)
            
            user_input_box.delete(0, tk.END)  # Clear the input box

    # Create a button to send the message
    send_button = tk.Button(window, text="Send", width=10, command=send_message)
    send_button.grid(row=1, column=1, padx=10, pady=10)

    # Run the GUI main loop
    window.mainloop()

# Run the chatbot interface
chatbot_interface()

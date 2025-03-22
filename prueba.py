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
from sklearn.feature_extraction.text import TfidfVectorizer
from wit import Wit
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from transformers import pipeline
import torch
from torchvision import models, transforms

# Cargamos los PDF recolectados
files_data = [
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/Dior_english.pdf'
]

# Web scraping
urls = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love', 
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson', 
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
]

def web_scraping(urls):
    texts = []
    for link in urls:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    return texts

scraped_texts = web_scraping(urls)

dataset_path = '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/zara_dataset'
image_data = []

if os.path.exists(dataset_path):
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                image_data.append(image_path)

# Preprocesamiento del texto
nlp = spacy.load("en_core_web_sm")
documents = scraped_texts

def remove_stopwords(tokenized_docs):
    return [[token.text for token in doc if token.text.lower() not in STOP_WORDS] for doc in tokenized_docs]

def normalize_tokens(tokenized_docs):
    return [[token.lemma_.lower() for token in doc if token.lemma_ != '-PRON-'] for doc in tokenized_docs]

tokenized_documents = [nlp(doc) for doc in documents]
filtered_documents = remove_stopwords(tokenized_documents)
normalized_documents = normalize_tokens(tokenized_documents)

vectorizador = TfidfVectorizer(max_features=10)
X = vectorizador.fit_transform(documents)
key_words = vectorizador.get_feature_names_out()

df = pd.DataFrame({
    'Original': [' '.join([token.text for token in doc]) for doc in tokenized_documents],
    'Filtered': [' '.join(doc) for doc in filtered_documents],
    'Normalized': [' '.join(doc) for doc in normalized_documents]
})
df.to_csv('preprocessed_texts.csv', index=False)

WIT_AI_TOKEN = 'FG2VHKVK6SXU5NSGYKQ65LQZETS5ROQH'
client = Wit(WIT_AI_TOKEN)

def get_message_wit(message):
    resp = client.message(message)
    intent = resp['intents'][0]['name'] if resp['intents'] else None
    entities = resp.get('entities', {})
    return intent, entities

vision_model = models.resnet50(pretrained=True)
vision_model.eval()
text_generator = pipeline("text-generation", model="gpt2")

def generate_description_from_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        features = vision_model(image)
    
    description = text_generator("This outfit features", max_length=50, num_return_sequences=1)
    return description[0]['generated_text']

def generate_description_from_preferences(preferences):
    color = preferences.get('color', 'neutral')
    style = preferences.get('style', 'casual')
    occasion = preferences.get('occasion', 'everyday')
    
    prompt = f"A {color} {style} outfit suitable for {occasion} occasions. This outfit includes"
    description = text_generator(prompt, max_length=50, num_return_sequences=1)
    return description[0]['generated_text']

def chatbot_interface():
    window = tk.Tk()
    window.title("Fashion Assistant Chatbot")
    
    image_label = None
    try:
        image = Image.open('/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/LOGO_UIE_CUADRADO-01.jpg')
        image = image.resize((200, 150), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(image)
        image_label = tk.Label(window, image=img)
        image_label.image = img
        image_label.grid(row=0, column=1, padx=10, pady=10)
    except Exception as e:
        print(f"Error loading image: {e}")

    conversation_area = scrolledtext.ScrolledText(window, width=90, height=40, wrap=tk.WORD, state=tk.DISABLED)
    conversation_area.grid(row=0, column=0, padx=10, pady=10)

    user_input_box = tk.Entry(window, width=60)
    user_input_box.grid(row=1, column=0, padx=10, pady=10)

    def send_message():
        user_input = user_input_box.get()
        if user_input.strip():
            conversation_area.config(state=tk.NORMAL)
            conversation_area.insert(tk.END, f"You: {user_input}\n")
            conversation_area.yview(tk.END)
            
            wit_data = get_message_wit(user_input)
            response = "I didn't understand that. Can you please rephrase?"
            
            if wit_data[0] == 'describe_outfit_image':
                image_path = wit_data[1].get('image_path', None)
                if image_path:
                    response = f"Description of the outfit: {generate_description_from_image(image_path)}"
                else:
                    response = "Please provide an image path."
            elif wit_data[0] == 'describe_outfit_preferences':
                preferences = wit_data[1].get('preferences', {})
                response = f"Personalized outfit description: {generate_description_from_preferences(preferences)}"
            
            conversation_area.insert(tk.END, f"Chatbot: {response}\n")
            conversation_area.yview(tk.END)
            user_input_box.delete(0, tk.END)

    send_button = tk.Button(window, text="Send", width=10, command=send_message)
    send_button.grid(row=1, column=1, padx=10, pady=10)

    window.mainloop()

chatbot_interface()
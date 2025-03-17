###### PROYECTOB CHATBOT MODA ######
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


# Cargamos los pdf recolectados
files_data = [
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/Dior_english.pdf'
]

# Web scraping
url = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love', 
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson', 
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
]

# Función para hacer web scraping y extraer los textos de los enlaces
def web_scraping(url):
    texts = []
    for link in url:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    return texts

scraped_texts = web_scraping(url)
print(scraped_texts)


###### Preprocesamiento del lenguaje
# Limpieza del texto
# 1. Tokenización con spaCy 
nlp = spacy.load("en_core_news_sm")
documents = files_data + scraped_texts
tokenized_documents = [nlp(document) for document in documents]
print(tokenized_documents)


# 2. Delete stopwords




# 3. Normalization

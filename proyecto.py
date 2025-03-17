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
from collections import Counter

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
print(scraped_texts)



###### PREPROCESAMIENTO DEL TEXTO ######
# Limpieza del texto
# 1. TOKENIZATION - SPACY 
nlp = spacy.load("en_core_news_sm")
documents = files_data + scraped_texts
tokenized_documents = [nlp(document) for document in documents]
print(tokenized_documents)

# 2. DELETE de stopwords
def remove_stopwords(tokenized_docs):
    filtered_docs = []
    for doc in tokenized_docs:
        filtered_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS]
        filtered_docs.append(filtered_tokens)
    return filtered_docs

filtered_documents = remove_stopwords(tokenized_documents)
print(filtered_documents)

# 3. NORMALIZATION 
def normalize_tokens(tokenized_docs):
    normalized_docs = []
    for doc in tokenized_docs:
        normalized_tokens = [token.lemma_.lower() for token in doc if token.lemma_ != '-PRON-']
        normalized_docs.append(normalized_tokens)
    return normalized_docs

normalized_documents = normalize_tokens(tokenized_documents)
print(normalized_documents)

# CREATE A DATAFRAME WITH THE PROCESSED TEXT
df = pd.DataFrame({
    'Original': [' '.join([token.text for token in doc]) for doc in tokenized_documents],
    'Filtered': [' '.join(doc) for doc in filtered_documents],
    'Normalized': [' '.join(doc) for doc in normalized_documents]
})

print(df)

# SAVE RESULTS ON A CSV
df.to_csv('preprocessed_texts.csv', index=False)

# VISUALIZATION
# FREQUENCY OF WORDS IN THE TEXT
all_normalized_tokens = [token for doc in normalized_documents for token in doc]
token_counts = Counter(all_normalized_tokens)

# SHOW THE MOST COMMON WORDS
common_tokens = token_counts.most_common(10)
print(common_tokens)

# GRAPH OF THE MOST COMMON WORDS
tokens, counts = zip(*common_tokens)
plt.figure(figsize=(10, 6))
plt.bar(tokens, counts)
plt.xlabel('Tokens')
plt.ylabel('Frecuencia')
plt.title('Top 10 Tokens más comunes')
plt.show()
###### PROYECTOB CHATBOT MODA ######
# Autora: Carlota Fernández del Riego

# Importamos las librerías necesarias
import os
import PyPDF2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Cargamos los pdf recolectados
files_data = [
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/christianDior.pdf',
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/PMM.pdf',
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/Sistema_Moda.pdf'
]

# Web scraping
url = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love', 
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson', 
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
]

# Función para hacer web scraping y extraer enlaces de una página web
def scrape_links_from_url(url):
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    return links

from rasa_sdk import Action
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import spacy

# Cargar modelos
text_generator = pipeline("text-generation", model="gpt2", framework="pt")
vision_model = models.resnet50(weights='IMAGENET1K_V1')
vision_model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
nlp = spacy.load("en_core_web_sm")

# Funciones de generación de descripciones (ya estaban)
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

# Función para analizar tendencias con BERT
def analyze_trends(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Aquí iría la lógica para extraer info de tendencias (reemplazar con lógica real)
    trend_info = "Tendence information extracted with BERT"
    return trend_info

# Funciones para obtener información de moda (debes implementarlas)
def get_fashion_trends():
    # Implementa la lógica para obtener tendencias de moda
    return "Here are the fashion trends..."

def give_style_suggestion(preferences):
    # Implementa la lógica para dar sugerencias de estilo
    return "Here are the style suggestions..."

def suggest_clothing_combination(item):
    # Implementa la lógica para sugerir combinaciones de ropa
    return "Here are the clothes suggestions..."

def suggest_clothing_color(color):
    # Implementa la lógica para sugerir colores que combinan
    return "Here are the colors that go together..."

def get_clothing_material(item):
    # Implementa la lógica para obtener el material de la prenda
    return "Here goes the garment material..."

def get_clothing_price(item):
    # Implementa la lógica para obtener el precio de la prenda
    return "Here's the price of the garment..."

# Clases de acciones personalizadas
class ActionDescribeOutfitImage(Action):
    def name(self) -> str:
        return "action_describe_outfit_image"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        image_path = tracker.get_slot("image_path")
        description = generate_description_from_image(image_path)
        dispatcher.utter_message(text=f"Description of the outfit: {description}")
        return []

class ActionDescribeOutfitPreferences(Action):
    def name(self) -> str:
        return "action_describe_outfit_preferences"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        preferences = tracker.get_slot("preferences")
        description = generate_description_from_preferences(preferences)
        dispatcher.utter_message(text=f"Personalized outfit description: {description}")
        return []

class ActionGetFashionTrends(Action):
    def name(self) -> str:
        return "action_get_fashion_trends"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        trends = get_fashion_trends()
        dispatcher.utter_message(text=trends)
        return []

class ActionGiveStyleSuggestion(Action):
    def name(self) -> str:
        return "action_give_style_suggestion"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        preferences = tracker.get_slot("preferences")
        suggestion = give_style_suggestion(preferences)
        dispatcher.utter_message(text=suggestion)
        return []

class ActionSuggestClothingCombination(Action):
    def name(self) -> str:
        return "action_suggest_clothing_combination"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        item = tracker.latest_message['text'] # Obtener el texto del último mensaje
        combination = suggest_clothing_combination(item)
        dispatcher.utter_message(text=combination)
        return []

class ActionSuggestClothingColor(Action):
    def name(self) -> str:
        return "action_suggest_clothing_color"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        color = tracker.latest_message['text']
        suggestion = suggest_clothing_color(color)
        dispatcher.utter_message(text=suggestion)
        return []

class ActionGetClothingMaterial(Action):
    def name(self) -> str:
        return "action_get_clothing_material"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        item = tracker.latest_message['text']
        material = get_clothing_material(item)
        dispatcher.utter_message(text=material)
        return []

class ActionGetClothingPrice(Action):
    def name(self) -> str:
        return "action_get_clothing_price"
    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        item = tracker.latest_message['text']
        price = get_clothing_price(item)
        dispatcher.utter_message(text=price)
        return []
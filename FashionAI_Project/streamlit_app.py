import streamlit as st
from PIL import Image
from recommendation_system import get_similar_items
from fashion_classifier import model
from fashion_detection import detect_fashion_items

st.title("Fashion AI Assistant")

uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_column_width=True)

    st.write("Looking for similar clothes...")
    detect_fashion_items(uploaded_file)

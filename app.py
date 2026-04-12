import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("🔍 Image Classification AI")

# Load pretrained model (no tensorflow needed)
classifier = pipeline("image-classification")

def predict(image):
    results = classifier(image)

    top = results[0]
    label = top["label"]
    confidence = top["score"] * 100

    return label, confidence


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    label, confidence = predict(image)

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")

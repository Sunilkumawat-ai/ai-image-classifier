import streamlit as st
from PIL import Image
from transformers import pipeline

# Load model (lightweight)
classifier = pipeline("image-classification")

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier (Offline AI)")
st.write("Upload an image to get AI prediction")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")

    result = classifier(img)

    for res in result[:3]:
        st.write(f"{res['label']} - {round(res['score']*100,2)}%")

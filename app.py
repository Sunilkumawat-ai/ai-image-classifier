import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("🧠 AI Image Classifier (Smart One Answer)")

file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# 🔥 Image Captioning AI (NO LABELS NEEDED)
model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    st.write("### 🔍 Final Answer:")

    result = model(image)

    # ONLY ONE OUTPUT
    answer = result[0]["generated_text"]

    st.success(answer)

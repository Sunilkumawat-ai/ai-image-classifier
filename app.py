import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="AI Image Classifier")

st.title("🧠 AI Image Classifier")

file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

@st.cache_resource
def load_model():
    return pipeline("image-classification")

model = load_model()

if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    st.write("### 🔍 Prediction:")

    result = model(image)

    for r in result[:5]:
        st.write(f"{r['label']} - {round(r['score']*100,2)}%")

import streamlit as st
from PIL import Image
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

headers = {
    "Authorization": "Bearer " + os.getenv("HF_TOKEN")
}

def query(image):
    response = requests.post(API_URL, headers=headers, data=image)
    return response.json()

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier (Real AI)")
st.write("Upload an image to get AI prediction")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    st.image(file, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")

    result = query(file.getvalue())

    if isinstance(result, list):
        for res in result[:3]:
            st.write(f"{res['label']} - {round(res['score']*100,2)}%")
    else:
        st.write("Error:", result)

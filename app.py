import streamlit as st
from PIL import Image
import requests
import os

# Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

headers = {
    "Authorization": "Bearer " + str(os.getenv("HF_TOKEN"))
}

# Safe API call
def query(image):
    try:
        response = requests.post(API_URL, headers=headers, data=image, timeout=30)
        
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        
        return response.json()
    
    except requests.exceptions.RequestException:
        return {"error": "Network/API error. Try again."}


# UI
st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier (Real AI)")
st.write("Upload an image and get AI prediction")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")

    result = query(file.getvalue())

    if isinstance(result, list):
        for res in result[:3]:
            st.write(f"{res['label']} - {round(res['score']*100,2)}%")
    elif isinstance(result, dict) and "error" in result:
        st.error(result["error"])
    else:
        st.write(result)

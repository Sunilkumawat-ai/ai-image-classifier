import streamlit as st
import requests

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier (Real AI)")
st.write("Upload an image to detect objects (dog, cat, etc.)")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    st.image(file, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔍 Prediction:")

    API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

    response = requests.post(API_URL, data=file.getvalue())

    try:
        result = response.json()

        if isinstance(result, list):
            for obj in result[:3]:
                label = obj.get("label", "Unknown")
                score = obj.get("score", 0)
                st.write(f"{label} - {round(score*100,2)}%")
        else:
            st.error("AI couldn't process image. Try another image.")

    except:
        st.error("Error connecting to AI. Try again.")

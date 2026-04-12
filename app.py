import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("🧠 AI Image Understanding")

file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# ✔️ correct pipeline task
model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    st.write("### 🔍 Final Answer:")

    result = model(image)

    st.success(result[0]["generated_text"])

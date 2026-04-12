import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image to analyze brightness")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)
    avg = img_array.mean()

    st.write("### 🔍 Result:")

    if avg > 150:
        st.success("🌞 Bright Image")
    else:
        st.info("🌑 Dark Image")

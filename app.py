import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI Image Classifier", layout="centered")

st.title("🧠 AI Image Classifier")
st.write("Upload an image to analyze brightness and color")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)

    avg_brightness = img_array.mean()

    st.write("### 🔍 Result:")

    if avg_brightness > 150:
        st.success("🌞 Bright Image (Day / Outdoor)")
    else:
        st.info("🌑 Dark Image (Night / Low Light)")
